#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
import os

# classes do modelo de semáforo de pedestre
CLASS_NAMES = ["ped_light_go", "ped_light_off", "ped_light_red"]

def letterbox(img, new_shape=640):
    """Redimensiona mantendo proporção com padding estilo YOLO."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114,114,114)
    )
    return im, r, (left, top)

def run_inference(net, frame, imgsz, conf_thres, iou_thres):
    """
    Faz inferência 1 vez usando o modelo ONNX (OpenCV DNN),
    parseia a saída no formato (1, 7, 8400) -> (8400, 7),
    aplica conf e NMS,
    retorna boxes, scores, class_ids e tempo em ms.
    """
    inp, r, (dx, dy) = letterbox(frame, imgsz)
    blob = cv2.dnn.blobFromImage(
        inp,
        1/255.0,
        (imgsz, imgsz),
        swapRB=True,
        crop=False
    )
    net.setInput(blob)

    t0 = time.time()
    raw = net.forward()
    infer_ms = (time.time() - t0) * 1000.0

    # raw esperado: (1, 7, 8400)
    raw = np.squeeze(raw)          # -> (7, 8400) ou (8400, 7)
    if raw.shape[0] == 7:
        raw = raw.transpose(1, 0)  # -> (8400, 7)

    boxes = []
    scores = []
    class_ids = []

    # Cada linha: [cx, cy, w, h, score_cls0, score_cls1, score_cls2]
    for det in raw:
        cx, cy, w_box, h_box = det[0:4]
        cls_scores = det[4:]
        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id])

        if score < conf_thres:
            continue

        # Converter cx,cy,w,h -> x1,y1,x2,y2 no frame original
        x1 = (cx - w_box/2 - dx) / r
        y1 = (cy - h_box/2 - dy) / r
        x2 = (cx + w_box/2 - dx) / r
        y2 = (cy + h_box/2 - dy) / r

        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        scores.append(score)
        class_ids.append(cls_id)

    # NMS
    nms_boxes = []
    for (x1, y1, x2, y2) in boxes:
        bw = int(x2 - x1)
        bh = int(y2 - y1)
        nms_boxes.append([x1, y1, bw, bh])

    idxs = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_thres, iou_thres)

    final_boxes = []
    final_scores = []
    final_class_ids = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])

    return final_boxes, final_scores, final_class_ids, infer_ms

def draw_boxes(frame, boxes, scores, class_ids, infer_ms):
    """
    Desenha bounding boxes e infos no frame.
    """
    for (box, score, cid) in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        cls_name = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)

        # cor por classe
        if "go" in cls_name:
            color = (0, 255, 0)      # verde
        elif "red" in cls_name:
            color = (0, 0, 255)      # vermelho
        else:
            color = (255, 255, 0)    # amarelo/alerta (off)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{cls_name} {score:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.putText(
        frame,
        f"{infer_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

def summarize_state(boxes, scores, class_ids):
    """
    Escolhe a detecção mais confiante e retorna rótulo do estado do semáforo.
    """
    if not boxes:
        return None
    best_i = int(max(range(len(scores)), key=lambda k: scores[k]))
    cid = class_ids[best_i]
    if 0 <= cid < len(CLASS_NAMES):
        return CLASS_NAMES[cid]
    return str(cid)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--onnx", type=str, default="best.onnx",
                    help="modelo ONNX exportado (ex: best.onnx)")
    ap.add_argument("--source", type=str, required=True,
                    help="caminho de vídeo .mp4 para testar (ou 0 para webcam)")
    ap.add_argument("--imgsz", type=int, default=576,
                    help="tamanho da imagem passada pro modelo (576 recomendado na Pi)")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="limiar de confiança mínima (0.5/0.6 reduz falso positivo)")
    ap.add_argument("--iou", type=float, default=0.6,
                    help="limiar IOU para NMS")
    ap.add_argument("--skip", type=int, default=2,
                    help="roda inferência a cada N frames (2 = metade dos frames)")
    ap.add_argument("--out", type=str, default="saida_video_otimizado.mp4",
                    help="vídeo anotado de saída")
    ap.add_argument("--nowindow", action="store_true",
                    help="se setado, NÃO mostra janela durante o processamento (só salva o vídeo final)")
    args = ap.parse_args()

    # valida modelo
    if not os.path.exists(args.onnx):
        print(f"[ERRO] Modelo ONNX não encontrado: {args.onnx}")
        return

    # carrega modelo
    net = cv2.dnn.readNetFromONNX(args.onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # abre vídeo
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERRO] Erro ao abrir fonte: {args.source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    out_writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (w, h)
    )
    print(f"[INFO] Gravando saída anotada em: {args.out}")

    frame_idx = 0

    # buffers pra reaproveitar detecção entre frames
    last_boxes, last_scores, last_class_ids = [], [], []
    last_infer_ms = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        run_now = (frame_idx % args.skip == 0)

        if run_now:
            boxes, scores, class_ids, infer_ms = run_inference(
                net,
                frame,
                args.imgsz,
                args.conf,
                args.iou
            )
            last_boxes, last_scores, last_class_ids = boxes, scores, class_ids
            last_infer_ms = infer_ms
        else:
            boxes, scores, class_ids = last_boxes, last_scores, last_class_ids
            infer_ms = last_infer_ms

        # estado textual (debug útil pra console / futura saída de áudio)
        state = summarize_state(boxes, scores, class_ids)
        if state is not None:
            print(f"[FRAME {frame_idx}] ESTADO DETECTADO: {state}")
        else:
            print(f"[FRAME {frame_idx}] Nenhum semáforo detectado")

        # desenha caixas e info de tempo
        draw_boxes(frame, boxes, scores, class_ids, infer_ms)

        # mostra janela (a não ser que esteja em modo nowindow)
        if not args.nowindow:
            cv2.imshow("Pedestrian Light Detector (Video Test / ONNX+skip)", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

        # escreve frame anotado no arquivo final
        out_writer.write(frame)

        frame_idx += 1

    cap.release()
    out_writer.release()
    if not args.nowindow:
        cv2.destroyAllWindows()

    print("[INFO] Finalizado. Vídeo anotado salvo em", args.out)

if __name__ == "__main__":
    main()
