#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
import os

# classes do modelo de semáforo de pedestre
CLASS_NAMES = ["ped_light_go", "ped_light_off", "ped_light_red"]

def letterbox(img, new_shape=640):
    """Redimensiona mantendo proporção, com padding estilo YOLO."""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="best.onnx", help="modelo ONNX exportado")
    ap.add_argument("--source", type=str, default="0", help="0 para webcam ou caminho de vídeo .mp4")
    ap.add_argument("--imgsz", type=int, default=640, help="tamanho da imagem de entrada (treino)")
    ap.add_argument("--conf", type=float, default=0.45, help="limiar de confiança mínima")
    ap.add_argument("--iou", type=float, default=0.6, help="limiar IOU para NMS")
    ap.add_argument("--out", type=str, default="saida_infer_onnx.mp4", help="vídeo anotado de saída")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        print(f"Modelo ONNX não encontrado: {args.onnx}")
        return

    # carrega modelo ONNX
    net = cv2.dnn.readNetFromONNX(args.onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # abre fonte de vídeo
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Erro ao abrir {args.source}")
        return

    # configura gravação de vídeo
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out_writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    print(f"Gravando saída em: {args.out}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # pré-processamento (letterbox + blob)
        inp, r, (dx, dy) = letterbox(frame, args.imgsz)
        blob = cv2.dnn.blobFromImage(
            inp,
            1/255.0,
            (args.imgsz, args.imgsz),
            swapRB=True,
            crop=False
        )
        net.setInput(blob)

        # inferência
        t0 = time.time()
        raw = net.forward()
        infer_ms = (time.time() - t0) * 1000.0

        # raw tem shape (1, 7, 8400)
        # vamos reorganizar para (8400, 7)
        raw = np.squeeze(raw)          # (7, 8400) ou (1,7,8400)->(7,8400)
        if raw.shape[0] == 7:
            raw = raw.transpose(1, 0)  # (8400, 7)

        # agora cada linha é [cx, cy, w, h, score_cls0, score_cls1, score_cls2]
        boxes = []
        scores = []
        class_ids = []

        for det in raw:
            cx, cy, w_box, h_box = det[0:4]

            # as últimas colunas são as probabilidades por classe
            cls_scores = det[4:]
            cls_id = int(np.argmax(cls_scores))
            score = float(cls_scores[cls_id])

            # filtra por confiança
            if score < args.conf:
                continue

            # converte cx,cy,w,h (formato YOLO) -> x1,y1,x2,y2 no espaço ORIGINAL do frame
            x1 = (cx - w_box/2 - dx) / r
            y1 = (cy - h_box/2 - dy) / r
            x2 = (cx + w_box/2 - dx) / r
            y2 = (cy + h_box/2 - dy) / r

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(cls_id)

        # aplica NMS
        # NMSBoxes espera [x,y,w,h]
        nms_boxes = []
        for b in boxes:
            bx = int(b[0])
            by = int(b[1])
            bw = int(b[2] - b[0])
            bh = int(b[3] - b[1])
            nms_boxes.append([bx, by, bw, bh])

        idxs = cv2.dnn.NMSBoxes(nms_boxes, scores, args.conf, args.iou)

        # desenha caixas aprovadas no frame
        if len(idxs) > 0:
            for i in idxs.flatten():
                x1, y1, x2, y2 = boxes[i]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                cls_name = CLASS_NAMES[class_ids[i]] if 0 <= class_ids[i] < len(CLASS_NAMES) else str(class_ids[i])

                # cor diferente pra ir / parar / desligado
                if "go" in cls_name:
                    color = (0, 255, 0)      # verde
                elif "red" in cls_name:
                    color = (0, 0, 255)      # vermelho
                else:
                    color = (255, 255, 0)    # amarelo/alerta

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{cls_name} {scores[i]:.2f}",
                    (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # info de latência
        cv2.putText(
            frame,
            f"{infer_ms:.1f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # mostra na tela
        cv2.imshow("Pedestrian Light Detector (ONNX)", frame)

        # salva o frame anotado
        out_writer.write(frame)

        # ESC pra sair
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print("Inferência finalizada.")

if __name__ == "__main__":
    main()
