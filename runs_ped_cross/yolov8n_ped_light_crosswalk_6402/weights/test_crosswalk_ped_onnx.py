#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
import os

# ordem padrão (ajuste se seu ONNX saiu diferente)
DEFAULT_CLASS_NAMES = ["crosswalk", "ped_light_go", "ped_light_stop", "ped_light_off"]


def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_h // 2)
    # opa, corrigindo: pad_w é pra esquerda/direita
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (left, top)


def run_inference(net, frame, imgsz, conf_thres, iou_thres, class_names):
    """
    Faz inferência e aplica NMS por classe, pra permitir várias boxes no mesmo frame.
    """
    inp, r, (dx, dy) = letterbox(frame, imgsz)
    blob = cv2.dnn.blobFromImage(inp, 1/255.0, (imgsz, imgsz), swapRB=True, crop=False)
    net.setInput(blob)

    t0 = time.time()
    raw = net.forward()
    infer_ms = (time.time() - t0) * 1000.0

    out = np.squeeze(raw)  # (C+4, N) ou (N, C+4)
    # se estiver no formato (num_feats, N), transpoe
    if out.shape[0] == len(class_names) + 4:
        out = out.T  # -> (N, 4 + num_classes)

    # vamos separar por classe
    per_class_boxes = {i: [] for i in range(len(class_names))}
    per_class_scores = {i: [] for i in range(len(class_names))}

    for det in out:
        cx, cy, w_box, h_box = det[0:4]
        cls_scores = det[4:]
        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id])
        if score < conf_thres:
            continue

        # converte pra coords originais
        x1 = (cx - w_box/2 - dx) / r
        y1 = (cy - h_box/2 - dy) / r
        x2 = (cx + w_box/2 - dx) / r
        y2 = (cy + h_box/2 - dy) / r

        per_class_boxes[cls_id].append([int(x1), int(y1), int(x2), int(y2)])
        per_class_scores[cls_id].append(score)

    final_boxes = []
    final_scores = []
    final_class_ids = []

    # NMS classe a classe
    for cid in range(len(class_names)):
        boxes_c = per_class_boxes[cid]
        scores_c = per_class_scores[cid]
        if not boxes_c:
            continue

        nms_boxes = []
        for (x1, y1, x2, y2) in boxes_c:
            nms_boxes.append([x1, y1, int(x2 - x1), int(y2 - y1)])

        idxs = cv2.dnn.NMSBoxes(nms_boxes, scores_c, conf_thres, iou_thres)

        if len(idxs) > 0:
            for i in idxs.flatten():
                final_boxes.append(boxes_c[i])
                final_scores.append(scores_c[i])
                final_class_ids.append(cid)

    return final_boxes, final_scores, final_class_ids, infer_ms


def draw_boxes(frame, boxes, scores, class_ids, infer_ms, class_names):
    for (box, score, cid) in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        if 0 <= cid < len(class_names):
            cls_name = class_names[cid]
        else:
            cls_name = str(cid)

        # cores
        if cls_name == "crosswalk":
            color = (255, 0, 255)
        elif "go" in cls_name:
            color = (0, 255, 0)
        elif "stop" in cls_name or "red" in cls_name:
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls_name} {score:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"{infer_ms:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="best.onnx", help="modelo ONNX exportado")
    ap.add_argument("--source", type=str, default="0", help="0 para webcam ou caminho de vídeo .mp4")
    ap.add_argument("--imgsz", type=int, default=640, help="tamanho da imagem de entrada")
    ap.add_argument("--conf", type=float, default=0.5, help="conf mínima")
    ap.add_argument("--iou", type=float, default=0.6, help="IoU p/ NMS")
    ap.add_argument("--skip", type=int, default=1, help="processa 1 a cada N frames")
    ap.add_argument("--out", type=str, default="saida_crosswalk_ped.mp4", help="vídeo anotado de saída")
    ap.add_argument("--nowindow", action="store_true", help="não mostrar janela")
    ap.add_argument("--class-order", type=str,
                    default="crosswalk,ped_light_go,ped_light_stop,ped_light_off",
                    help="ordem real das classes do ONNX")
    args = ap.parse_args()

    # classes
    class_names = [c.strip() for c in args.class_order.split(",")]
    if len(class_names) < 2:
        class_names = DEFAULT_CLASS_NAMES

    if not os.path.exists(args.onnx):
        print(f"[ERRO] modelo ONNX não encontrado: {args.onnx}")
        return

    net = cv2.dnn.readNetFromONNX(args.onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERRO] não abriu fonte: {args.source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = cv2.VideoWriter(args.out,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_in, (w, h))
    print(f"[INFO] gravando saída em: {args.out}")
    print(f"[INFO] classes: {class_names}")

    frame_idx = 0
    last_boxes, last_scores, last_class_ids = [], [], []
    last_infer_ms = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        run_now = (frame_idx % args.skip == 0)

        if run_now:
            boxes, scores, cids, infer_ms = run_inference(
                net, frame, args.imgsz, args.conf, args.iou, class_names
            )
            last_boxes, last_scores, last_class_ids = boxes, scores, cids
            last_infer_ms = infer_ms
        else:
            boxes, scores, cids = last_boxes, last_scores, last_class_ids
            infer_ms = last_infer_ms

        draw_boxes(frame, boxes, scores, cids, infer_ms, class_names)

        if not args.nowindow:
            cv2.imshow("Crosswalk + Pedestrian Light (ONNX)", frame)
            if cv2.waitKey(1) == 27:
                break

        writer.write(frame)

    cap.release()
    writer.release()
    if not args.nowindow:
        cv2.destroyAllWindows()
    print("[INFO] finalizado.")


if __name__ == "__main__":
    main()
