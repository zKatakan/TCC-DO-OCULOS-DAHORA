#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
import os

# classes do modelo treinado: 0=pedestrian, 1=car
CLASS_NAMES = ["pedestrian", "car"]

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

def parse_outputs(raw, imgsz):
    """
    raw pode vir como:
      (1, nc+4, 8400)  -> squeeze -> (nc+4, 8400) -> transpose -> (8400, nc+4)
      (1, 8400, nc+4)  -> squeeze -> (8400, nc+4)
    vamos devolver (8400, nc+4)
    """
    out = np.squeeze(raw)
    # se ainda tiver formato (C, N), invertemos
    if out.ndim == 2 and out.shape[0] < out.shape[1]:
        # ex: (6, 8400) -> (8400, 6)
        out = out.transpose(1, 0)
    return out  # (8400, nc+4)

def run_inference(net, frame, imgsz, conf_thres, iou_thres):
    inp, r, (dx, dy) = letterbox(frame, imgsz)
    blob = cv2.dnn.blobFromImage(
        inp, 1/255.0, (imgsz, imgsz),
        swapRB=True, crop=False
    )
    net.setInput(blob)

    t0 = time.time()
    raw = net.forward()
    infer_ms = (time.time() - t0) * 1000.0

    out = parse_outputs(raw, imgsz)  # (N, 6) se forem 2 classes -> 4+2

    boxes = []
    scores = []
    class_ids = []

    for det in out:
        # det = [cx, cy, w, h, cls0, cls1]  (supondo 2 classes)
        if det.shape[0] < 6:
            # algo estranho no onnx
            continue

        cx, cy, bw, bh = det[0:4]
        cls_scores = det[4:]

        cid = int(np.argmax(cls_scores))
        score = float(cls_scores[cid])
        if score < conf_thres:
            continue

        # converter pro espaço original
        x1 = (cx - bw/2 - dx) / r
        y1 = (cy - bh/2 - dy) / r
        x2 = (cx + bw/2 - dx) / r
        y2 = (cy + bh/2 - dy) / r

        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        scores.append(score)
        class_ids.append(cid)

    # NMS
    nms_boxes = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
    idxs = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_thres, iou_thres)

    final_boxes, final_scores, final_cids = [], [], []
    if len(idxs) > 0:
        for i in idxs.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            final_cids.append(class_ids[i])

    return final_boxes, final_scores, final_cids, infer_ms

def draw(frame, boxes, scores, cids, infer_ms):
    for (box, sc, cid) in zip(boxes, scores, cids):
        x1, y1, x2, y2 = box
        name = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)

        if name == "pedestrian":
            color = (0, 255, 0)
        else:
            color = (0, 200, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {sc:.2f}", (x1, max(15, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"{infer_ms:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="best.onnx", help="modelo ONNX exportado")
    ap.add_argument("--source", type=str, default="0", help="0 webcam ou caminho de vídeo")
    ap.add_argument("--imgsz", type=int, default=640, help="tamanho da entrada")
    ap.add_argument("--conf", type=float, default=0.45, help="conf mínima")
    ap.add_argument("--iou", type=float, default=0.6, help="IoU NMS")
    ap.add_argument("--skip", type=int, default=1, help="processa 1 a cada N frames")
    ap.add_argument("--out", type=str, default="saida_people_cars.mp4", help="vídeo de saída")
    ap.add_argument("--show", action="store_true", help="mostrar janela")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        print(f"[ERRO] modelo não encontrado: {args.onnx}")
        return

    net = cv2.dnn.readNetFromONNX(args.onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERRO] não abriu {args.source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    print(f"[INFO] salvando em: {args.out}")

    frame_id = 0
    last_boxes, last_scores, last_cids, last_ms = [], [], [], 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        do_infer = (frame_id % args.skip == 0)

        if do_infer:
            boxes, scores, cids, infer_ms = run_inference(
                net, frame, args.imgsz, args.conf, args.iou
            )
            last_boxes, last_scores, last_cids, last_ms = boxes, scores, cids, infer_ms
        else:
            boxes, scores, cids, infer_ms = last_boxes, last_scores, last_cids, last_ms

        draw(frame, boxes, scores, cids, infer_ms)

        if args.show:
            cv2.imshow("people+cars ONNX", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

        writer.write(frame)

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print("[INFO] finalizado.")

if __name__ == "__main__":
    main()
