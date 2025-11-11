#!/usr/bin/env python3
import cv2
import numpy as np
import time
import argparse
import os
import subprocess

# ======================================================
# === Função para tocar áudio via mpg123 (Linux/RPi) ===
# ======================================================
def play_audio(mp3_path, label="", mute=False):
    if mute:
        print(f"[AUDIO muted] {label}")
        return
    if not os.path.exists(mp3_path):
        print(f"[WARN] áudio não encontrado: {mp3_path}")
        return
    print(f"[AUDIO] {label}")
    try:
        subprocess.Popen(["mpg123", "-q", mp3_path])
    except Exception as e:
        print(f"[WARN] Falha ao tocar áudio: {e}")

# ======================================================
def letterbox(img, new_shape=640):
    """Redimensiona mantendo proporção (como o YOLO)."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=(114,114,114))
    return im, r, (left, top)

# ======================================================
def run_inference(net, frame, imgsz, conf_thres, iou_thres):
    inp, r, (dx, dy) = letterbox(frame, imgsz)
    blob = cv2.dnn.blobFromImage(inp, 1/255.0, (imgsz, imgsz), swapRB=True, crop=False)
    net.setInput(blob)

    t0 = time.time()
    raw = net.forward()
    infer_ms = (time.time() - t0) * 1000.0

    out = np.squeeze(raw)
    if out.ndim == 2 and out.shape[0] < out.shape[1]:
        out = out.T

    boxes, scores, cids = [], [], []
    for det in out:
        if len(det) < 5:
            continue
        cx, cy, bw, bh = det[:4]
        cls_scores = det[4:]
        cid = int(np.argmax(cls_scores))
        score = float(cls_scores[cid])
        if score < conf_thres:
            continue
        x1 = (cx - bw/2 - dx) / r
        y1 = (cy - bh/2 - dy) / r
        x2 = (cx + bw/2 - dx) / r
        y2 = (cy + bh/2 - dy) / r
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        scores.append(score)
        cids.append(cid)

    # NMS (Non-Maximum Suppression)
    nms_boxes = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
    idxs = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_thres, iou_thres)

    fb, fs, fc = [], [], []
    if len(idxs) > 0:
        for i in idxs.flatten():
            fb.append(boxes[i])
            fs.append(scores[i])
            fc.append(cids[i])
    return fb, fs, fc, infer_ms

# ======================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--class-names", type=str, default="pedestrian,car")
    ap.add_argument("--cooldown", type=float, default=2.0, help="Tempo mínimo entre sons (segundos)")
    ap.add_argument("--ttl", type=int, default=15)
    ap.add_argument("--max-ped", type=int, default=5)
    ap.add_argument("--max-veh", type=int, default=3)
    ap.add_argument("--audio-ped", type=str, default="audio_pedestrian.mp3")
    ap.add_argument("--audio-veh", type=str, default="audio_vehicle.mp3")
    args = ap.parse_args()

    class_names = [c.strip() for c in args.class_names.split(",")]
    print(f"[INFO] classes ONNX: {class_names}")
    print(f"[INFO] cooldown: {args.cooldown:.1f}s entre áudios")

    if not os.path.exists(args.onnx):
        print(f"[ERRO] modelo não encontrado: {args.onnx}")
        return

    net = cv2.dnn.readNetFromONNX(args.onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERRO] não abriu fonte: {args.source}")
        return

    memory = {}
    last_audio = {"ped": 0, "veh": 0}
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        if frame_id % args.skip != 0:
            continue

        boxes, scores, cids, infer_ms = run_inference(net, frame, args.imgsz, args.conf, args.iou)
        now = time.time()

        for box, sc, cid in zip(boxes, scores, cids):
            name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            if "ped" in name and (now - last_audio["ped"]) > args.cooldown:
                play_audio(args.audio_ped, "pedestre detectado")
                last_audio["ped"] = now
            elif ("car" in name or "veh" in name) and (now - last_audio["veh"]) > args.cooldown:
                play_audio(args.audio_veh, "veículo detectado")
                last_audio["veh"] = now

            x1, y1, x2, y2 = box
            color = (0,255,0) if "ped" in name else (0,180,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{name} {sc:.2f}", (x1, max(15,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if args.show:
            cv2.imshow("Raspberry Pi 5 - Pessoas & Veículos", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] finalizado.")

# ======================================================
if __name__ == "__main__":
    main()
