#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
import os
import subprocess
from collections import deque

def play_audio(mp3_path, label, mute=False):
    if mute:
        print(f"[AUDIO muted] {label}")
        return
    if not mp3_path or not os.path.exists(mp3_path):
        print(f"[WARN] áudio não encontrado: {mp3_path} ({label})")
        return
    try:
        # rasp: mpg123
        subprocess.Popen(["mpg123", "-q", mp3_path])
    except Exception as e:
        print(f"[WARN] falha tocando {mp3_path}: {e}")

def letterbox(img, new_shape):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return im, r, (left, top)

def run_inference(net, frame, imgsz, conf_thres, iou_thres):
    inp, r, (dx, dy) = letterbox(frame, imgsz)
    blob = cv2.dnn.blobFromImage(inp, 1/255.0, (imgsz, imgsz), swapRB=True, crop=False)
    net.setInput(blob)

    t0 = time.time()
    raw = net.forward()
    infer_ms = (time.time() - t0) * 1000.0

    out = np.squeeze(raw)
    # garantir que fique (N, 4+cls)
    if out.ndim == 2 and out.shape[0] < out.shape[1]:
        out = out.transpose(1, 0)

    boxes, scores, cids = [], [], []

    for det in out:
        cx, cy, w_box, h_box = det[0:4]
        cls_scores = det[4:]
        cid = int(np.argmax(cls_scores))
        score = float(cls_scores[cid])
        if score < conf_thres:
            continue

        x1 = (cx - w_box/2 - dx) / r
        y1 = (cy - h_box/2 - dy) / r
        x2 = (cx + w_box/2 - dx) / r
        y2 = (cy + h_box/2 - dy) / r

        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        scores.append(score)
        cids.append(cid)

    # NMS
    nms_boxes = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
    idxs = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_thres, iou_thres)

    f_boxes, f_scores, f_cids = [], [], []
    if len(idxs) > 0:
        for i in idxs.flatten():
            f_boxes.append(boxes[i])
            f_scores.append(scores[i])
            f_cids.append(cids[i])

    return f_boxes, f_scores, f_cids, infer_ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--skip", type=int, default=5, help="roda inferência a cada N frames")
    ap.add_argument("--show", action="store_true")
    # ordem real do seu modelo:
    # 0 crosswalk, 1 ped_light_go, 2 ped_light_stop, 3 ped_light_off
    ap.add_argument("--class-names", type=str,
                    default="crosswalk,ped_light_go,ped_light_stop,ped_light_off")
    ap.add_argument("--cooldown", type=float, default=2.0,
                    help="tempo mínimo entre duas falas iguais (s)")
    # nomes dos seus áudios (iguais aos que você mostrou)
    ap.add_argument("--audio-go", type=str, default="audio_go.mp3")
    ap.add_argument("--audio-stop", type=str, default="audio_stop.mp3")
    ap.add_argument("--audio-off", type=str, default="audio_off.mp3")
    ap.add_argument("--audio-cross", type=str, default="audio_crosswalk.mp3")
    ap.add_argument("--noaudio", action="store_true")
    args = ap.parse_args()

    class_names = [c.strip() for c in args.class_names.split(",")]
    print("[INFO] classes do modelo:", class_names)

    if not os.path.exists(args.onnx):
        print("[ERRO] modelo não encontrado:", args.onnx)
        return

    net = cv2.dnn.readNetFromONNX(args.onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERRO] não abriu fonte:", args.source)
        return

    history = deque(maxlen=10)
    last_spoken_time = 0.0

    # buffers pra NÃO piscar
    last_boxes, last_scores, last_cids, last_infer_ms = [], [], [], 0.0

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        run_now = (frame_id % args.skip == 0)

        if run_now:
            boxes, scores, cids, infer_ms = run_inference(
                net, frame, args.imgsz, args.conf, args.iou
            )
            # atualiza buffer
            last_boxes, last_scores, last_cids, last_infer_ms = boxes, scores, cids, infer_ms
        else:
            # usa o último resultado pra não piscar
            boxes, scores, cids, infer_ms = last_boxes, last_scores, last_cids, last_infer_ms

        # ver classes vistas
        seen = set()
        for cid in cids:
            if 0 <= cid < len(class_names):
                seen.add(class_names[cid])

        # prioridade
        if "crosswalk" in seen:
            current_state = "crosswalk"
        elif "ped_light_stop" in seen:
            current_state = "ped_light_stop"
        elif "ped_light_go" in seen:
            current_state = "ped_light_go"
        elif "ped_light_off" in seen:
            current_state = "ped_light_off"
        else:
            current_state = "none"

        history.append(current_state)
        # estado estável
        if history:
            counts = {}
            for s in history:
                counts[s] = counts.get(s, 0) + 1
            stable_state = max(counts, key=counts.get)
        else:
            stable_state = "none"

        now = time.time()
        if stable_state != "none" and (now - last_spoken_time) > args.cooldown:
            if stable_state == "ped_light_go":
                play_audio(args.audio_go, "semáforo aberto", mute=args.noaudio)
            elif stable_state == "ped_light_stop":
                play_audio(args.audio_stop, "aguarde, sinal vermelho", mute=args.noaudio)
            elif stable_state == "ped_light_off":
                play_audio(args.audio_off, "semáforo não identificado", mute=args.noaudio)
            elif stable_state == "crosswalk":
                play_audio(args.audio_cross, "faixa de pedestre detectada", mute=args.noaudio)
            last_spoken_time = now

        if args.show:
            for (box, score, cid) in zip(boxes, scores, cids):
                x1, y1, x2, y2 = box
                name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
                if name == "ped_light_go":
                    color = (0, 255, 0)
                elif name == "ped_light_stop":
                    color = (0, 0, 255)
                elif name == "crosswalk":
                    color = (255, 255, 0)
                else:
                    color = (255, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {score:.2f}", (x1, max(15, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(frame, f"stable: {stable_state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("Semáforo + Faixa (RPi)", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    print("[INFO] finalizado.")

if __name__ == "__main__":
    main()
