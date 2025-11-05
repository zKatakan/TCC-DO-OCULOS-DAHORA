#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
import os
import collections
from playsound import playsound  # no Windows: pip install playsound==1.2.2

# ===== util =====
def parse_class_order(order_str: str):
    """
    Converte algo como 'go,red,off' para
    ['ped_light_go','ped_light_red','ped_light_off'].
    """
    tokens = [t.strip().lower() for t in order_str.split(",")]
    valid = {"go": "ped_light_go", "red": "ped_light_red", "off": "ped_light_off"}
    if len(tokens) != 3 or any(t not in valid for t in tokens):
        raise ValueError("Use --class-order com exatamente 3 itens (go,red,off) em alguma ordem. Ex: go,red,off")
    return [valid[t] for t in tokens]

def play_audio(mp3_path, label, mute=False):
    if mute:
        print(f"[AUDIO muted] {label} -> {mp3_path}")
        return
    if not mp3_path or not os.path.exists(mp3_path):
        print(f"[WARN] áudio não encontrado: {mp3_path} ({label})")
        return
    try:
        print(f"[AUDIO] {label}")
        playsound(mp3_path, block=False)
    except Exception as e:
        print(f"[WARN] Falha ao tocar {mp3_path}: {e}")

def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return im, r, (left, top)

# ===== inferência (mantida igual ao seu parser que funciona) =====
def run_inference(net, frame, imgsz, conf_thres, iou_thres):
    inp, r, (dx, dy) = letterbox(frame, imgsz)
    blob = cv2.dnn.blobFromImage(inp, 1/255.0, (imgsz, imgsz), swapRB=True, crop=False)
    net.setInput(blob)

    t0 = time.time()
    raw = net.forward()
    infer_ms = (time.time() - t0) * 1000.0

    raw = np.squeeze(raw)            # -> (7, 8400) ou (8400, 7)
    if raw.shape[0] == 7:
        raw = raw.transpose(1, 0)    # -> (8400, 7)

    boxes, scores, class_ids = [], [], []

    # Cada linha: [cx, cy, w, h, score_cls0, score_cls1, score_cls2]
    for det in raw:
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
        class_ids.append(cid)

    # NMS
    nms_boxes = [[x1, y1, int(x2-x1), int(y2-y1)] for (x1,y1,x2,y2) in boxes]
    idxs = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_thres, iou_thres)

    final_boxes, final_scores, final_class_ids = [], [], []
    if len(idxs) > 0:
        for i in idxs.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])

    return final_boxes, final_scores, final_class_ids, infer_ms

def draw_boxes(frame, boxes, scores, class_ids, infer_ms, stable_state, class_names):
    for (box, score, cid) in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        cls_name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        if "go" in cls_name:   color = (0,255,0)
        elif "red" in cls_name:color = (0,0,255)
        else:                  color = (255,255,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{cls_name} {score:.2f}", (x1, max(15,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"{infer_ms:.1f} ms", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"stable: {stable_state}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

# ===== mapeamento por ÍNDICE (robusto à ordem) =====
def state_from_frame_idx(class_ids, scores, idx_go, idx_red, idx_off):
    if not class_ids:
        return "none"
    best_i = int(max(range(len(scores)), key=lambda k: scores[k]))
    cid = class_ids[best_i]
    if cid == idx_red: return "red"
    if cid == idx_go:  return "go"
    if cid == idx_off: return "off"
    return "none"

def pick_stable_state(history_deque):
    if not history_deque:
        return "none"
    counts = {}
    for st in history_deque:
        counts[st] = counts.get(st, 0) + 1
    return max(counts, key=counts.get)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="best.onnx", help="modelo ONNX (ex: best.onnx)")
    ap.add_argument("--source", type=str, required=True, help="vídeo .mp4 ou 0 para webcam")
    ap.add_argument("--imgsz", type=int, default=576, help="tem que bater com o export do ONNX")
    ap.add_argument("--conf", type=float, default=0.5, help="conf mínima")
    ap.add_argument("--iou", type=float, default=0.6, help="IoU NMS")
    ap.add_argument("--skip", type=int, default=2, help="processa 1 a cada N frames")
    ap.add_argument("--out", type=str, default="saida_video_otimizado.mp4", help="vídeo anotado de saída")
    ap.add_argument("--nowindow", action="store_true", help="não mostra janela")
    # acessibilidade
    ap.add_argument("--history", type=int, default=15, help="tamanho da janela de suavização")
    ap.add_argument("--cooldown", type=float, default=2.0, help="mínimo em segundos entre falas")
    ap.add_argument("--audio_go", type=str, default="audio_go.mp3")
    ap.add_argument("--audio_red", type=str, default="audio_stop.mp3")
    ap.add_argument("--audio_off", type=str, default="audio_off.mp3")
    ap.add_argument("--noaudio", action="store_true", help="não toca áudio de verdade")
    # NOVO: ordem real das classes no ONNX
    ap.add_argument("--class-order", type=str, default="go,off,red",
                    help="ordem real das classes no ONNX (go,red,off em alguma ordem)")
    args = ap.parse_args()

    # resolve nomes e índices da ordem informada
    CLASS_NAMES = parse_class_order(args.class_order)
    idx_go  = CLASS_NAMES.index("ped_light_go")
    idx_red = CLASS_NAMES.index("ped_light_red")
    idx_off = CLASS_NAMES.index("ped_light_off")
    print(f"[INFO] CLASS ORDER: {CLASS_NAMES} (go={idx_go}, red={idx_red}, off={idx_off})")

    # valida modelo
    if not os.path.exists(args.onnx):
        print(f"[ERRO] Modelo ONNX não encontrado: {args.onnx}")
        return

    # carrega modelo
    net = cv2.dnn.readNetFromONNX(args.onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # abre vídeo/câmera
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERRO] Erro ao abrir fonte: {args.source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
    print(f"[INFO] Gravando saída anotada em: {args.out}")

    # buffers/estado
    frame_idx = 0
    last_boxes = last_scores = last_class_ids = []
    last_infer_ms = 0.0
    history_states = collections.deque(maxlen=args.history)
    last_announced = "none"
    last_announce_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        run_now = (frame_idx % args.skip == 0)
        if run_now:
            boxes, scores, class_ids, infer_ms = run_inference(net, frame, args.imgsz, args.conf, args.iou)
            last_boxes, last_scores, last_class_ids = boxes, scores, class_ids
            last_infer_ms = infer_ms
        else:
            boxes, scores, class_ids = last_boxes, last_scores, last_class_ids
            infer_ms = last_infer_ms

        current_state = state_from_frame_idx(class_ids, scores, idx_go, idx_red, idx_off)
        history_states.append(current_state)
        stable_state = pick_stable_state(history_states)

        now = time.time()
        should_speak = (stable_state != last_announced) and ((now - last_announce_time) > args.cooldown)
        print(f"[FRAME {frame_idx}] inst={current_state} stable={stable_state} detec={len(boxes)} infer={infer_ms:.1f}ms speak={should_speak}")

        if should_speak:
            if stable_state == "go":
                play_audio(args.audio_go, "Pode atravessar com cuidado.", mute=args.noaudio)
            elif stable_state == "red":
                play_audio(args.audio_red, "Aguarde. Sinal vermelho.", mute=args.noaudio)
            elif stable_state == "off":
                play_audio(args.audio_off, "Semáforo não identificado.", mute=args.noaudio)
            last_announced = stable_state
            last_announce_time = now

        draw_boxes(frame, boxes, scores, class_ids, infer_ms, stable_state, CLASS_NAMES)

        if not args.nowindow:
            cv2.imshow("Pedestrian Light Assist (Audio + ClassOrder)", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    if not args.nowindow:
        cv2.destroyAllWindows()
    print("[INFO] Finalizado. Último estado falado:", last_announced)

if __name__ == "__main__":
    main()
