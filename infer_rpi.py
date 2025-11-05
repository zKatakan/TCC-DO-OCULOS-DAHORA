
import cv2, numpy as np, time, argparse, os

def letterbox(img, new_shape=640, stride=32):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return im, r, (left, top)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="best.onnx", help="caminho para o modelo ONNX")
    ap.add_argument("--source", type=str, default="0", help="0 (webcam) ou caminho de vídeo")
    ap.add_argument("--imgsz", type=int, default=640, help="tamanho de entrada quadrado")
    ap.add_argument("--conf", type=float, default=0.40, help="limiar de confiança")
    ap.add_argument("--iou", type=float, default=0.55, help="limiar de IOU para NMS")
    args = ap.parse_args()

    onnx_path = args.onnx
    if not os.path.exists(onnx_path):
        print("Modelo ONNX não encontrado:", onnx_path)
        return

    net = cv2.dnn.readNetFromONNX(onnx_path)
    # Tente preferências de backend/target adequadas para o Pi 5:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Fonte de vídeo
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Falha ao abrir a fonte de vídeo:", args.source)
        return

    CLASS_NAMES = ["ped_light_go", "ped_light_off", "ped_light_red"]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        inp, r, (dx, dy) = letterbox(frame, args.imgsz)
        blob = cv2.dnn.blobFromImage(inp, 1/255.0, (args.imgsz, args.imgsz), swapRB=True, crop=False)
        net.setInput(blob)

        t0 = time.time()
        out = net.forward()
        dt = (time.time() - t0) * 1000.0  # ms

        out = np.squeeze(out)  # [N, 85] ou [84, 8400] dependendo da exportação
        boxes, scores, class_ids = [], [], []

        # Tente detectar o formato automaticamente
        if out.ndim == 2 and out.shape[1] >= 85:
            # Formato [N, 85] (x, y, w, h, conf, cls...)
            for det in out:
                conf = float(det[4])
                if conf < args.conf:
                    continue
                cls_scores = det[5:]
                cls_id = int(np.argmax(cls_scores))
                score  = float(cls_scores[cls_id] * conf)
                if score < args.conf:
                    continue
                cx, cy, w, h = det[:4]
                x1 = (cx - w/2 - dx) / r
                y1 = (cy - h/2 - dy) / r
                x2 = (cx + w/2 - dx) / r
                y2 = (cy + h/2 - dy) / r
                boxes.append([x1, y1, x2, y2]); scores.append(score); class_ids.append(cls_id)
        else:
            # Ex.: [84, 8400] → precisamos reconstruir; este ramo é conservador
            # Assumindo: out[0:4,:] = [cx, cy, w, h], out[4,:] = conf, out[5:,:] = classes
            if out.ndim == 2 and out.shape[0] >= 85:
                cx, cy, w, h = out[0], out[1], out[2], out[3]
                confs = out[4]
                cls_scores_mat = out[5:]
                for i in range(out.shape[1]):
                    conf = float(confs[i])
                    if conf < args.conf:
                        continue
                    cls_id = int(np.argmax(cls_scores_mat[:, i]))
                    score = float(cls_scores_mat[cls_id, i] * conf)
                    if score < args.conf:
                        continue
                    x1 = (cx[i] - w[i]/2 - dx) / r
                    y1 = (cy[i] - h[i]/2 - dy) / r
                    x2 = (cx[i] + w[i]/2 - dx) / r
                    y2 = (cy[i] + h[i]/2 - dy) / r
                    boxes.append([x1, y1, x2, y2]); scores.append(score); class_ids.append(cls_id)

        # NMS
        nms_boxes = [ [int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in boxes ]
        idxs = cv2.dnn.NMSBoxes(nms_boxes, scores, args.conf, args.iou)

        if len(idxs) > 0:
            for i in idxs.flatten():
                x1, y1, x2, y2 = map(int, boxes[i])
                cls = CLASS_NAMES[class_ids[i]] if 0 <= class_ids[i] < len(CLASS_NAMES) else str(class_ids[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{cls} {scores[i]:.2f}", (x1, max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(frame, f"{dt:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Pedestrian Light — ONNX (OpenCV DNN)", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
