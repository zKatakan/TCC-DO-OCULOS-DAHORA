from ultralytics import YOLO
import os, shutil

# caminho REAL do best.pt que o treino acabou de gerar
BEST_PT = r"G:\Downloads\TCC_CROSSWALKLIGHT\runs_ped_cross\yolov8n_ped_light_crosswalk_6402\weights\best.pt"

print("[INFO] carregando modelo:", BEST_PT)
model = YOLO(BEST_PT)

print("[INFO] exportando para ONNX (640)...")
onnx_path = model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    simplify=False,
)

print("[INFO] ONNX gerado em:", onnx_path)

# opcional: copiar pra um nome mais simples na pasta atual
dst = r"G:\Downloads\TCC_CROSSWALKLIGHT\best_ped_light_crosswalk_640.onnx"
try:
    shutil.copy(onnx_path, dst)
    print("[INFO] cópia criada em:", dst)
except Exception as e:
    print("[WARN] não consegui copiar:", e)
