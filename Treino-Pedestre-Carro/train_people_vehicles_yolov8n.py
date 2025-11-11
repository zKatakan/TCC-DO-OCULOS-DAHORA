#!/usr/bin/env python3
"""
Treino YOLOv8n para detectar:
- pessoas
- veículos (carro, moto, caminhão... o que estiver no seu .yaml)

Pensado pra dataset grande (~11k imagens), então epochs reduzidas (50).
Exporta para ONNX (640) no final.
"""

from ultralytics import YOLO
import torch
import os
import shutil

# =========================
# CONFIGURAÇÕES
# =========================
# TROQUE AQUI pelo caminho real do seu dataset de pessoas + veículos
DATA_YAML = r"C:\Users\evosystem03.ti\Downloads\Treino-Pedestre-Carro\PEDEST_CAR_6K\data.yaml"

PROJECT_DIR = "runs_people_vehicles"
RUN_NAME = "yolov8n_people_vehicles_640"
IMGSZ = 640
EPOCHS = 175          # dataset grande -> menos épocas
BATCH = 16           # se faltar VRAM, baixa pra 8

os.makedirs(PROJECT_DIR, exist_ok=True)


def pick_device():
    """Escolhe GPU se existir, senão CPU."""
    print("=== Checando ambiente PyTorch ===")
    print("PyTorch:", torch.__version__)
    has_cuda = torch.cuda.is_available()
    print("CUDA disponível?:", has_cuda)
    if has_cuda:
        name = torch.cuda.get_device_name(0)
        print("GPU detectada:", name)
        print("=================================\n")
        return 0
    else:
        print("⚠️  Nenhuma GPU/CUDA detectada, vou usar CPU. O treino pode demorar mais.")
        print("=================================\n")
        return "cpu"


def main():
    device = pick_device()

    print("[INFO] carregando modelo base YOLOv8n...")
    model = YOLO("yolov8n.pt")

    print("[INFO] iniciando treino para pessoas + veículos ...")
    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=device,
        optimizer="AdamW",
        lr0=0.003,
        weight_decay=0.0005,
        # aug padrão do YOLO
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        # como o dataset é grande, mas queremos parar se estabilizar:
        patience=50,
        project=PROJECT_DIR,
        name=RUN_NAME,
        cache="disk",
        close_mosaic=10,
    )

    print("[INFO] treino finalizado. runs em:", results.save_dir)

    # monta caminho do best a partir do save_dir pra não dar o erro de antes
    save_dir = results.save_dir  # ex: G:\...\runs_people_vehicles\yolov8n_people_vehicles_640
    best_path = os.path.join(save_dir, "weights", "best.pt")

    if not os.path.exists(best_path):
        raise FileNotFoundError(f"best.pt não encontrado em {best_path}")

    print("[INFO] validando best.pt ...")
    model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        weights=best_path,
        device=device,
    )

    print("[INFO] exportando para ONNX ...")
    best_model = YOLO(best_path)
    onnx_path = best_model.export(
        format="onnx",
        imgsz=IMGSZ,
        dynamic=False,
        simplify=False,
    )
    print("[INFO] ONNX gerado em:", onnx_path)

    # opcional: copiar pra um nome mais simples
    simple_name = "best_people_vehicles_640.onnx"
    try:
        shutil.copy(onnx_path, simple_name)
        print("[INFO] cópia criada em:", simple_name)
    except Exception as e:
        print("[WARN] não consegui copiar o ONNX pra", simple_name, "->", e)


if __name__ == "__main__":
    main()
