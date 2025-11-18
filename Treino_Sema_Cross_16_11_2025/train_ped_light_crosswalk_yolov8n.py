#!/usr/bin/env python3
"""
Treino YOLOv8n para:
- semáforo de pedestre
- faixa de pedestre

Exporta para ONNX (imgsz=640) e faz checagem automática de GPU/CUDA.
"""

from ultralytics import YOLO
import torch, os, shutil

# =========================
# CONFIGS
# =========================
DATA_YAML = r"C:\Users\evosystem03.ti\Downloads\Treino_Sema_Cross_16_11_2025\merged_crossing_3k_3k\crossing_3k3k.yaml"  # <-- troque aqui
RUN_NAME = "yolov8n_ped_light_crosswalk_640"
PROJECT_DIR = "runs_ped_cross"
IMGSZ = 640
EPOCHS = 175
BATCH = 16

os.makedirs(PROJECT_DIR, exist_ok=True)


def pick_device():
    """retorna '0' se tiver GPU, senão 'cpu'"""
    has_cuda = torch.cuda.is_available()
    print("=== Checando ambiente PyTorch ===")
    print("PyTorch:", torch.__version__)
    print("CUDA disponível?:", has_cuda)
    if has_cuda:
        name = torch.cuda.get_device_name(0)
        print("GPU detectada:", name)
        print("=================================\n")
        return 0  # ultralytics entende 0 como GPU 0
    else:
        print("⚠️  Nenhuma GPU/CUDA detectada, vou usar CPU.")
        print("Se você TEM uma RTX 2070, precisa instalar o PyTorch com CUDA.")
        print("=================================\n")
        return "cpu"


def main():
    device = pick_device()

    print("[INFO] carregando modelo base YOLOv8n...")
    model = YOLO("yolov8n.pt")

    print("[INFO] iniciando treino...")
    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=device,          # <- agora é dinâmico
        optimizer="AdamW",
        lr0=0.003,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        patience=45,
        project=PROJECT_DIR,
        name=RUN_NAME,
        cache="disk",
        close_mosaic=10,
    )

    print("[INFO] treino finalizado. runs em:", results.save_dir)

    best_path = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
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

    simple_name = "best_ped_light_crosswalk_640.onnx"
    if os.path.exists(onnx_path):
        shutil.copy(onnx_path, simple_name)
        print("[INFO] cópia criada em:", simple_name)
    else:
        print("[WARN] ONNX não encontrado para copiar.")


if __name__ == "__main__":
    main()
