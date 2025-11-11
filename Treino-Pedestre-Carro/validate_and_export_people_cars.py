#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    # === CONFIGURAÇÕES ===
    DATA_YAML = r"G:\Downloads\PEDEST_CAR_6K\data.yaml"
    BEST_PT   = r"G:\Downloads\TCC3\TCC-DO-OCULOS-DAHORA\Treino-Pedestre-Carro\runs_people_vehicles\yolov8n_people_vehicles_640\weights\best.pt"
    IMG_SIZE  = 640

    # === VERIFICAÇÕES ===
    best_path = Path(BEST_PT)
    if not best_path.exists():
        print(f"[ERRO] best.pt não encontrado em {best_path}")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Usando device={device}")
    print(f"[INFO] Carregando modelo {best_path.name}")

    # === VALIDAR ===
    model = YOLO(str(best_path))
    print("[INFO] Rodando validação...")
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        split="val",
        device=device
    )

    print("[OK] Validação concluída.")
    print(f"[INFO] mAP50-95: {metrics.box.map:.4f} | mAP50: {metrics.box.map50:.4f}")

    # === EXPORTAR ===
    print("[INFO] Exportando para ONNX...")
    onnx_path = model.export(format="onnx", imgsz=IMG_SIZE)
    print(f"[OK] Export finalizado: {onnx_path}")

if __name__ == "__main__":
    main()
