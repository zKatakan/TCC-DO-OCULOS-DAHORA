import os
import random
import shutil
from pathlib import Path

# ==============================
# CONFIGURAÇÕES
# ==============================
DATASET_ROOT = Path(r"G:\Downloads\8289874\CDSet\CDSet\dataset_YOLO_format_3434")
OUT_ROOT = Path(r"G:\Downloads\crosswalk_yolo")

CLASS_NAME = "crosswalk"  # única classe
VAL_SPLIT = 0.2  # 20% validação

# ==============================
# FUNÇÕES AUXILIARES
# ==============================

def ensure_dirs(root: Path):
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)

def read_label_file(label_path: Path):
    """Lê um arquivo .txt YOLO e retorna as linhas válidas da classe crosswalk (id 0)."""
    lines_out = []
    if not label_path.exists():
        return lines_out
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id == 0:  # manter só crosswalk
                lines_out.append(line.strip())
    return lines_out

def write_label_file(dst, lines):
    with open(dst, "w") as f:
        for ln in lines:
            f.write(ln + "\n")

# ==============================
# EXECUÇÃO PRINCIPAL
# ==============================

def main():
    img_dir = DATASET_ROOT / "images"
    lbl_dir = DATASET_ROOT / "labels"

    assert img_dir.is_dir(), f"❌ Pasta não encontrada: {img_dir}"
    assert lbl_dir.is_dir(), f"❌ Pasta não encontrada: {lbl_dir}"

    out_img_train = OUT_ROOT / "images" / "train"
    out_lbl_train = OUT_ROOT / "labels" / "train"
    out_img_val   = OUT_ROOT / "images" / "val"
    out_lbl_val   = OUT_ROOT / "labels" / "val"
    ensure_dirs(OUT_ROOT)

    # todas as imagens
    imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    random.shuffle(imgs)
    n_val = int(len(imgs) * VAL_SPLIT)
    val_imgs = set(imgs[:n_val])
    train_imgs = set(imgs[n_val:])

    for img_path in imgs:
        label_path = lbl_dir / (img_path.stem + ".txt")
        lines = read_label_file(label_path)

        # destinos
        if img_path in val_imgs:
            dst_img = out_img_val / img_path.name
            dst_lbl = out_lbl_val / (img_path.stem + ".txt")
        else:
            dst_img = out_img_train / img_path.name
            dst_lbl = out_lbl_train / (img_path.stem + ".txt")

        shutil.copy2(img_path, dst_img)
        write_label_file(dst_lbl, lines)

    # cria YAML final
    yaml_path = OUT_ROOT / "crosswalk.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: { (OUT_ROOT / 'images/train').as_posix() }\n")
        f.write(f"val: { (OUT_ROOT / 'images/val').as_posix() }\n\n")
        f.write("nc: 1\n")
        f.write(f"names: ['{CLASS_NAME}']\n")

    print("\n✅ Conversão concluída!")
    print("Dataset pronto em:", OUT_ROOT.resolve())
    print("Use para treinar com:")
    print(f"yolo train model=yolov8n.pt data={ (OUT_ROOT / 'crosswalk.yaml').as_posix() } imgsz=640 batch=16 epochs=100\n")

if __name__ == "__main__":
    main()
