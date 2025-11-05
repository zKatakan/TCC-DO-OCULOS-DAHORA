import os
import random
import shutil
from pathlib import Path

# ==============================
# AJUSTE AQUI
# ==============================

# Raiz onde está esse dataset de veículos
# (A pasta que contém "train/images" e "train/labels")
VEH_ROOT = Path(r"G:\Downloads\archive")

# Pasta final convertida para YOLOv8 com classe única "auto"
OUT_ROOT = Path(r"G:\Downloads\vehicles_yolo")

# nome da classe final
CLASS_NAME = "auto"

# porcentagem das imagens que vão pra validação
VAL_SPLIT = 0.2  # 20% validação


# ==============================
# FUNÇÕES AUX
# ==============================

def ensure_dirs(root: Path):
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)

def read_label_file(label_path: Path):
    """
    Lê um .txt de label YOLO.
    Retorna lista de linhas já normalizadas mas forçando a classe = 0 (auto).
    Se o arquivo não existir, retorna lista vazia.
    """
    lines_out = []
    if not label_path.exists():
        return lines_out

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # parts[0] = classe original
            # parts[1:] = x_c y_c w h normalizados
            x_c, y_c, w, h = parts[1:]
            # força classe única "auto" = 0
            new_line = f"0 {x_c} {y_c} {w} {h}"
            lines_out.append(new_line)
    return lines_out

def write_label_file(dst_path: Path, lines):
    with open(dst_path, "w") as f:
        for ln in lines:
            f.write(ln + "\n")

# ==============================
# PIPELINE
# ==============================

def main():
    train_img_dir = VEH_ROOT / "train" / "images"
    train_lbl_dir = VEH_ROOT / "train" / "labels"

    assert train_img_dir.is_dir(), f"❌ Não achei {train_img_dir}"
    assert train_lbl_dir.is_dir(), f"❌ Não achei {train_lbl_dir}"

    # cria estrutura de saída
    ensure_dirs(OUT_ROOT)
    out_img_train = OUT_ROOT / "images" / "train"
    out_lbl_train = OUT_ROOT / "labels" / "train"
    out_img_val   = OUT_ROOT / "images" / "val"
    out_lbl_val   = OUT_ROOT / "labels" / "val"

    # pega todas as imagens disponíveis
    imgs = [
        p for p in train_img_dir.glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    random.shuffle(imgs)

    # split em train/val
    n_val = int(len(imgs) * VAL_SPLIT)
    val_set = set(imgs[:n_val])
    train_set = set(imgs[n_val:])

    print(f"[INFO] total imagens: {len(imgs)}")
    print(f"[INFO] train: {len(train_set)}, val: {len(val_set)}")

    # processar cada imagem
    for img_path in imgs:
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")

        # lê labels e força classe = 0 ("auto")
        yolo_lines = read_label_file(lbl_path)

        if img_path in val_set:
            dst_img = out_img_val / img_path.name
            dst_lbl = out_lbl_val / (img_path.stem + ".txt")
        else:
            dst_img = out_img_train / img_path.name
            dst_lbl = out_lbl_train / (img_path.stem + ".txt")

        # copia imagem
        shutil.copy2(img_path, dst_img)
        # salva labels convertidas
        write_label_file(dst_lbl, yolo_lines)

    # cria YAML final
    yaml_path = OUT_ROOT / "vehicles.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: { (OUT_ROOT / 'images' / 'train').as_posix() }\n")
        f.write(f"val:   { (OUT_ROOT / 'images' / 'val').as_posix() }\n\n")
        f.write("nc: 1\n")
        f.write(f"names: ['{CLASS_NAME}']\n")

    print("\n======================================")
    print("✅ Dataset de veículos convertido!")
    print("Saída em:", OUT_ROOT.resolve())
    print("Treine com:")
    print(f"yolo train model=yolov8n.pt data={yaml_path.as_posix()} imgsz=640 batch=16 epochs=100")
    print("======================================\n")

if __name__ == "__main__":
    main()
