import os
import shutil
from pathlib import Path

# ==========================
# CONFIGURAÇÃO DOS CAMINHOS
# ==========================

# pasta que você gerou com convert_crosswalk_to_yolo_final.py
ROOT_CROSSWALK = Path(r"G:\Downloads\crosswalk_yolo")

# pasta do dataset de semáforo de pedestre já convertido (ptld_yolo)
ROOT_LIGHTS    = Path(r"G:\Downloads\ptld_yolo-20251022T023154Z-1-001\ptld_yolo")

# pasta de saída do dataset combinado "travessia"
OUT_ROOT       = Path(r"G:\Downloads\crossing_dataset")

YAML_NAME      = "crossing_dataset.yaml"

# classes finais do novo dataset combinado
GLOBAL_CLASSES = [
    "crosswalk",        # 0
    "ped_light_go",     # 1
    "ped_light_stop",   # 2
    "ped_light_off"     # 3
]

# ==========================
# REMAPEAMENTO DE CLASSE
# ==========================

def remap_crosswalk(old_cls_str: str):
    # crosswalk_yolo tem só classe 0 = crosswalk
    # no dataset final queremos crosswalk = 0
    return 0

def remap_lights(old_cls_str: str):
    # ped_lights_yolo:
    #   0 -> ped_light_go
    #   1 -> ped_light_stop
    #   2 -> ped_light_off
    # no dataset final:
    #   go   -> 1
    #   stop -> 2
    #   off  -> 3
    mapping = {
        "0": 1,
        "1": 2,
        "2": 3,
    }
    return mapping.get(old_cls_str, None)

# ==========================
# HELPERS
# ==========================

def ensure_dirs(root: Path):
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)

def gather_split_pairs(dataset_root: Path):
    """
    Retorna dict:
    {
        "train": [(img_path, lbl_path), ...],
        "val":   [(img_path, lbl_path), ...]
    }
    Aceita 'val', 'valid', 'validation' como split de validação.
    """
    out = {"train": [], "val": []}

    for split in ["train", "val", "valid", "validation", "test"]:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue

        # normalizar nome de split: tudo que não é train vira val
        norm_split = "train" if split == "train" else "val"

        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            out[norm_split].append((img_path, lbl_path))

    return out

def copy_and_remap(
    pairs,
    split,
    dataset_tag,
    out_img_dir,
    out_lbl_dir,
    remap_fn
):
    """
    Copia imagens e converte labels para o espaço global final.
    `dataset_tag` entra no prefixo do nome final p/ evitar colisão.
    """
    total_imgs = len(pairs)
    count = 0

    for img_path, lbl_path in pairs:
        count += 1
        if count == 1 or count % 50 == 0 or count == total_imgs:
            print(f"[{dataset_tag} {split}] {count}/{total_imgs}")

        # nome único final
        new_stem = f"{dataset_tag}_{split}_{img_path.stem}"

        # destino da imagem
        dst_img = out_img_dir / f"{new_stem}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dst_img)

        # converter labels
        new_label_lines = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    old_cls = parts[0]
                    x_c, y_c, w, h = parts[1:]

                    new_cls = remap_fn(old_cls)
                    if new_cls is None:
                        continue

                    new_label_lines.append(f"{new_cls} {x_c} {y_c} {w} {h}")

        dst_lbl = out_lbl_dir / f"{new_stem}.txt"
        with open(dst_lbl, "w") as f:
            for ln in new_label_lines:
                f.write(ln + "\n")

    return total_imgs

def write_yaml(out_root: Path, yaml_name: str, classes: list):
    yaml_path = out_root / yaml_name
    with open(yaml_path, "w") as f:
        f.write(f"train: { (out_root / 'images' / 'train').as_posix() }\n")
        f.write(f"val:   { (out_root / 'images' / 'val').as_posix() }\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names: [")
        f.write(", ".join([f"'{c}'" for c in classes]))
        f.write("]\n")

# ==========================
# MAIN
# ==========================

def main():
    print("==> Criando dataset combinado de travessia (crosswalk + ped lights)...")
    ensure_dirs(OUT_ROOT)

    OUT_IMG_TRAIN = OUT_ROOT / "images" / "train"
    OUT_IMG_VAL   = OUT_ROOT / "images" / "val"
    OUT_LBL_TRAIN = OUT_ROOT / "labels" / "train"
    OUT_LBL_VAL   = OUT_ROOT / "labels" / "val"

    # ler os dois datasets de origem
    cw_pairs = gather_split_pairs(ROOT_CROSSWALK)
    pl_pairs = gather_split_pairs(ROOT_LIGHTS)

    stats = []

    # crosswalk -> classe global 0
    print("\n[MESCLANDO CROSSWALK]")
    n_train = copy_and_remap(
        cw_pairs["train"],
        "train",
        "crosswalk",
        OUT_IMG_TRAIN,
        OUT_LBL_TRAIN,
        remap_crosswalk
    )
    n_val = copy_and_remap(
        cw_pairs["val"],
        "val",
        "crosswalk",
        OUT_IMG_VAL,
        OUT_LBL_VAL,
        remap_crosswalk
    )
    stats.append(("crosswalk", n_train, n_val))

    # ped_lights -> classes globais 1,2,3
    print("\n[MESCLANDO PEDESTRIAN LIGHTS]")
    n_train = copy_and_remap(
        pl_pairs["train"],
        "train",
        "pedlight",
        OUT_IMG_TRAIN,
        OUT_LBL_TRAIN,
        remap_lights
    )
    n_val = copy_and_remap(
        pl_pairs["val"],
        "val",
        "pedlight",
        OUT_IMG_VAL,
        OUT_LBL_VAL,
        remap_lights
    )
    stats.append(("ped_lights", n_train, n_val))

    # gera YAML final
    write_yaml(OUT_ROOT, YAML_NAME, GLOBAL_CLASSES)

    # print resumo
    print("\n==================== RESUMO ====================")
    for name, tr_count, val_count in stats:
        print(f"{name:12s} -> train:{tr_count:5d} imgs  | val:{val_count:5d} imgs")
    print("================================================")

    print("\n✅ Dataset combinado gerado!")
    print("Pasta final:", OUT_ROOT.resolve())
    print("\nTreine com:")
    print(f"yolo train model=yolov8n.pt data={ (OUT_ROOT / YAML_NAME).as_posix() } imgsz=640 batch=16 epochs=100")
    print("================================================\n")

if __name__ == "__main__":
    main()
