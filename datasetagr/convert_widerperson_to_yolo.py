import os
from pathlib import Path
import shutil
import cv2

# ============================
# CONFIGURAÇÕES QUE VOCÊ ALTERA
# ============================

# Caminho da pasta WiderPerson original (a que tem Annotations/, Images/, train.txt etc.)
WIDER_ROOT = Path(r"G:\Downloads\WiderPerson")

# Pasta de saída já no formato YOLOv8
OUT_ROOT = Path(r"G:\Downloads\widerperson_yolo")

# Nome do YAML final
YAML_NAME = "widerperson.yaml"

# Nome da classe única
CLASS_NAME = "person"   # classe 0


# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_split_list(split_file: Path):
    """
    Lê train.txt / val.txt / etc.
    Cada linha costuma ser o nome do arquivo de imagem (ex: 000123.jpg)
    Retorna lista de nomes de imagem.
    """
    with open(split_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines

def read_annotation_file(ann_path: Path):
    """
    Lê anotação WiderPerson para UMA imagem.
    Formato esperado por linha:
      x1 y1 x2 y2 occlusion class_id
    Vamos ignorar occlusion e class_id.
    Retorna lista de boxes em coordenadas absolutas (x1,y1,x2,y2).
    Se não existir arquivo de anotação, retorna lista vazia.
    """
    if not ann_path.exists():
        return []

    boxes = []
    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            # Algumas anotações podem ter ruído, então tratamos com try/except
            try:
                x1, y1, x2, y2 = map(float, parts[:4])
            except ValueError:
                continue
            boxes.append((x1, y1, x2, y2))
    return boxes

def convert_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Converte box canto sup-esq / inf-dir -> YOLO normalizado:
    class x_center y_center w h
    """
    bw = x2 - x1
    bh = y2 - y1
    x_c = x1 + bw / 2.0
    y_c = y1 + bh / 2.0

    # normalizar [0,1]
    x_c_n = x_c / img_w
    y_c_n = y_c / img_h
    bw_n = bw / img_w
    bh_n = bh / img_h

    return x_c_n, y_c_n, bw_n, bh_n

def ensure_dirs(root: Path):
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)

def write_label_file(label_path: Path, yolo_lines):
    with open(label_path, "w") as f:
        for line in yolo_lines:
            f.write(line + "\n")

def process_split(
    split_name: str,
    img_names: list,
    in_img_dir: Path,
    in_ann_dir: Path,
    out_img_split_dir: Path,
    out_lbl_split_dir: Path
):
    """
    Converte um split específico (train ou val):
    - Copia imagem
    - Cria label YOLO correspondente
    """
    for img_name in img_names:
        # caminho da imagem original
        src_img_path = in_img_dir / img_name

        # se a extensão listada em train.txt/val.txt não bater com a imagem real,
        # tentamos outras extensões
        if not src_img_path.exists():
            base = Path(img_name).stem
            found = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                cand = in_img_dir / f"{base}{ext}"
                if cand.exists():
                    found = cand
                    break
            if found is None:
                print(f"[WARN] imagem {img_name} não encontrada, pulando")
                continue
            src_img_path = found

        # inferir caminho da anotação
        # regra: mesmo nome, mas .txt (ex: 000123.jpg -> 000123.txt)
        ann_file_name = (
            Path(src_img_path.name)
            .with_suffix(".txt")
            .name
        )
        src_ann_path = in_ann_dir / ann_file_name

        # ler imagem pra pegar w/h
        img = cv2.imread(str(src_img_path))
        if img is None:
            print(f"[WARN] falha ao ler {src_img_path}, pulando")
            continue
        h, w = img.shape[:2]

        # ler boxes no formato WiderPerson
        boxes_xyxy = read_annotation_file(src_ann_path)

        # converter cada bbox em linha YOLO: "0 x_c y_c w h"
        # classe fixa = 0 (person)
        yolo_lines = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            # validar bbox
            if x2 <= x1 or y2 <= y1:
                continue

            x_c, y_c, bw, bh = convert_bbox_to_yolo(x1, y1, x2, y2, w, h)

            # sanity check
            if bw <= 0 or bh <= 0:
                continue
            if not (0 <= x_c <= 1 and 0 <= y_c <= 1):
                # bbox fora do frame ou bugada
                continue

            yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        # nomes finais (mantém nome original da imagem)
        final_img_name = src_img_path.name
        final_lbl_name = Path(final_img_name).with_suffix(".txt").name

        dst_img_path = out_img_split_dir / final_img_name
        dst_lbl_path = out_lbl_split_dir / final_lbl_name

        # copiar imagem
        shutil.copy2(src_img_path, dst_img_path)

        # escrever .txt YOLO
        write_label_file(dst_lbl_path, yolo_lines)

def write_yaml(out_root: Path, yaml_name: str, class_name: str):
    """
    Cria arquivo YAML no padrão Ultralytics.
    """
    yaml_path = out_root / yaml_name
    with open(yaml_path, "w") as f:
        f.write(f"train: { (out_root / 'images' / 'train').as_posix() }\n")
        f.write(f"val: { (out_root / 'images' / 'val').as_posix() }\n\n")
        f.write("nc: 1\n")
        f.write(f"names: ['{class_name}']\n")

def main():
    images_dir = WIDER_ROOT / "Images"
    ann_dir    = WIDER_ROOT / "Annotations"

    train_txt  = WIDER_ROOT / "train.txt"
    val_txt    = WIDER_ROOT / "val.txt"

    # segurança: garante pastas de saída
    ensure_dirs(OUT_ROOT)

    # carrega quem é train e quem é val
    train_list = load_split_list(train_txt)
    val_list   = load_split_list(val_txt)

    # define destino de cada split
    out_img_train = OUT_ROOT / "images" / "train"
    out_lbl_train = OUT_ROOT / "labels" / "train"
    out_img_val   = OUT_ROOT / "images" / "val"
    out_lbl_val   = OUT_ROOT / "labels" / "val"

    print(f"[INFO] Convertendo treino ({len(train_list)} imagens)...")
    process_split(
        split_name="train",
        img_names=train_list,
        in_img_dir=images_dir,
        in_ann_dir=ann_dir,
        out_img_split_dir=out_img_train,
        out_lbl_split_dir=out_lbl_train
    )

    print(f"[INFO] Convertendo validação ({len(val_list)} imagens)...")
    process_split(
        split_name="val",
        img_names=val_list,
        in_img_dir=images_dir,
        in_ann_dir=ann_dir,
        out_img_split_dir=out_img_val,
        out_lbl_split_dir=out_lbl_val
    )

    # gera YAML final
    write_yaml(OUT_ROOT, YAML_NAME, CLASS_NAME)

    print("\n======================================")
    print("Concluído!")
    print("Dataset YOLO salvo em:", OUT_ROOT.resolve())
    print("Use para treinar YOLOv8n com:")
    print(f"yolo train model=yolov8n.pt data={ (OUT_ROOT / YAML_NAME).as_posix() } imgsz=640")
    print("======================================\n")

if __name__ == "__main__":
    main()
