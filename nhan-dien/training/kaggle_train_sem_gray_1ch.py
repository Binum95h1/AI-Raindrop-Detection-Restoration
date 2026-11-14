# ============================================================
# YOLO26n Semantic Segmentation - GRAYSCALE 1 CHANNEL
# PERCENT = 50:  640 x 360
# PERCENT = 100: 1280 x 720
# ============================================================

PERCENT = 100  # DOI THANH 100 NEU MUON TRAIN FULL SIZE

import sys
import subprocess
from pathlib import Path
from IPython.display import FileLink, display

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "-U", "ultralytics>=8.4.147"
])

SCRIPT = r'''
from pathlib import Path
import random
import hashlib
import zipfile
import argparse
import shutil
from collections import defaultdict

import cv2
import numpy as np
import yaml
import torch

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import SemanticSegmentationModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--percent",
        type=int,
        choices=[50, 100],
        required=True
    )
    opt = parser.parse_args()

    PCT = opt.percent
    SIZE = (640, 360) if PCT == 50 else (1280, 720)

    # DUONG DAN DATASET KAGGLE
    IMG_ROOT = Path(
        "/kaggle/input/datasets/nguyentuananh1/"
        "muaxla/Train/drop"
    )

    MASK_ROOT = Path(
        "/kaggle/input/datasets/nguyentuananh1/"
        "muaxla/Train/mask_drop/01drop/01drop"
    )

    DATA = Path(
        f"/kaggle/working/rain_sem_1ch_{PCT}pct"
    )

    PROJECT = Path(
        "/kaggle/working/runs_rain_sem_1ch"
    )

    NAME = f"yolo26n_sem_{PCT}pct_1ch_pw0p5_b4"
    RUN_DIR = PROJECT / NAME

    ZIP_PATH = Path("/kaggle/working") / (
        f"{NAME}_weights.zip"
    )

    EXTS = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff", ".webp"
    }

    # -------------------------------
    # 1. KIEM TRA MOI TRUONG
    # -------------------------------
    if "cls_pw" not in DEFAULT_CFG_DICT:
        raise RuntimeError(
            "Ultralytics chua ho tro cls_pw"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Hay bat GPU trong Kaggle!"
        )

    if (RUN_DIR / "weights" / "last.pt").exists():
        raise RuntimeError(
            f"Checkpoint da ton tai: {RUN_DIR}\n"
            "Doi NAME neu muon train run moi."
        )

    if not IMG_ROOT.is_dir() or not MASK_ROOT.is_dir():
        raise FileNotFoundError(
            "Khong tim thay dataset muaxla. "
            "Kiem tra Add Input."
        )

    # -------------------------------
    # 2. TIM ANH VA MASK
    # -------------------------------
    def scan(root):
        found = {}

        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue

            if p.suffix.lower() not in EXTS:
                continue

            key = (
                p.relative_to(root)
                 .with_suffix("")
                 .as_posix()
                 .lower()
            )

            if key in found:
                raise RuntimeError(
                    f"Trung ten anh: {key}"
                )

            found[key] = p

        return found

    imgs = scan(IMG_ROOT)
    masks = scan(MASK_ROOT)

    if not imgs or set(imgs) != set(masks):
        print("Images:", len(imgs))
        print("Masks:", len(masks))
        print("Pairs:", len(set(imgs) & set(masks)))
        raise RuntimeError(
            "Anh va mask khong khop nhau!"
        )

    keys = sorted(imgs)

    print("TOTAL PAIRS:", len(keys))
    print("RESOLUTION:", SIZE)
    print("INPUT CHANNELS: 1")

    # -------------------------------
    # 3. CHIA TRAIN / VAL THEO SCENE
    # -------------------------------
    groups = defaultdict(list)

    for key in keys:
        rel = imgs[key].relative_to(IMG_ROOT)

        scene = (
            rel.parts[0].lower()
            if len(rel.parts) > 1
            else "__root__"
        )

        groups[scene].append(key)

    if len(groups) < 2:
        raise RuntimeError(
            "Khong du scene de chia train/val!"
        )

    scenes = sorted(groups)
    random.Random(42).shuffle(scenes)

    val_scenes = set()
    val_count = 0
    target = round(len(keys) * 0.25)

    for scene in scenes[:-1]:
        if val_count >= target and val_scenes:
            break

        val_scenes.add(scene)
        val_count += len(groups[scene])

    print("VAL SCENES:", sorted(val_scenes))

    # -------------------------------
    # 4. KIEM TRA CNN 1 KENH THAT
    # -------------------------------
    base = YOLO("yolo26n-sem.pt")

    if base.task != "semantic":
        raise RuntimeError(
            f"Sai model task: {base.task}"
        )

    probe = SemanticSegmentationModel(
        base.model.yaml,
        nc=2,
        ch=1,
        verbose=False
    )

    in_ch = probe.model[0].conv.in_channels

    print("FIRST CONV INPUT CHANNELS:", in_ch)

    assert in_ch == 1, (
        "CNN van dang dung 3 kenh!"
    )

    del probe, base

    # -------------------------------
    # 5. TAO DATASET GRAYSCALE 1CH
    # -------------------------------
    # Xoa du lieu xu ly cu CUA DUNG THI NGHIEM
    # de tranh anh con sot khi chay lai.
    for folder in [
        DATA / "images",
        DATA / "masks"
    ]:
        if folder.exists():
            shutil.rmtree(folder)

    train_count = 0
    val_count = 0

    for i, key in enumerate(keys, 1):
        img_path = imgs[key]
        mask_path = masks[key]

        # Doc anh that su thanh 1 kenh
        gray = cv2.imread(
            str(img_path),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED
        )

        if gray is None or mask is None:
            raise RuntimeError(
                f"Khong doc duoc: {key}"
            )

        if mask.ndim == 3:
            if not (
                np.array_equal(
                    mask[:, :, 0],
                    mask[:, :, 1]
                )
                and np.array_equal(
                    mask[:, :, 1],
                    mask[:, :, 2]
                )
            ):
                raise ValueError(
                    f"Mask khong phai grayscale: {key}"
                )

            mask = mask[:, :, 0]

        if (
            gray.shape != (720, 1280)
            or mask.shape != gray.shape
        ):
            raise ValueError(
                f"Sai kich thuoc: {key}\n"
                f"Image: {gray.shape}\n"
                f"Mask: {mask.shape}"
            )

        # Chuyen mask 0/255 thanh 0/1
        if np.all(
            (mask == 0) | (mask == 255)
        ):
            mask = (mask == 255).astype(np.uint8)

        elif np.all(
            (mask == 0) | (mask == 1)
        ):
            mask = mask.astype(np.uint8)

        else:
            raise ValueError(
                f"Mask co gia tri khong hop le: {key}"
            )

        # Resize neu chon 50%
        if PCT == 50:
            gray = cv2.resize(
                gray,
                SIZE,
                interpolation=cv2.INTER_AREA
            )

            mask = cv2.resize(
                mask,
                SIZE,
                interpolation=cv2.INTER_NEAREST
            )

        # Scene giu nguyen giua 2 thuc nghiem
        rel = img_path.relative_to(IMG_ROOT)

        scene = (
            rel.parts[0].lower()
            if len(rel.parts) > 1
            else "__root__"
        )

        split = (
            "val"
            if scene in val_scenes
            else "train"
        )

        # Ten anh duy nhat va mask trung ten
        stem = hashlib.sha256(
            key.encode()
        ).hexdigest()[:24]

        out_img = (
            DATA / "images" / split / f"{stem}.png"
        )

        out_mask = (
            DATA / "masks" / split / f"{stem}.png"
        )

        out_img.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        out_mask.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not cv2.imwrite(str(out_img), gray):
            raise RuntimeError("Khong ghi duoc anh!")

        if not cv2.imwrite(str(out_mask), mask):
            raise RuntimeError("Khong ghi duoc mask!")

        if split == "train":
            train_count += 1
        else:
            val_count += 1

        if i % 500 == 0 or i == len(keys):
            print(
                f"Prepared {i}/{len(keys)}",
                flush=True
            )

    print("TRAIN IMAGES:", train_count)
    print("VAL IMAGES:", val_count)

    # -------------------------------
    # 6. DATA YAML - 1 KENH
    # -------------------------------
    data_yaml = DATA / "data.yaml"

    config = {
        "path": str(DATA),
        "train": "images/train",
        "val": "images/val",
        "masks_dir": "masks",
        "channels": 1,
        "names": {
            0: "background",
            1: "raindrop"
        }
    }

    data_yaml.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False
        ),
        encoding="utf-8"
    )

    print("DATA YAML:", data_yaml)

    # -------------------------------
    # 7. TRAIN
    # -------------------------------
    device = (
        [0, 1]
        if torch.cuda.device_count() >= 2
        else 0
    )

    print("DEVICE:", device)
    print("BATCH TOTAL: 4")
    print("CLS_PW: 0.5")
    print("PATIENCE: 15")

    model = YOLO("yolo26n-sem.pt")

    model.train(
        data=str(data_yaml),

        # Anh 50% hoac 100%
        imgsz=SIZE[0],
        rect=True,

        # Training
        epochs=100,
        patience=15,
        batch=4,
        device=device,
        workers=2,

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        # Weighted CE + Dice + Auxiliary
        cls_pw=0.5,

        warmup_epochs=2.0,
        cos_lr=True,

        seed=42,
        deterministic=False,

        amp=True,
        cache=False,

        # Disable heavy augmentation
        multi_scale=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        cutmix=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        translate=0.02,
        scale=0.0,
        shear=0.0,
        perspective=0.0,

        # Output
        project=str(PROJECT),
        name=NAME,
        exist_ok=False,
        save=True,
        val=True,
        plots=True
    )

    # -------------------------------
    # 8. KIEM TRA CHECKPOINT
    # -------------------------------
    actual_dir = Path(model.trainer.save_dir)

    best_path = actual_dir / "weights" / "best.pt"
    last_path = actual_dir / "weights" / "last.pt"

    assert best_path.exists(), "Thieu best.pt!"
    assert last_path.exists(), "Thieu last.pt!"

    # Kiem tra model da train van la 1ch
    best_model = YOLO(str(best_path))

    assert (
        best_model.model.model[0].conv.in_channels == 1
    ), "Checkpoint KHONG phai model 1 channel!"

    del best_model

    # -------------------------------
    # 9. NEN ZIP CHECKPOINT
    # -------------------------------
    files = [
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "args.yaml"
    ]

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        zipfile.ZIP_DEFLATED,
        compresslevel=3
    ) as z:
        for filename in files:
            p = actual_dir / filename

            if p.exists():
                z.write(
                    p,
                    arcname=filename
                )

    print("\n========== TRAIN FINISHED ==========")
    print("RUN:", actual_dir)
    print("BEST:", best_path)
    print("LAST:", last_path)
    print("ZIP:", ZIP_PATH)

    print(
        "ZIP SIZE:",
        round(
            ZIP_PATH.stat().st_size / 1024**2,
            2
        ),
        "MB"
    )


if __name__ == "__main__":
    main()
'''

# Tao Python script that, tranh loi %%writefile va DDP
script_path = Path("/kaggle/working/train_sem_gray_1ch.py")
script_path.write_text(SCRIPT, encoding="utf-8")

# Train
subprocess.run(
    [
        sys.executable,
        "-u",
        str(script_path),
        "--percent",
        str(PERCENT)
    ],
    check=True
)

# Hien link download ZIP ngay sau train
name = f"yolo26n_sem_{PERCENT}pct_1ch_pw0p5_b4"
zip_path = Path("/kaggle/working") / f"{name}_weights.zip"

if zip_path.exists():
    print("\nDA TAO ZIP:", zip_path)
    display(
        FileLink(
            str(zip_path),
            result_html_prefix="BAM DE TAI ZIP: "
        )
    )