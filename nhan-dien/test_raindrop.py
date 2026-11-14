"""Test the uploaded raindrop OBB checkpoint on local images."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE / "input"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=BASE / "model" / "best.pt")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help="One image or a folder (recursive); default: input folder beside this script")
    p.add_argument("--output", type=Path, default=BASE / "output")
    p.add_argument("--mode", choices=("full", "grid"), default="full")
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=None, help="Default: full=640, grid=256")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=1000, help="Maximum boxes per image/crop")
    p.add_argument("--batch", type=int, default=8, help="Crop inference batch size")
    p.add_argument("--device", default="cpu", help="cpu, or 0 if CUDA PyTorch is installed")
    a = p.parse_args()
    if not 0 <= a.conf <= 1 or not 0 <= a.iou <= 1:
        p.error("--conf and --iou must be between 0 and 1")
    if min(a.cols, a.rows, a.batch, a.max_det) < 1:
        p.error("--cols, --rows, --batch and --max-det must be positive")
    if a.imgsz is None:
        a.imgsz = 640 if a.mode == "full" else 256
    if a.imgsz < 32 or a.imgsz % 32:
        p.error("--imgsz must be a positive multiple of 32")
    return a

def image_paths(source):
    if source.is_file():
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError("Input must be an image, not a ZIP or model file")
        return [source], source.parent
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS), source
    raise FileNotFoundError(f"Input does not exist: {source}")

def new_run(root, label):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(1, 10000):
        run = root / f"{label}_{i:03d}"
        try:
            run.mkdir()
            return run
        except FileExistsError:
            continue
    raise RuntimeError("Too many output runs")

def crop_specs(width, height, cols, rows):
    if width < cols or height < rows:
        raise ValueError("Image is smaller than the selected grid; use --mode full")
    return [(width * c // cols, height * r // rows,
             width * (c + 1) // cols, height * (r + 1) // rows)
            for r in range(rows) for c in range(cols)]

def main():
    a = arguments()
    try:
        import cv2
        import numpy as np
        import ultralytics
        from PIL import Image, ImageOps
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("Missing dependency. Run 01_SETUP_CPU.cmd first. " + str(e)) from e
    paths, source_root = image_paths(a.input.resolve())
    if not paths:
        raise ValueError(f"No images found in: {a.input.resolve()}")
    weights = a.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model not found: {weights}")
    model = YOLO(str(weights), task="obb")
    if model.task != "obb":
        raise ValueError("This script requires an OBB model")
    label = "full" if a.mode == "full" else f"grid{a.cols}x{a.rows}"
    run = new_run(a.output.resolve(), label)
    settings = {"input": str(a.input.resolve()), "weights": str(weights), "image_count": len(paths),
                "mode": a.mode, "imgsz": a.imgsz, "conf": a.conf, "iou": a.iou,
                "device": a.device, "max_det_per_crop": a.max_det,
                "grid_columns": a.cols if a.mode == "grid" else 1,
                "grid_rows": a.rows if a.mode == "grid" else 1,
                "grid_overlap": 0, "batch": a.batch, "ultralytics": ultralytics.__version__}
    (run / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    fields = ["image", "status", "detections", "max_confidence", "elapsed_seconds",
              "output_image", "output_json", "error"]
    failures = 0
    with (run / "summary.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for number, path in enumerate(paths, 1):
            relative = path.relative_to(source_root)
            start = time.perf_counter()
            try:
                with Image.open(path) as opened:
                    rgb = np.array(ImageOps.exif_transpose(opened).convert("RGB"))
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                height, width = bgr.shape[:2]
                specs = [(0, 0, width, height)] if a.mode == "full" else crop_specs(width, height, a.cols, a.rows)
                detections = []
                for offset in range(0, len(specs), a.batch):
                    chunk = specs[offset:offset + a.batch]
                    crops = [np.ascontiguousarray(bgr[y0:y1, x0:x1]) for x0, y0, x1, y1 in chunk]
                    results = model.predict(source=crops, imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                                            max_det=a.max_det, device=a.device, rect=False,
                                            save=False, verbose=False)
                    for result, (x0, y0, x1, y1) in zip(results, chunk):
                        if result.obb is None:
                            raise RuntimeError("Model did not return OBB results")
                        polygons = result.obb.xyxyxyxy.cpu().numpy()
                        confidences = result.obb.conf.cpu().numpy()
                        classes = result.obb.cls.cpu().numpy().astype(int)
                        for polygon, confidence, cls in zip(polygons, confidences, classes):
                            polygon = polygon + np.array([x0, y0])
                            detections.append({"class_id": int(cls), "class_name": result.names[int(cls)],
                                               "confidence": round(float(confidence), 6),
                                               "corners_xy": polygon.round(3).tolist(),
                                               "crop_xyxy": [x0, y0, x1, y1]})
                annotated = bgr.copy()
                thickness = max(1, round(min(width, height) / 500))
                font_scale = max(0.35, min(width, height) / 1800)
                for detection in detections:
                    corners = np.array(detection["corners_xy"])
                    points = np.rint(corners).astype(np.int32)
                    cv2.polylines(annotated, [points], True, (0, 255, 0), thickness, cv2.LINE_AA)
                    x = int(np.clip(points[:, 0].min(), 0, max(0, width - 1)))
                    y = int(np.clip(points[:, 1].min() - 5, 12, max(12, height - 1)))
                    text = f'{detection["class_name"]} {detection["confidence"]:.2f}'
                    cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                output_image = run / "images" / relative.parent / f"{relative.name}.detected.jpg"
                output_json = run / "detections" / relative.parent / f"{relative.name}.json"
                output_image.parent.mkdir(parents=True, exist_ok=True)
                output_json.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(annotated[:, :, ::-1]).save(output_image, quality=95)
                elapsed = time.perf_counter() - start
                payload = {"source_image": str(path), "model": str(weights), "width": width, "height": height,
                           "settings": settings, "detected_box_count": len(detections), "detections": detections}
                output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                writer.writerow({"image": str(relative), "status": "ok", "detections": len(detections),
                                 "max_confidence": max((d["confidence"] for d in detections), default=""),
                                 "elapsed_seconds": round(elapsed, 3), "output_image": str(output_image),
                                 "output_json": str(output_json), "error": ""})
                print(f"[{number}/{len(paths)}] {relative}: {len(detections)} boxes, {elapsed:.2f} s")
            except Exception as e:
                failures += 1
                writer.writerow({"image": str(relative), "status": "error", "error": str(e)})
                print(f"[{number}/{len(paths)}] ERROR {relative}: {e}", file=sys.stderr)
            csv_file.flush()
    print(f"Output: {run}\nImages: {len(paths)}, errors: {failures}")
    return 1 if failures else 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
