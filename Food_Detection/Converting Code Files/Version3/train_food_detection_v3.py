"""
Food Detection Version 3 - YOLOv8 Training Script (221 Classes + Multi-Dish Plates)
Fine-tunes from Version 2 weights (if available) or yolov8s.pt on Version 3 dataset
containing single-dish images, 500 real multi-dish Thali/Banana-leaf images, and
600 synthetic multi-dish combo plates.
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO


def train_v3(epochs=100, batch=16, imgsz=640):
    base_dir = Path(__file__).resolve().parents[2]
    data_yaml = base_dir / "DataSet" / "Version3 Used Data" / "dataset.yaml"
    v2_weights = base_dir / "Models" / "Food Detection" / "Version2" / "food_detection_v2_best.pt"
    out_dir = base_dir / "Models" / "Food Detection" / "Version3"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    model_src = str(v2_weights) if v2_weights.exists() else "yolov8s.pt"
    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("🚀 FOOD DETECTION VERSION 3 TRAINING (221 CLASSES + MULTI-DISH)")
    print("=" * 65)
    print(f"Base Weights : {model_src}")
    print(f"Dataset YAML : {data_yaml}")
    print(f"Device       : {device}")
    print("=" * 65)

    model = YOLO(model_src)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(base_dir / "runs" / "train"),
        name="food_detection_v3_run",
        patience=15,
        mosaic=1.0,
        mixup=0.1,
        save=True,
        save_period=5,
        exist_ok=True,
        plots=True,
    )
    return results


if __name__ == "__main__":
    train_v3()

