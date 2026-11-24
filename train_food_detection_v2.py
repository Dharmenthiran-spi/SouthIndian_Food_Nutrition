"""
Food Detection Version 2 Training Script (YOLOv8)
Trains a brand new YOLOv8 object detection model on all 221 food & nutrition classes.
Dataset: Food_Detection/DataSet/Version2 Used Data/dataset.yaml
Total Classes: 221
Total Images: 8,932 (Train: 6,843, Val: 1,053, Test: 1,036)
"""

import os
import argparse
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "Food_Detection", "DataSet", "Version2 Used Data", "dataset.yaml")

def train_model(epochs=100, batch_size=16, imgsz=640, device="0", weights="yolov8s.pt"):
    print("=" * 60)
    print("🚀 FOOD DETECTION VERSION 2 TRAINING PIPELINE (221 CLASSES)")
    print("=" * 60)
    print(f"Dataset YAML: {DATA_YAML}")
    print(f"Base Weights: {weights}")
    print(f"Epochs:       {epochs}")
    print(f"Batch Size:   {batch_size}")
    print(f"Image Size:   {imgsz}")
    print(f"Device:       {device}")
    print("=" * 60)

    # Initialize fresh model from pretrained backbone
    model = YOLO(weights)

    # Train model
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project="runs/detect/Food_Detection_V2",
        name="yolov8_221classes",
        save=True,
        plots=True,
        verbose=True
    )

    print("\nTraining completed successfully!")
    print(f"Best model weights saved in runs/detect/Food_Detection_V2/yolov8_221classes/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Food Detection Version 2 Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce to 8 or 4 if CUDA out of memory)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="GPU device id or 'cpu'")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Starting backbone weights")

    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        weights=args.weights
    )

