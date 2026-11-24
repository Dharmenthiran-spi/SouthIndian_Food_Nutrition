"""
Food Detection Version 2 - YOLOv8 Training Script (221 Food Categories)
Supports:
- Local GPU/CPU training
- Google Colab GPU training (T4, A100, L4, V100)
- Auto dataset path resolution
- Loss and mAP curve plotting
- Best weights (.pt) and ONNX export
"""

import os
import sys
import argparse
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Food Detection Version 2 (221 Classes)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce if CUDA Out of Memory)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training and validation")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Base pretrained YOLOv8 model")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker threads")
    parser.add_argument("--name", type=str, default="food_detection_v2", help="Run experiment name")
    return parser.parse_args()

def find_dataset_yaml():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(base_dir, "DataSet", "Version2 Used Data", "dataset.yaml"),
        "Food_Detection/DataSet/Version2 Used Data/dataset.yaml",
        "/content/SouthIndian_Food_Nutrition/Food_Detection/DataSet/Version2 Used Data/dataset.yaml"
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Could not locate dataset.yaml in candidates: {candidates}")

def configure_yaml(yaml_path):
    dataset_dir = os.path.dirname(yaml_path)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    data['path'] = dataset_dir
    data['train'] = "images/train"
    data['val'] = "images/val"
    data['test'] = "images/test"
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False)
        
    print(f"✅ Configured dataset.yaml at: {yaml_path}")
    print(f"📁 Dataset Path: {dataset_dir}")
    print(f"🔥 Total Classes: {len(data.get('names', {}))}")
    return yaml_path

def train():
    args = parse_args()
    print("=" * 60)
    print("🚀 FOOD DETECTION VERSION 2 TRAINING (221 CLASSES)")
    print("=" * 60)
    
    yaml_path = find_dataset_yaml()
    configure_yaml(yaml_path)
    
    print(f"\n📦 Loading Base Model: {args.model}")
    model = YOLO(args.model)
    
    print(f"\n🔥 Starting Training:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Device: {args.device}")
    
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        patience=15,
        save=True,
        save_period=5,
        exist_ok=True,
        plots=True,
        verbose=True
    )
    
    print("\n✅ Training Complete!")
    save_dir = model.trainer.save_dir
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    
    if os.path.exists(best_pt):
        print(f"🏆 Best Weights Saved At: {best_pt}")
        
        # Save copy into Models folder
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Models"))
        os.makedirs(models_dir, exist_ok=True)
        dest_pt = os.path.join(models_dir, "food_detection_v2_best.pt")
        import shutil
        shutil.copy(best_pt, dest_pt)
        print(f"📦 Saved Best Model to: {dest_pt}")
        
        # Export ONNX
        print("\n⚡ Exporting Best Model to ONNX format...")
        try:
            onnx_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
            dest_onnx = os.path.join(models_dir, "food_detection_v2_best.onnx")
            if os.path.exists(onnx_path):
                shutil.copy(onnx_path, dest_onnx)
                print(f"⚡ ONNX Model Exported: {dest_onnx}")
        except Exception as e:
            print(f"⚠️ ONNX Export note: {e}")
            
    # Display plot paths
    results_png = os.path.join(save_dir, "results.png")
    if os.path.exists(results_png):
        print(f"📊 Training curves plot: {results_png}")

if __name__ == "__main__":
    train()

