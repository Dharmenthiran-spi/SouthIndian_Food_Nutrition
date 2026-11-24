"""
Food Detection Version 2 - Inference and Evaluation Script
Runs inference on test images using trained YOLOv8 model and displays detections.
"""

import os
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 Food Detection Version 2 Inference")
    parser.add_argument("--model", type=str, default="Food_Detection/Models/food_detection_v2_best.pt", help="Path to trained .pt weights")
    parser.add_argument("--source", type=str, default="Food_Detection/DataSet/Version2 Used Data/images/test", help="Path to image or image folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated images")
    parser.add_argument("--save-dir", type=str, default="Food_Detection/runs/inference_v2", help="Directory to save outputs")
    return parser.parse_args()

def run_inference():
    args = parse_args()
    
    # Resolve model path
    model_path = args.model
    if not os.path.exists(model_path):
        alt = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Models", "food_detection_v2_best.pt"))
        if os.path.exists(alt):
            model_path = alt
        else:
            print(f"⚠️ Model not found at '{model_path}'. Using base yolov8s.pt for demonstration.")
            model_path = "yolov8s.pt"
            
    print(f"🚀 Loading Model: {model_path}")
    model = YOLO(model_path)
    
    print(f"🔍 Running inference on: {args.source} (Confidence threshold: {args.conf})")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        project=args.save_dir,
        name="test_results",
        exist_ok=True
    )
    
    print(f"\n✅ Inference completed for {len(results)} image(s)!")
    print(f"📁 Results saved to: {args.save_dir}/test_results")

if __name__ == "__main__":
    run_inference()

