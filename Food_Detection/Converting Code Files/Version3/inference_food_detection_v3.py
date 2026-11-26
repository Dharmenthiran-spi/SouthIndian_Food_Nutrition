"""
Food Detection Version 3 - Inference Script (Single & Multi-Dish Plates)
"""

import sys
from pathlib import Path
from ultralytics import YOLO


def predict_plate(image_path: str, conf: float = 0.22):
    base_dir = Path(__file__).resolve().parents[2]
    v3_weights = base_dir / "Models" / "Food Detection" / "Version3" / "food_detection_v3_best.pt"
    v2_weights = base_dir / "Models" / "Food Detection" / "Version2" / "food_detection_v2_best.pt"

    weights = v3_weights if v3_weights.exists() else v2_weights
    if not weights.exists():
        raise FileNotFoundError("No Version 3 or Version 2 weights found.")

    model = YOLO(str(weights))
    results = model.predict(source=image_path, conf=conf, iou=0.45, verbose=False)
    for b in results[0].boxes:
        cls_id = int(b.cls[0])
        score = float(b.conf[0])
        print(f"🍽️ {results[0].names[cls_id]} ({score*100:.1f}%)")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_food_detection_v3.py <image_path>")
    else:
        predict_plate(sys.argv[1])

