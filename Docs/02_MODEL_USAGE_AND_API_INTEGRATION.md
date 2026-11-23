# 🔌 Model Usage, Inputs, Outputs & Application Reuse Guide

This document details **how to call both Model 1 and Model 2**, what parameters to pass, expected input and output data shapes, and how to integrate both models into external applications (FastAPI/Flask REST APIs, Mobile Apps, and Python microservices).

---

## ⚡ Quick Cheat Sheet: How to Call Each Function & Parameters

| Task | Function / Method | Required Parameters | Optional Parameters | Return Value |
|:---|:---|:---|:---|:---|
| **Detect Food from Photo** (Model 1) | `YOLO(weights).predict(...)` | `source` (image path, PIL image, OpenCV array) | `conf=0.25`, `iou=0.45`, `imgsz=640` | YOLO Results object (boxes, classes, confidence, plot) |
| **Predict Ingredients & Nutrition** (Model 2) | `predict_with_model(food_name)` | `food_name` (`"Idly"`, `"Dosa"`, `"Ven Pongal"`, etc.) | None | Dict: predicted ingredients, confidence %, 5 macro nutrients, health profile |
| **Hotel Proportional Split** (Model 2) | `calculate_hotel_proportional(...)` | `food_name`, `target_weight_g` (e.g. `300.0`) | None | Dict: split ingredient weights (grams) & scaled restaurant nutrition |
| **Home Custom Recipe** (Model 2) | `calculate_custom_nutrition(...)` | `food_name`, `custom_ingredients_dict` | `additional_ingredients_dict` (e.g. `{"Extra Ghee": 10}`) | Dict: calculated total cooked weight & live customized nutrition |

---

## 1. 🎯 Model 1: Food Plate Object Detector (YOLOv8)

### Weights Path:
- `Food_Detection/Models/Food Detection/Version1/food_detection_v1_best.pt`
- File Size: ~22.5 MB (YOLOv8 Small)
- Architecture: 31 South Indian Food Classes

### Input Requirements:
| Parameter | Supported Types | Details |
|:---|:---|:---|
| `source` | `str`, `PIL.Image`, `np.ndarray`, `torch.Tensor` | File path to image (`.jpg`, `.png`), loaded PIL image, OpenCV BGR image array, or batch tensor. |
| `conf` | `float` (Default: `0.25`) | Minimum confidence threshold to accept detection (0.10 to 1.00). |
| `iou` | `float` (Default: `0.45`) | Intersection-over-Union threshold for Non-Maximum Suppression (NMS). |
| `imgsz` | `int` (Default: `640`) | Input image resolution for inference. |

### Python Code Example:
```python
from ultralytics import YOLO
from PIL import Image

# 1. Load Model 1
model = YOLO("Food_Detection/Models/Food Detection/Version1/food_detection_v1_best.pt")

# 2. Run Inference
results = model.predict(source="path/to/food_plate.jpg", conf=0.25, iou=0.45)
first_res = results[0]

# 3. Extract Detections
for box in first_res.boxes:
    class_id = int(box.cls[0])
    food_name = first_res.names[class_id]
    confidence = float(box.conf[0]) * 100
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
    
    print(f"🍽️ Detected: {food_name} ({confidence:.1f}%) | BBox: [{x1}, {y1}, {x2}, {y2}]")
```

### Output Data Format:
```json
{
  "total_detections": 2,
  "items": [
    {
      "class_id": 30,
      "food_name": "Ven Pongal",
      "confidence_percent": 91.4,
      "bounding_box": [45, 80, 320, 410]
    },
    {
      "class_id": 24,
      "food_name": "Sambar",
      "confidence_percent": 84.7,
      "bounding_box": [340, 95, 580, 390]
    }
  ]
}
```

---

## 2. 🌿 Model 2: Ingredient Identification & Nutrition Predictor

### Weights & Engine:
- `Incredient_Identification/Models/Ingredient Identification/Version1/ingredient_identification_v1_best.pt`
- File Size: ~1.2 MB
- Core Module: `Incredient_Identification/ingredient_predictor.py`

### Input Requirements:
| Parameter | Type | Details |
|:---|:---|:---|
| `food_name` | `str` | Name of the food item (e.g. `"Ven Pongal"`, `"Chicken briyani"`, `"Dosa"`, `"Idly"`). Automatically normalizes spelling and aliases (`"briyani"` $\leftrightarrow$ `"biryani"`). |
| `custom_ingredients_dict` | `dict[str, float]` | *(Optional for Home Custom)* Dict of ingredient names to typed weights in grams. |
| `additional_ingredients_dict` | `dict[str, float]` | *(Optional for Home Custom)* Dict of extra added ingredients (e.g. `{"Extra Ghee": 15.0}`). |
| `target_weight_g` | `float` | *(Optional for Hotel Proportional)* Total serving weight in grams to proportionally split ingredients. |

### Python Code Example:
```python
from Incredient_Identification.ingredient_predictor import (
    predict_with_model,
    calculate_hotel_proportional,
    calculate_custom_nutrition
)

# 1. Predict Ingredients & Baseline Nutrition using Neural Network
res = predict_with_model("Ven Pongal")
print(f"Dish: {res['food_name']} ({res['regional_name']})")
print("Top Ingredients:", res["predicted_ingredients"][:5])
print("Macros:", res["nutrition"])

# 2. Hotel Proportional Scaling (e.g., user types 300g serving)
hotel_calc = calculate_hotel_proportional("Ven Pongal", target_weight_g=300.0)
print("\nHotel Split Ingredients:", hotel_calc["split_ingredients"])
print("Hotel Nutrition (300g):", hotel_calc["nutrition"])

# 3. Home Custom Typing & Extra Ingredients (e.g., user adds 15g cashews)
home_calc = calculate_custom_nutrition(
    food_name="Ven Pongal",
    custom_ingredients_dict={"raw_rice": 90, "yellow_moong_dal": 50, "cow_ghee": 15, "cashews_pepper": 10},
    additional_ingredients_dict={"Fried Cashews": 15.0}
)
print("\nHome Custom Cooked Weight:", home_calc["total_weight_g"], "g")
print("Home Custom Nutrition:", home_calc["nutrition"])
```

### Output Data Format:
```json
{
  "food_name": "Ven Pongal",
  "regional_name": "வெண் பொங்கல் / Khara Pongal",
  "category": "Breakfast / Tiffin",
  "estimated_weight_g": 180,
  "weight_range_desc": "~180g (Home) to ~240g (Hotel)",
  "predicted_ingredients": [
    ["Rice", 0.998, 85],
    ["Moong Dal", 0.955, 45],
    ["Ghee", 0.996, 12],
    ["Cashew Nuts", 0.994, 15],
    ["Black Pepper", 0.996, 3],
    ["Curry Leaves", 1.0, 3]
  ],
  "nutrition": {
    "calories_kcal": 343.8,
    "protein_g": 8.2,
    "carbs_g": 49.0,
    "fat_g": 12.6,
    "fiber_g": 3.5
  }
}
```

---

## 3. 🌐 Reusing in Other Applications: REST API Microservice (FastAPI)

Below is a complete, production-ready REST API microservice you can run in any backend service:

```python
# main_api.py - Run with: uvicorn main_api:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from ultralytics import YOLO
from Incredient_Identification.ingredient_predictor import (
    predict_with_model,
    calculate_hotel_proportional,
    calculate_custom_nutrition
)

app = FastAPI(title="South Indian Food Inspection & Nutrition API", version="1.0")

# Load YOLO model once on startup
yolo_model = YOLO("Food_Detection/Models/Food Detection/Version1/food_detection_v1_best.pt")

@app.post("/api/v1/inspect-food-plate")
async def inspect_food_plate(file: UploadFile = File(...)):
    """Receives image upload, detects foods, and returns ingredients & nutrition."""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 1. Run Model 1 Detection
    results = yolo_model.predict(source=image, conf=0.25, verbose=False)
    first_res = results[0]
    
    detected_dishes = []
    for box in first_res.boxes:
        cid = int(box.cls[0])
        name = first_res.names[cid]
        conf = float(box.conf[0])
        bbox = [int(v) for v in box.xyxy[0].tolist()]
        
        # 2. Enrich with Model 2 Intelligence
        m2_info = predict_with_model(name)
        
        detected_dishes.append({
            "food_name": name,
            "confidence": round(conf, 3),
            "bounding_box": bbox,
            "regional_name": m2_info["regional_name"] if m2_info else name,
            "ingredients": m2_info["predicted_ingredients"] if m2_info else [],
            "baseline_nutrition": m2_info["nutrition"] if m2_info else {}
        })
        
    return {"status": "success", "total_detected": len(detected_dishes), "dishes": detected_dishes}

@app.post("/api/v1/calculate-nutrition")
def calculate_nutrition(payload: dict):
    """
    Calculates dynamic Home or Hotel nutrition.
    Payload:
      {"food_name": "Chicken briyani", "prep_mode": "hotel", "total_weight_g": 350}
    OR:
      {"food_name": "Chicken briyani", "prep_mode": "home", "ingredients": {...}, "extras": {...}}
    """
    food_name = payload.get("food_name")
    prep_mode = payload.get("prep_mode", "home")
    
    if prep_mode == "hotel":
        target_wt = payload.get("total_weight_g", 300.0)
        return calculate_hotel_proportional(food_name, target_wt)
    else:
        ings = payload.get("ingredients", {})
        extras = payload.get("extras", {})
        return calculate_custom_nutrition(food_name, ings, extras)
```

---

## 4. 📱 Edge & Mobile Deployment (ONNX Format)

Both models can be exported to cross-platform **ONNX** runtime for deployment on Android, iOS, React Native, Flutter, and embedded IoT edge cameras:

### Exporting Model 1 to ONNX:
```python
from ultralytics import YOLO

model = YOLO("Food_Detection/Models/Food Detection/Version1/food_detection_v1_best.pt")
model.export(format="onnx", dynamic=True, simplify=True)
# Outputs: food_detection_v1_best.onnx (~22 MB)
```

### Exporting Model 2 to ONNX:
```python
import torch
from Incredient_Identification.ingredient_predictor import get_predictor

p = get_predictor()
dummy_food_id = torch.tensor([0], dtype=torch.long)

torch.onnx.export(
    p.model,
    dummy_food_id,
    "ingredient_identification_v1.onnx",
    input_names=["food_id"],
    output_names=["ingredient_logits", "nutrition_normalized"],
    dynamic_axes={"food_id": {0: "batch_size"}}
)
```
The resulting ONNX files can be loaded via `onnxruntime` or `onnxruntime-react-native` with zero PyTorch or Ultralytics runtime dependencies.

