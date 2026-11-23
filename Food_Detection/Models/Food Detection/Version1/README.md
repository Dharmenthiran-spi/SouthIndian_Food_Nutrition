# 🎯 Food Detection Version 1 - Model Weights

This directory stores the official trained model checkpoints for **Food Detection Version 1** (31 South Indian food categories).

---

## 📦 Checkpoint Types

| File Name | Purpose | When to Use |
|---|---|---|
| **`best.pt`** | Highest Validation mAP Model | Use for **inference**, evaluation, mobile/web deployment, and production. |
| **`last.pt`** | Latest Epoch Checkpoint | Use to **resume training** if a training run or Google Colab session is interrupted. |
| **`best.onnx`** | Cross-Platform Export | Optimized ONNX Runtime model for CPU, mobile apps, or web backends. |

---

## 🚀 How to Use

### 1. Run Inference with `best.pt`

```python
from ultralytics import YOLO

# Load the best model
model = YOLO('Food_Detection/Models/Food Detection/Version1/best.pt')

# Run prediction
results = model.predict(source='path/to/food_image.jpg', conf=0.25, save=True)
```

### 2. Resume Interrupted Training with `last.pt`

```python
from ultralytics import YOLO

# Load the last checkpoint
model = YOLO('Food_Detection/Models/Food Detection/Version1/last.pt')

# Resume training from the exact interrupted epoch
results = model.train(resume=True)
```

Or via CLI:

```bash
yolo detect train resume model="Food_Detection/Models/Food Detection/Version1/last.pt"
```

