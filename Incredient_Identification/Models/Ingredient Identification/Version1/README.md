# 🎯 Ingredient Identification Version 1 - Model Weights

This directory stores the official trained deep learning checkpoints for **Model 2: Ingredient Identification & Nutrition Prediction (Version 1)**.

---

## 📦 Checkpoint Naming Standard

All checkpoints follow the strict standardized naming scheme:
`model_name_version_checkpoint.extension`

| Standard File Name | Legacy Alias | Purpose | When to Use |
|---|---|---|---|
| **`ingredient_identification_v1_best.pt`** | `best.pt` | Highest Validation F1 / Lowest Loss Checkpoint | Use for **inference**, evaluation, web app (`app.py`), and production. |
| **`ingredient_identification_v1_last.pt`** | `last.pt` | Latest Epoch Checkpoint | Use to **resume training** if interrupted. |
| **`food_ingredients_nutrition.json`** | - | Fast O(1) Offline Knowledge Lookup | Instant inference for the 31 South Indian food classes. |

---

## 🏗️ Model Architecture Overview

- **Input**: Food Class ID ($0 \dots 30$) / One-Hot Food Representation
- **Shared Representation**: Multi-layer Dense Feature Extractor (ReLU, BatchNorm, Dropout)
- **Head 1 (Ingredient Prediction)**: Multi-label classification with Sigmoid output across 68 canonical ingredients (`BCEWithLogitsLoss`)
- **Head 2 (Nutrition Estimation)**: Multi-target regression predicting `[calories, protein, carbs, fat, fiber]` (`SmoothL1Loss`)

---

## 🚀 Loading in Python

```python
import torch

checkpoint = torch.load("Incredient_Identification/Models/Ingredient Identification/Version1/best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded successfully!")
```

