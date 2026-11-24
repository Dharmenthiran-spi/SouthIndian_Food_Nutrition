# Ingredient Identification Version 2 - Converting Code Files

This directory contains the Version 2 codebase for **Ingredient Identification & Comprehensive Nutritional Profiling** across all 221 food categories.

## 📁 Directory Structure
- `build_ingredient_nutrition_v2.py`: Generates the master CSV and JSON datasets containing home vs. hotel ingredient lists and multi-micronutrient profiles for 221 food categories.
- `train_ingredient_predictor_v2.py`: Trains the multi-label ingredient model and saves artifacts into `Incredient_Identification/Models/`.
- `predict_ingredients_v2.py`: CLI and modular prediction script returning detected ingredients and complete nutritional profile (calories, protein, carbs, fat, fiber, sodium, potassium, calcium, iron, vitamin C).

## 🚀 How to Run

### 1. Build Master Dataset
```bash
python build_ingredient_nutrition_v2.py
```

### 2. Train Multi-Label Predictor
```bash
python train_ingredient_predictor_v2.py
```

### 3. Predict Ingredients & Nutrition
```bash
python predict_ingredients_v2.py --food "Masala dosa"
python predict_ingredients_v2.py --food "Chicken biryani"
```

