# 🥗 South Indian Food Ingredient & Nutrition Dataset (Version 1)

This dataset serves as the knowledge & prediction ground truth for **Model 2: Ingredient Identification & Nutrition Module**.

## 🔄 Project Pipeline Integration
```
[ Food Image ] 
      │
      ▼
[ Model 1: Food Detection (YOLOv8) ] ──▶ Predicts: "Food Name" (31 Classes)
      │
      ▼
[ Model 2: Ingredient Identification & Nutrition Module ]
      │
      ├─▶ Identifies: Main Authentic Used Ingredients (e.g. Rice, Urad Dal, Fenugreek)
      ├─▶ Computes: Full Nutritional Breakdown (Calories, Protein, Carbs, Fats, Fiber)
      └─▶ Evaluates: Dietary Profile & Health Benefits (Probiotic, Gluten-Free, Low Fat)
```

## 📊 Dataset Specifications
- **Classes Covered**: Exactly the **31 South Indian food classes** from `Food_Detection`
- **Total Unique Ingredients**: 68 canonical culinary ingredients
- **Files Included**:
  - `south_indian_food_ingredients_nutrition.csv`: Master tabular dataset with all nutritional metrics
  - `dishes_ingredients_map.json`: Fast $O(1)$ lookup mapping for real-time app inference
  - `ingredients_vocabulary.csv`: Vocabulary list of ingredients ranked by dish frequency
  - `dataset.yaml`: Dataset configuration and class IDs

## 📋 The 31 Dishes Covered
Appam, Beetroot poriyal, Boiled Egg, Carrot poriyal, Chicken 65, Chicken briyani, Dosa, Idly, Kaara chutney, Kali, Koozh, Lemon Rice, Mushroom briyani, Mutton Briyani, Nandu masala, Nei satham, Paal kolukattai, Paneer briyani, Panner masala, Parupu vada, Pidi kolukattai, Poorna kolukattai, Prawn thokku, Puthina Chutney, Sambar, Sambar satham, Satham, Thengai chutney, Uzhuntha vadai, Veg briyani, Ven Pongal
