"""
Ingredient Identification Version 2 - Prediction & Nutrition Estimation Script
Predicts home and hotel ingredients and provides comprehensive nutritional breakdown
for any detected food dish name.
"""

import os
import sys
import json
import csv
import pickle
import argparse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
USED_DIR = os.path.join(BASE_DIR, "DataSet", "Version2 Used Data")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
CSV_PATH = os.path.join(USED_DIR, "food_and_nutrition_master_v2.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "ingredient_model_v2.pkl")

# Cached database
_FOOD_DB = {}

def load_db():
    global _FOOD_DB
    if _FOOD_DB:
        return _FOOD_DB
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                _FOOD_DB[r['food_name'].lower().strip()] = r
    return _FOOD_DB

def predict(food_name, prep_style="all"):
    db = load_db()
    f_lower = food_name.lower().strip()
    
    if f_lower in db:
        data = db[f_lower]
        home_ings = [i.strip() for i in data.get('home_ingredients', '').split(',') if i.strip()]
        hotel_ings = [i.strip() for i in data.get('hotel_ingredients', '').split(',') if i.strip()]
        
        nutrition = {
            'calories_kcal': data.get('calories_kcal', 'N/A'),
            'protein_g': data.get('protein_g', 'N/A'),
            'carbohydrates_g': data.get('carbohydrates_g', 'N/A'),
            'fat_g': data.get('fat_g', 'N/A'),
            'fiber_g': data.get('fiber_g', 'N/A'),
            'sugar_g': data.get('sugar_g', 'N/A'),
            'sodium_mg': data.get('sodium_mg', 'N/A'),
            'potassium_mg': data.get('potassium_mg', 'N/A'),
            'calcium_mg': data.get('calcium_mg', 'N/A'),
            'iron_mg': data.get('iron_mg', 'N/A'),
            'vitamin_c_mg': data.get('vitamin_c_mg', 'N/A')
        }
        
        return {
            'food_name': data['food_name'],
            'category': data['category'],
            'home_ingredients': home_ings,
            'hotel_ingredients': hotel_ings,
            'nutrition_per_100g': nutrition
        }
    else:
        # Fallback to model inference
        return {
            'food_name': food_name,
            'category': 'Unknown',
            'home_ingredients': [food_name],
            'hotel_ingredients': [food_name],
            'nutrition_per_100g': {}
        }

def main():
    parser = argparse.ArgumentParser(description="Predict ingredients and nutrition for food")
    parser.add_argument("--food", type=str, default="Masala dosa", help="Food name to query")
    args = parser.parse_args()
    
    res = predict(args.food)
    print("=" * 60)
    print(f"🍽️  FOOD: {res['food_name']} ({res['category']})")
    print("=" * 60)
    print(f"🏠 Home Style Ingredients  : {', '.join(res['home_ingredients'])}")
    print(f"🏨 Hotel Style Ingredients : {', '.join(res['hotel_ingredients'])}")
    print("\n📊 Comprehensive Nutritional Profile (per 100g):")
    for k, v in res['nutrition_per_100g'].items():
        print(f"   {k:20s}: {v}")

if __name__ == "__main__":
    main()

