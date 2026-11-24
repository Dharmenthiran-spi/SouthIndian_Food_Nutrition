"""
Unified Version 2 Food Detection & Category Identification Engine.
Detects foods from any image (single item or multi-food banana leaf / thali plate),
identifying:
1. Food Name & Class ID (0..220)
2. Exact Taxonomy Category & Functional Group
3. Dietary Classification (Vegetarian, Non-Vegetarian, Vegan, Eggetarian)
4. Fitness & Gym Recommendation (Muscle Building, Weight Loss, Weight Gain)
5. Authentic Ingredient Breakdown & Nutrition (Home vs Hotel Styles)
"""

import os
import sys
import json
import yaml
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

DATASET_YAML = os.path.join(CURRENT_DIR, "DataSet", "Version2 Used Data", "dataset.yaml")
DISHES_MAP_V2 = os.path.join(PROJECT_ROOT, "Incredient_Identification", "DataSet", "Version2 Used Data", "dishes_ingredients_map_v2.json")
HOME_HOTEL_V2 = os.path.join(PROJECT_ROOT, "Incredient_Identification", "DataSet", "Version2 Used Data", "home_hotel_recipes_v2.json")

class FoodDetectorV2:
    def __init__(self, model_path=None, conf_threshold=0.25):
        self.conf_threshold = conf_threshold
        self.classes = {}
        self.dishes_info = {}
        self.home_hotel = {}
        self.model = None

        # 1. Load class mapping from dataset.yaml
        if os.path.exists(DATASET_YAML):
            with open(DATASET_YAML, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                self.classes = cfg.get("names", {})
        
        # 2. Load V2 Ingredient & Nutrition Knowledge Graph
        if os.path.exists(DISHES_MAP_V2):
            with open(DISHES_MAP_V2, "r", encoding="utf-8") as f:
                self.dishes_info = json.load(f)
                
        if os.path.exists(HOME_HOTEL_V2):
            with open(HOME_HOTEL_V2, "r", encoding="utf-8") as f:
                self.home_hotel = json.load(f)

        # 3. Load YOLO model if available
        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"[FoodDetectorV2] Loaded YOLO model from {model_path}")
            except Exception as e:
                print(f"[FoodDetectorV2] Warning: Could not load YOLO model: {e}")

    def get_food_metadata(self, food_name_or_id):
        """O(1) lookup of comprehensive food name, category, macros, and ingredients."""
        name = None
        if isinstance(food_name_or_id, int):
            name = self.classes.get(food_name_or_id)
        elif isinstance(food_name_or_id, str):
            name = food_name_or_id

        if not name:
            return None

        dish_data = self.dishes_info.get(name, {})
        recipe_data = self.home_hotel.get(name, {})

        return {
            "food_name": name,
            "category": dish_data.get("category", "General Food"),
            "group_category": dish_data.get("group_category", "Standard Category"),
            "diet_type": dish_data.get("diet_type", "Standard"),
            "fitness_goal": dish_data.get("fitness_goal", "Balanced Nutrition"),
            "portion_size": dish_data.get("portion_size", "1 portion"),
            "nutrition_per_100g": dish_data.get("nutrition_per_100g", {}),
            "ingredients": dish_data.get("main_ingredients", []),
            "ingredient_display": dish_data.get("ingredient_display", ""),
            "cooking_method": dish_data.get("cooking_method", ""),
            "home_recipe": recipe_data.get("home", {}),
            "hotel_recipe": recipe_data.get("hotel", {})
        }

    def detect(self, image_path_or_array):
        """
        Runs object detection on the image (supporting single dishes and banana leaf thali platters).
        Returns a list of detected items, each enriched with food name, category, and ingredient nutrition.
        """
        results = []
        if self.model is None:
            # Fallback mock detection for testing or when model is not loaded
            print("[FoodDetectorV2] Note: No YOLO model weights loaded. Provide model_path to run active neural inference.")
            return results

        yolo_res = self.model.predict(image_path_or_array, conf=self.conf_threshold, verbose=False)
        for r in yolo_res:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = [round(x, 1) for x in box.xyxy[0].tolist()]
                
                food_name = self.classes.get(cls_id, f"Class_{cls_id}")
                meta = self.get_food_metadata(food_name)

                item = {
                    "class_id": cls_id,
                    "food_name": food_name,
                    "confidence": round(conf, 3),
                    "box_xyxy": xyxy,
                    "category": meta["category"] if meta else "Food",
                    "group_category": meta["group_category"] if meta else "Standard",
                    "diet_type": meta["diet_type"] if meta else "Standard",
                    "fitness_goal": meta["fitness_goal"] if meta else "Balanced Nutrition",
                    "portion_size": meta["portion_size"] if meta else "1 serving",
                    "nutrition_per_100g": meta["nutrition_per_100g"] if meta else {},
                    "ingredients": meta["ingredients"] if meta else [],
                    "ingredient_display": meta["ingredient_display"] if meta else "",
                    "home_nutrition": meta["home_recipe"].get("nutrition", {}) if meta else {},
                    "hotel_nutrition": meta["hotel_recipe"].get("nutrition", {}) if meta else {}
                }
                results.append(item)

        return results

def main():
    detector = FoodDetectorV2()
    print("==================================================================")
    print("Version 2 Unified Food Detection & Ingredient Identification Test")
    print("==================================================================")
    print(f"Total Configured Classes : {len(detector.classes)}")
    print(f"Dishes in Knowledge Base : {len(detector.dishes_info)}")
    print(f"Recipes in Home/Hotel Map: {len(detector.home_hotel)}")
    
    # Test sample lookups across categories
    samples = [
        0,    # Idli
        59,   # Chicken biryani
        78,   # Burger
        96,   # Carrot
        118,  # Mango
        143,  # Almonds
        211   # Whey protein
    ]
    
    print("\n--- Testing Sample Class Identification & Categorization ---")
    for s in samples:
        meta = detector.get_food_metadata(s)
        if meta:
            print(f"\n[Food ID {s:3d}] {meta['food_name']}")
            print(f"  Category    : {meta['category']} ({meta['group_category']})")
            print(f"  Diet Type   : {meta['diet_type']}")
            print(f"  Fitness Goal: {meta['fitness_goal']}")
            print(f"  Ingredients : {meta['ingredient_display']}")
            nut = meta['nutrition_per_100g']
            print(f"  Macros/100g : {nut.get('calories_kcal', 0)} kcal | P: {nut.get('protein_g', 0)}g | C: {nut.get('carbohydrates_g', 0)}g | F: {nut.get('fat_g', 0)}g")

if __name__ == "__main__":
    main()

