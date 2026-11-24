"""
Ingredient & Nutrition Predictor Module
Provides both:
1. PyTorch Deep Learning Model Inference (ingredient_identification_v1_best.pt)
2. Fast O(1) Knowledge Database Lookup (dishes_ingredients_map.json)

Usage:
    from Incredient_Identification.ingredient_predictor import get_ingredient_nutrition, predict_with_model
    result = predict_with_model("Idly")
"""

import os
import json
import re
import torch
import torch.nn as nn
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Prioritize Version 2 datasets (221 classes), fallback to Version 1 (31 classes)
V2_JSON_PATH = os.path.join(CURRENT_DIR, "DataSet", "Version2 Used Data", "dishes_ingredients_map_v2.json")
V1_JSON_PATH = os.path.join(CURRENT_DIR, "DataSet", "Version1 Used Data", "dishes_ingredients_map.json")
JSON_PATH = V2_JSON_PATH if os.path.exists(V2_JSON_PATH) else V1_JSON_PATH

V2_HOME_HOTEL = os.path.join(CURRENT_DIR, "DataSet", "Version2 Used Data", "home_hotel_recipes_v2.json")
V1_HOME_HOTEL = os.path.join(CURRENT_DIR, "DataSet", "Version1 Used Data", "home_hotel_recipes.json")
HOME_HOTEL_PATH = V2_HOME_HOTEL if os.path.exists(V2_HOME_HOTEL) else V1_HOME_HOTEL

MODEL_PATHS = [
    os.path.join(CURRENT_DIR, "Models", "Ingredient Identification", "Version2", "best.pt"),
    os.path.join(CURRENT_DIR, "Models", "Ingredient Identification", "Version1", "ingredient_identification_v1_best.pt"),
    os.path.join(CURRENT_DIR, "Models", "Ingredient Identification", "Version1", "best.pt")
]

# Standard Nutritional Density (per 100g raw/base component) for dynamic adjustments
INGREDIENT_MACRO_MAP = {
    "rice": {"cal": 350, "p": 7.0, "c": 78.0, "f": 0.5, "fib": 1.0},
    "dal": {"cal": 340, "p": 24.0, "c": 60.0, "f": 1.5, "fib": 9.0},
    "urad": {"cal": 340, "p": 24.0, "c": 59.0, "f": 1.4, "fib": 9.8},
    "toor": {"cal": 335, "p": 22.0, "c": 62.0, "f": 1.5, "fib": 9.0},
    "moong": {"cal": 345, "p": 24.0, "c": 60.0, "f": 1.2, "fib": 8.0},
    "chana": {"cal": 360, "p": 21.0, "c": 60.0, "f": 5.0, "fib": 15.0},
    "oil": {"cal": 884, "p": 0.0, "c": 0.0, "f": 100.0, "fib": 0.0},
    "ghee": {"cal": 890, "p": 0.0, "c": 0.0, "f": 99.5, "fib": 0.0},
    "butter": {"cal": 717, "p": 0.8, "c": 0.1, "f": 81.0, "fib": 0.0},
    "chicken": {"cal": 165, "p": 31.0, "c": 0.0, "f": 3.6, "fib": 0.0},
    "mutton": {"cal": 250, "p": 25.0, "c": 0.0, "f": 16.0, "fib": 0.0},
    "prawn": {"cal": 105, "p": 20.0, "c": 0.5, "f": 1.5, "fib": 0.0},
    "crab": {"cal": 97, "p": 19.0, "c": 0.0, "f": 1.5, "fib": 0.0},
    "egg": {"cal": 145, "p": 12.5, "c": 1.0, "f": 10.0, "fib": 0.0},
    "paneer": {"cal": 296, "p": 18.0, "c": 4.5, "f": 22.0, "fib": 0.0},
    "mushroom": {"cal": 22, "p": 3.1, "c": 3.3, "f": 0.3, "fib": 1.0},
    "coconut": {"cal": 354, "p": 3.3, "c": 15.0, "f": 33.5, "fib": 9.0},
    "sugar": {"cal": 387, "p": 0.0, "c": 100.0, "f": 0.0, "fib": 0.0},
    "jaggery": {"cal": 383, "p": 0.4, "c": 95.0, "f": 0.1, "fib": 0.0},
    "milk": {"cal": 65, "p": 3.3, "c": 4.8, "f": 3.7, "fib": 0.0},
    "cashew": {"cal": 553, "p": 18.2, "c": 30.2, "f": 43.8, "fib": 3.3},
    "veg": {"cal": 40, "p": 1.8, "c": 8.0, "f": 0.3, "fib": 3.0},
    "default": {"cal": 100, "p": 2.0, "c": 15.0, "f": 1.0, "fib": 1.5}
}

class SouthIndianIngredientNutritionModel(nn.Module):
    def __init__(self, num_foods=31, num_ingredients=68, embed_dim=64, num_nutrition=5):
        super().__init__()
        self.food_embedding = nn.Embedding(num_foods, embed_dim)
        self.shared_trunk = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.ingredient_head = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, num_ingredients)
        )
        self.nutrition_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_nutrition)
        )

    def forward(self, food_ids):
        embed = self.food_embedding(food_ids)
        shared = self.shared_trunk(embed)
        return self.ingredient_head(shared), self.nutrition_head(shared)

class IngredientNutritionPredictor:
    def __init__(self, data_path=None):
        path = data_path or JSON_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ingredient nutrition database not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self.database = json.load(f)
        
        # Build normalized index
        self.normalized_index = {}
        for key, val in self.database.items():
            norm_key = self._normalize(key)
            self.normalized_index[norm_key] = val
            alias = norm_key.replace("briyani", "biryani").replace("idly", "idli").replace("satham", "rice")
            self.normalized_index[alias] = val

        # Lazy load PyTorch Model 2
        self.model = None
        self.checkpoint = None
        self._load_model()

        # Load Home vs Hotel Recipe Database
        self.home_hotel_db = {}
        self.home_hotel_index = {}
        if os.path.exists(HOME_HOTEL_PATH):
            with open(HOME_HOTEL_PATH, "r", encoding="utf-8") as f:
                self.home_hotel_db = json.load(f)
            for k, v in self.home_hotel_db.items():
                nk = self._normalize(k)
                self.home_hotel_index[nk] = v
                alias = nk.replace("briyani", "biryani").replace("idly", "idli").replace("satham", "rice")
                self.home_hotel_index[alias] = v

    def _normalize(self, text):
        t = text.lower()
        t = re.sub(r'[^a-z0-9]', '', t)
        return t

    def _load_model(self):
        chosen_path = None
        for p in MODEL_PATHS:
            if os.path.exists(p):
                chosen_path = p
                break
        if chosen_path:
            try:
                ckpt = torch.load(chosen_path, map_location="cpu")
                num_foods = len(ckpt.get("food2idx", {})) or ckpt.get("num_foods", 221)
                num_ings = len(ckpt.get("ing2idx", {})) or ckpt.get("num_ingredients", 324)
                nut_cols = ckpt.get("nut_cols", ["calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g"])
                num_nut = len(nut_cols)
                m = SouthIndianIngredientNutritionModel(num_foods=num_foods, num_ingredients=num_ings, num_nutrition=num_nut)
                m.load_state_dict(ckpt["model_state_dict"])
                m.eval()
                self.model = m
                self.checkpoint = ckpt
                if "ing2idx" in ckpt:
                    self.idx2ing = {v: k for k, v in ckpt["ing2idx"].items()}
                if "food2idx" in ckpt:
                    self.idx2food = {v: k for k, v in ckpt["food2idx"].items()}
                if "nut_mean" in ckpt:
                    self.nut_mean = np.array(ckpt["nut_mean"])
                if "nut_std" in ckpt:
                    self.nut_std = np.array(ckpt["nut_std"])
            except Exception as e:
                print("Notice: Could not load .pt model weights:", e)

    def get_info(self, food_name):
        """Retrieve authentic ingredients and nutrition from knowledge database."""
        if not food_name:
            return None
        if food_name in self.database:
            return self.database[food_name]
        norm = self._normalize(food_name)
        if norm in self.normalized_index:
            return self.normalized_index[norm]
        for key, val in self.database.items():
            if norm in self._normalize(key) or self._normalize(key) in norm:
                return val
        return None

    def _estimate_ingredient_weight(self, ing_name, recipe_ingredients):
        """Estimate gram weight for a predicted ingredient from standard recipe."""
        clean_name = self._normalize(ing_name)
        # 1. Exact or partial match with recipe ingredients
        for r_key, r_wt in recipe_ingredients.items():
            clean_r = self._normalize(r_key)
            if clean_name in clean_r or clean_r in clean_name:
                return int(r_wt)
        
        # 2. Heuristic estimate based on ingredient category
        lower = ing_name.lower()
        if any(w in lower for w in ["rice", "batter", "flour", "noodles", "ragi", "wheat"]):
            return 80
        elif any(w in lower for w in ["chicken", "mutton", "fish", "prawn", "crab", "meat"]):
            return 90
        elif any(w in lower for w in ["dal", "lentil", "chana", "paneer", "egg", "mushroom"]):
            return 40
        elif any(w in lower for w in ["ghee", "oil", "butter"]):
            return 12
        elif any(w in lower for w in ["cashew", "coconut", "onion", "tomato", "potato", "carrot", "beetroot"]):
            return 15
        elif any(w in lower for w in ["sugar", "jaggery"]):
            return 10
        else:  # Spices, tempering seeds, aromatics, leaves
            return 3

    def predict_with_nn(self, food_name, threshold=0.35):
        """Run forward pass using PyTorch Model 2 (ingredient_identification_v1_best.pt)."""
        info = self.get_info(food_name)
        if not info:
            return None

        canonical_name = [k for k, v in self.database.items() if v["food_id"] == info["food_id"]][0]
        recipe = self.get_home_hotel_recipe(canonical_name)
        home_recipe = recipe.get("home", {}) if recipe else {}
        hotel_recipe = recipe.get("hotel", {}) if recipe else {}
        recipe_ings = home_recipe.get("ingredients", {})
        est_total_weight = home_recipe.get("total_weight_g", 150)
        hotel_total_weight = hotel_recipe.get("total_weight_g", 200)

        if self.model is None or canonical_name not in self.checkpoint["food2idx"]:
            # Fallback to database if model not yet trained
            return {
                "source": "Knowledge Database",
                "food_name": canonical_name,
                "regional_name": info["regional_name"],
                "category": info["category"],
                "portion_size": info["portion_size"],
                "estimated_weight_g": est_total_weight,
                "weight_range_desc": f"~{est_total_weight}g (Home) to ~{hotel_total_weight}g (Hotel)",
                "predicted_ingredients": [(ing.replace('_', ' ').title(), 1.0, self._estimate_ingredient_weight(ing, recipe_ings)) for ing in info["main_ingredients"]],
                "nutrition": info["nutrition"],
                "health_benefits": info["health_benefits"],
                "cooking_method": info["cooking_method"]
            }

        food_idx = self.checkpoint["food2idx"][canonical_name]
        food_tensor = torch.tensor([food_idx], dtype=torch.long)

        with torch.no_grad():
            pred_logits, pred_nut_norm = self.model(food_tensor)
            probs = torch.sigmoid(pred_logits)[0].numpy()
            pred_nut = (pred_nut_norm[0].numpy() * self.nut_std) + self.nut_mean
            # Enforce physical reality: nutrition cannot be negative
            pred_nut = np.maximum(0.0, pred_nut)

        # Extract predicted ingredients above threshold with weights
        ing_tuples = []
        for i, p in enumerate(probs):
            if p >= threshold:
                ing_name = self.idx2ing[i].replace('_', ' ').title()
                est_wt = self._estimate_ingredient_weight(ing_name, recipe_ings)
                ing_tuples.append((ing_name, float(p), est_wt))

        ing_tuples.sort(key=lambda x: -x[1])

        # If too few predicted, fallback to ground truth ingredients
        if len(ing_tuples) < 2:
            ing_tuples = [
                (ing.replace('_', ' ').title(), 1.0, self._estimate_ingredient_weight(ing, recipe_ings))
                for ing in info["main_ingredients"]
            ]

        return {
            "source": "Neural Network (ingredient_identification_v1_best.pt)",
            "food_name": canonical_name,
            "regional_name": info["regional_name"],
            "category": info["category"],
            "portion_size": info["portion_size"],
            "estimated_weight_g": est_total_weight,
            "weight_range_desc": f"~{est_total_weight}g (Home) to ~{hotel_total_weight}g (Hotel)",
            "predicted_ingredients": ing_tuples,
            "nutrition": {
                "calories_kcal": round(float(pred_nut[0]), 1),
                "protein_g": round(float(pred_nut[1]), 1),
                "carbs_g": round(float(pred_nut[2]), 1),
                "fat_g": round(float(pred_nut[3]), 1),
                "fiber_g": round(float(pred_nut[4]), 1)
            },
            "ground_truth_nutrition": info["nutrition"],
            "health_benefits": info["health_benefits"],
            "cooking_method": info["cooking_method"]
        }

    def predict(self, food_name, threshold=0.35):
        """Standard convenience wrapper for predict_with_nn."""
        return self.predict_with_nn(food_name, threshold=threshold)

    def get_home_hotel_recipe(self, food_name):
        """Retrieve Home vs Hotel recipe details, portion sizes, and ingredient breakdown."""
        if not food_name:
            return None
        if food_name in self.home_hotel_db:
            return self.home_hotel_db[food_name]
        norm = self._normalize(food_name)
        if norm in self.home_hotel_index:
            return self.home_hotel_index[norm]
        for key, val in self.home_hotel_db.items():
            if norm in self._normalize(key) or self._normalize(key) in norm:
                return val
        return None

    def _match_ingredient_macro(self, ing_name):
        """Match ingredient name to nutritional profile density."""
        name_lower = ing_name.lower().replace(" ", "_")
        for key in ["ghee", "butter", "oil", "chicken", "mutton", "prawn", "crab", 
                    "egg", "paneer", "mushroom", "coconut", "sugar", "jaggery", 
                    "milk", "cashew", "urad", "toor", "moong", "chana", "dal", "rice"]:
            if key in name_lower:
                return INGREDIENT_MACRO_MAP[key]
        for v in ["onion", "tomato", "potato", "carrot", "beetroot", "veg", "chili", "spinach", "leaves"]:
            if v in name_lower:
                return INGREDIENT_MACRO_MAP["veg"]
        return INGREDIENT_MACRO_MAP["default"]

    def calculate_hotel_proportional(self, food_name, target_weight_g):
        """
        Split hotel ingredient weights and recalculate nutrition
        proportionally based on user's typed total weight.
        """
        recipe = self.get_home_hotel_recipe(food_name)
        if not recipe or "hotel" not in recipe:
            return None
        
        hotel = recipe["hotel"]
        base_wt = float(hotel["total_weight_g"])
        target_wt = max(10.0, float(target_weight_g))
        scale = target_wt / base_wt if base_wt > 0 else 1.0
        
        # Split ingredients proportionally
        split_ingredients = {}
        for ing, orig_g in hotel["ingredients"].items():
            split_ingredients[ing] = round(orig_g * scale, 1)
            
        h_nut = hotel["nutrition"]
        scaled_nut = {
            "calories_kcal": round(h_nut["calories_kcal"] * scale, 1),
            "protein_g": round(h_nut["protein_g"] * scale, 1),
            "carbs_g": round(h_nut["carbs_g"] * scale, 1),
            "fat_g": round(h_nut["fat_g"] * scale, 1),
            "fiber_g": round(h_nut["fiber_g"] * scale, 1)
        }
        
        return {
            "food_name": food_name,
            "prep_mode": "hotel_proportional",
            "base_weight_g": base_wt,
            "total_weight_g": target_wt,
            "scale_factor": round(scale, 2),
            "split_ingredients": split_ingredients,
            "nutrition": scaled_nut,
            "note": hotel.get("note", "")
        }

    def calculate_custom_nutrition(self, food_name, custom_ingredients_dict, additional_ingredients_dict=None):
        """
        Dynamically calculate overall weight and nutritional profile
        when a user customizes ingredient sizes or adds new ingredients in Home preparation mode.
        """
        recipe = self.get_home_hotel_recipe(food_name)
        if not recipe or "home" not in recipe:
            return None

        base_home = recipe["home"]
        base_nut = base_home["nutrition"]
        base_ings = base_home["ingredients"]
        base_weight = base_home["total_weight_g"]

        delta_p = 0.0
        delta_c = 0.0
        delta_f = 0.0
        delta_fib = 0.0
        weight_delta = 0.0

        combined_ingredients = dict(custom_ingredients_dict)

        for ing, new_val in custom_ingredients_dict.items():
            orig_val = base_ings.get(ing, 0)
            diff_g = float(new_val) - float(orig_val)
            weight_delta += diff_g
            macro = self._match_ingredient_macro(ing)
            delta_p += (diff_g * macro["p"]) / 100.0
            delta_c += (diff_g * macro["c"]) / 100.0
            delta_f += (diff_g * macro["f"]) / 100.0
            delta_fib += (diff_g * macro["fib"]) / 100.0

        # Include additional user-added ingredients
        if additional_ingredients_dict:
            for extra_ing, extra_g in additional_ingredients_dict.items():
                eg = float(extra_g)
                if eg > 0:
                    combined_ingredients[extra_ing] = eg
                    weight_delta += eg
                    macro = self._match_ingredient_macro(extra_ing)
                    delta_p += (eg * macro["p"]) / 100.0
                    delta_c += (eg * macro["c"]) / 100.0
                    delta_f += (eg * macro["f"]) / 100.0
                    delta_fib += (eg * macro["fib"]) / 100.0

        new_protein = max(0.0, round(base_nut["protein_g"] + delta_p, 1))
        new_carbs = max(0.0, round(base_nut["carbs_g"] + delta_c, 1))
        new_fat = max(0.0, round(base_nut["fat_g"] + delta_f, 1))
        new_fiber = max(0.0, round(base_nut["fiber_g"] + delta_fib, 1))
        new_calories = max(0.0, round(4.0 * new_carbs + 4.0 * new_protein + 9.0 * new_fat, 1))
        new_weight = max(10.0, round(base_weight + weight_delta, 1))

        return {
            "food_name": food_name,
            "prep_mode": "home_custom",
            "portion_desc": f"Custom Home Portion (~{int(new_weight)}g)",
            "total_weight_g": new_weight,
            "ingredients": combined_ingredients,
            "nutrition": {
                "calories_kcal": new_calories,
                "protein_g": new_protein,
                "carbs_g": new_carbs,
                "fat_g": new_fat,
                "fiber_g": new_fiber
            }
        }

_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = IngredientNutritionPredictor()
    return _predictor_instance

def get_ingredient_nutrition(food_name):
    return get_predictor().get_info(food_name)

def predict_with_model(food_name):
    return get_predictor().predict_with_nn(food_name)

def get_home_hotel_recipe(food_name):
    return get_predictor().get_home_hotel_recipe(food_name)

def calculate_custom_nutrition(food_name, custom_ingredients_dict, additional_ingredients_dict=None):
    return get_predictor().calculate_custom_nutrition(food_name, custom_ingredients_dict, additional_ingredients_dict)

def calculate_hotel_proportional(food_name, target_weight_g):
    return get_predictor().calculate_hotel_proportional(food_name, target_weight_g)

if __name__ == "__main__":
    p = get_predictor()
    for test in ["Idly", "Dosa", "Ven Pongal", "Chicken briyani"]:
        res = p.predict_with_nn(test)
        print(f"=== {test} ({res['source']}) ===")
        print("Ingredients:", res["predicted_ingredients"][:5])
        print("Nutrition:", res["nutrition"])
