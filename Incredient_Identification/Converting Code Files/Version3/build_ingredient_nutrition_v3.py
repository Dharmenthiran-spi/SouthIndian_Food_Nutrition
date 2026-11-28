"""
Ingredient Identification Version 3 - Master Dataset Builder
Builds the 231-class ingredient and nutrition dataset aligned 1-to-1 with
Food_Detection/DataSet/Version3 Used Data/dataset.yaml (classes 0..230).
"""

import os
import json
import yaml
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ING_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(ING_ROOT, ".."))

V2_DIR = os.path.join(ING_ROOT, "DataSet", "Version2 Used Data")
V3_DIR = os.path.join(ING_ROOT, "DataSet", "Version3 Used Data")
FD_V3_YAML = os.path.join(REPO_ROOT, "Food_Detection", "DataSet", "Version3 Used Data", "dataset.yaml")

NUT_COLS = [
    "calories_kcal", "protein_g", "carbohydrates_g", "fat_g", "fiber_g",
    "sugar_g", "sodium_mg", "potassium_mg", "calcium_mg", "iron_mg", "vitamin_c_mg"
]

# Full profiles for classes 221..230 (per ~100g standard serving)
NEW_V3_FOODS = {
    221: {
        "food_name": "Poori",
        "category": "Indian Breads",
        "group_category": "Full Vegetarian Foods",
        "diet_type": "Vegetarian",
        "fitness_goal": "High Energy / Calorie Surplus",
        "ingredients": ["whole_wheat_flour", "oil", "salt", "water"],
        "calories_kcal": 296.0, "protein_g": 6.8, "carbohydrates_g": 41.0, "fat_g": 11.5,
        "fiber_g": 3.4, "sugar_g": 0.8, "sodium_mg": 210.0, "potassium_mg": 135.0,
        "calcium_mg": 18.0, "iron_mg": 1.9, "vitamin_c_mg": 0.0,
    },
    222: {
        "food_name": "Chutney",
        "category": "South Indian",
        "group_category": "Full Vegetarian Foods",
        "diet_type": "Vegetarian",
        "fitness_goal": "Balanced Diet / Daily Fitness",
        "ingredients": ["fresh_grated_coconut", "chana_dal", "green_chilli", "ginger", "curry_leaves", "mustard_seeds", "oil", "salt"],
        "calories_kcal": 185.0, "protein_g": 4.2, "carbohydrates_g": 11.0, "fat_g": 14.0,
        "fiber_g": 4.5, "sugar_g": 2.4, "sodium_mg": 320.0, "potassium_mg": 210.0,
        "calcium_mg": 28.0, "iron_mg": 1.4, "vitamin_c_mg": 4.5,
    },
    223: {
        "food_name": "Momos",
        "category": "Snacks & Street Food",
        "group_category": "Full Vegetarian Foods",
        "diet_type": "Vegetarian",
        "fitness_goal": "Balanced Diet / Daily Fitness",
        "ingredients": ["maida_refined_flour", "cabbage", "carrot", "onion", "garlic", "ginger", "soy_sauce", "black_pepper", "oil", "salt"],
        "calories_kcal": 175.0, "protein_g": 5.2, "carbohydrates_g": 27.5, "fat_g": 4.8,
        "fiber_g": 2.2, "sugar_g": 2.1, "sodium_mg": 410.0, "potassium_mg": 165.0,
        "calcium_mg": 24.0, "iron_mg": 1.3, "vitamin_c_mg": 8.0,
    },
    224: {
        "food_name": "Pani puri",
        "category": "Snacks & Street Food",
        "group_category": "Full Vegetarian Foods",
        "diet_type": "Vegetarian",
        "fitness_goal": "Light Snack / Portion Control",
        "ingredients": ["roasted_semolina", "maida_refined_flour", "potato", "boiled_chickpeas", "tamarind", "mint", "coriander", "green_chilli", "cumin", "salt"],
        "calories_kcal": 182.0, "protein_g": 4.1, "carbohydrates_g": 31.0, "fat_g": 4.5,
        "fiber_g": 3.0, "sugar_g": 3.8, "sodium_mg": 490.0, "potassium_mg": 220.0,
        "calcium_mg": 26.0, "iron_mg": 1.7, "vitamin_c_mg": 7.5,
    },
    225: {
        "food_name": "Jalebi",
        "category": "Sweets & Desserts",
        "group_category": "Full Vegetarian Foods",
        "diet_type": "Vegetarian",
        "fitness_goal": "High Energy / Occasional Treat",
        "ingredients": ["maida_refined_flour", "curd", "sugar", "ghee", "cardamom", "saffron", "water"],
        "calories_kcal": 378.0, "protein_g": 2.6, "carbohydrates_g": 62.0, "fat_g": 13.5,
        "fiber_g": 0.6, "sugar_g": 42.0, "sodium_mg": 45.0, "potassium_mg": 55.0,
        "calcium_mg": 22.0, "iron_mg": 0.9, "vitamin_c_mg": 0.0,
    },
    226: {
        "food_name": "Vada pav",
        "category": "Snacks & Street Food",
        "group_category": "Full Vegetarian Foods",
        "diet_type": "Vegetarian",
        "fitness_goal": "High Energy / Calorie Surplus",
        "ingredients": ["bun", "potato", "gram_flour_besan", "garlic", "green_chilli", "mustard_seeds", "curry_leaves", "turmeric", "oil", "salt"],
        "calories_kcal": 268.0, "protein_g": 6.2, "carbohydrates_g": 38.5, "fat_g": 9.8,
        "fiber_g": 2.8, "sugar_g": 2.6, "sodium_mg": 460.0, "potassium_mg": 240.0,
        "calcium_mg": 34.0, "iron_mg": 1.8, "vitamin_c_mg": 5.0,
    },
    227: {
        "food_name": "Pav",
        "category": "Indian Breads",
        "group_category": "Full Vegetarian Foods",
        "diet_type": "Vegetarian",
        "fitness_goal": "Balanced Diet / Daily Fitness",
        "ingredients": ["maida_refined_flour", "yeast", "milk", "butter", "sugar", "salt", "water"],
        "calories_kcal": 265.0, "protein_g": 7.6, "carbohydrates_g": 49.0, "fat_g": 4.2,
        "fiber_g": 2.1, "sugar_g": 4.5, "sodium_mg": 390.0, "potassium_mg": 115.0,
        "calcium_mg": 42.0, "iron_mg": 2.0, "vitamin_c_mg": 0.0,
    },
    228: {
        "food_name": "Naan",
        "category": "Indian Breads",
        "group_category": "Full Vegetarian Foods",
        "diet_type": "Vegetarian",
        "fitness_goal": "High Energy / Calorie Surplus",
        "ingredients": ["maida_refined_flour", "curd", "butter", "yeast", "garlic", "milk", "sugar", "salt"],
        "calories_kcal": 290.0, "protein_g": 8.4, "carbohydrates_g": 48.0, "fat_g": 6.8,
        "fiber_g": 2.2, "sugar_g": 3.4, "sodium_mg": 410.0, "potassium_mg": 130.0,
        "calcium_mg": 58.0, "iron_mg": 2.1, "vitamin_c_mg": 0.5,
    },
    229: {
        "food_name": "Cutlet",
        "category": "Snacks & Street Food",
        "group_category": "Full Vegetarian Foods",
        "diet_type": "Vegetarian",
        "fitness_goal": "Balanced Diet / Daily Fitness",
        "ingredients": ["potato", "carrot", "green_peas", "beans", "onion", "bread_crumbs", "garam_masala", "ginger", "coriander", "oil", "salt"],
        "calories_kcal": 232.0, "protein_g": 5.4, "carbohydrates_g": 29.0, "fat_g": 10.6,
        "fiber_g": 3.6, "sugar_g": 2.8, "sodium_mg": 380.0, "potassium_mg": 270.0,
        "calcium_mg": 28.0, "iron_mg": 1.6, "vitamin_c_mg": 9.0,
    },
    230: {
        "food_name": "Poha",
        "category": "Breakfast",
        "group_category": "Full Vegetarian Foods",
        "diet_type": "Vegetarian",
        "fitness_goal": "Balanced Diet / Daily Fitness",
        "ingredients": ["flattened_rice_poha", "onion", "peanuts", "mustard_seeds", "curry_leaves", "turmeric", "green_chilli", "lemon_juice", "coriander", "oil", "salt"],
        "calories_kcal": 158.0, "protein_g": 3.5, "carbohydrates_g": 27.5, "fat_g": 3.9,
        "fiber_g": 2.1, "sugar_g": 1.4, "sodium_mg": 280.0, "potassium_mg": 145.0,
        "calcium_mg": 20.0, "iron_mg": 2.4, "vitamin_c_mg": 6.0,
    },
}


def make_variant_row(food_id, info, recipe_type, cal_mul, prot_mul, carb_mul, fat_mul, fib_mul, sug_mul, sod_mul):
    tokens = info["ingredients"]
    display = ", ".join(t.replace("_", " ").title() for t in tokens)
    return {
        "food_id": food_id,
        "food_name": info["food_name"],
        "category": info["category"],
        "recipe_type": recipe_type,
        "ingredients_tokens": ";".join(tokens),
        "ingredients_display": display,
        "calories_kcal": round(info["calories_kcal"] * cal_mul, 1),
        "protein_g": round(info["protein_g"] * prot_mul, 1),
        "carbohydrates_g": round(info["carbohydrates_g"] * carb_mul, 1),
        "fat_g": round(info["fat_g"] * fat_mul, 1),
        "fiber_g": round(info["fiber_g"] * fib_mul, 1),
        "sugar_g": round(info["sugar_g"] * sug_mul, 1),
        "sodium_mg": round(info["sodium_mg"] * sod_mul, 1),
        "potassium_mg": info["potassium_mg"],
        "calcium_mg": info["calcium_mg"],
        "iron_mg": info["iron_mg"],
        "vitamin_c_mg": info["vitamin_c_mg"],
    }


def build_v3():
    os.makedirs(V3_DIR, exist_ok=True)
    with open(FD_V3_YAML, "r", encoding="utf-8") as f:
        fd_cfg = yaml.safe_load(f)
    fd_names = fd_cfg["names"]
    assert len(fd_names) == 231, f"Expected 231 classes in Food_Detection V3, got {len(fd_names)}"

    # Load V2 base files
    master_v2 = pd.read_csv(os.path.join(V2_DIR, "food_and_nutrition_master_v2.csv"))
    train_v2 = pd.read_csv(os.path.join(V2_DIR, "train.csv"))
    val_v2 = pd.read_csv(os.path.join(V2_DIR, "val.csv"))
    test_v2 = pd.read_csv(os.path.join(V2_DIR, "test.csv"))

    with open(os.path.join(V2_DIR, "dishes_ingredients_map_v2.json"), "r", encoding="utf-8") as f:
        dishes_map = json.load(f)
    with open(os.path.join(V2_DIR, "home_hotel_recipes_v2.json"), "r", encoding="utf-8") as f:
        hh_recipes = json.load(f)

    # Extend master CSV with 221..230
    new_master_rows = []
    new_train_rows, new_val_rows, new_test_rows = [], [], []
    for fid in range(221, 231):
        info = NEW_V3_FOODS[fid]
        assert info["food_name"] == fd_names[fid], f"Name mismatch at {fid}: {info['food_name']} vs {fd_names[fid]}"

        net_carbs = round(max(0.0, info["carbohydrates_g"] - info["fiber_g"]), 1)
        fat = info["fat_g"]
        new_master_rows.append({
            "food_id": fid,
            "food_name": info["food_name"],
            "category": info["category"],
            "group_category": info["group_category"],
            "diet_type": info["diet_type"],
            "fitness_goal": info["fitness_goal"],
            "calories_kcal": info["calories_kcal"],
            "protein_g": info["protein_g"],
            "carbohydrates_g": info["carbohydrates_g"],
            "fat_g": fat,
            "fiber_g": info["fiber_g"],
            "sugar_g": info["sugar_g"],
            "net_carbs_g": net_carbs,
            "saturated_fat_g": round(fat * 0.30, 2),
            "monounsaturated_fat_g": round(fat * 0.45, 2),
            "polyunsaturated_fat_g": round(fat * 0.25, 2),
            "trans_fat_g": 0.0,
            "cholesterol_mg": 0.0,
            "sodium_mg": info["sodium_mg"],
            "potassium_mg": info["potassium_mg"],
            "calcium_mg": info["calcium_mg"],
            "iron_mg": info["iron_mg"],
            "magnesium_mg": 28.0,
            "zinc_mg": 0.7,
            "phosphorus_mg": 65.0,
            "vitamin_a_mcg": 12.0,
            "vitamin_c_mg": info["vitamin_c_mg"],
            "vitamin_d_mcg": 0.0,
            "vitamin_e_mg": 0.6,
            "vitamin_b1_thiamine_mg": 0.12,
            "vitamin_b6_mg": 0.11,
            "vitamin_b9_folate_mcg": 24.0,
            "vitamin_b12_mcg": 0.0,
            "water_moisture_g": 45.0,
            "ingredients": ", ".join(info["ingredients"]),
        })

        # 4 recipe variations matching V2 splits (2 train, 1 val, 1 test)
        new_train_rows.append(make_variant_row(fid, info, "hotel_style", 1.15, 0.98, 1.05, 1.25, 0.95, 1.10, 1.15))
        new_train_rows.append(make_variant_row(fid, info, "home_style", 0.95, 1.02, 0.98, 0.85, 1.05, 0.95, 0.90))
        new_val_rows.append(make_variant_row(fid, info, "healthy_choice", 0.85, 1.12, 0.90, 0.70, 1.20, 0.75, 0.75))
        new_test_rows.append(make_variant_row(fid, info, "standard", 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00))

        # Update JSON knowledge maps
        name = info["food_name"]
        ing_dict = {t: round(100.0 / len(info["ingredients"]), 1) for t in info["ingredients"]}
        nut_std = {c: info[c] for c in NUT_COLS}
        nut_home = {c: new_train_rows[-1][c] for c in NUT_COLS}
        nut_hotel = {c: new_train_rows[-2][c] for c in NUT_COLS}

        dishes_map[name] = {
            "food_id": fid,
            "food_name": name,
            "category": info["category"],
            "group_category": info["group_category"],
            "diet_type": info["diet_type"],
            "fitness_goal": info["fitness_goal"],
            "portion_size": "1 serving (~100g)",
            "nutrition": nut_std,
            "ingredients": info["ingredients"],
        }
        hh_recipes[name] = {
            "food_id": fid,
            "food_name": name,
            "category": info["category"],
            "home": {
                "portion_desc": "1 serving (~100g) (Home Style)",
                "total_weight_g": 100,
                "ingredients": ing_dict,
                "nutrition": nut_home,
            },
            "hotel": {
                "portion_desc": "1 serving (~100g) (Hotel Style)",
                "total_weight_g": 100,
                "ingredients": ing_dict,
                "nutrition": nut_hotel,
            },
        }

    master_v3 = pd.concat([master_v2, pd.DataFrame(new_master_rows)], ignore_index=True)
    train_v3 = pd.concat([train_v2, pd.DataFrame(new_train_rows)], ignore_index=True)
    val_v3 = pd.concat([val_v2, pd.DataFrame(new_val_rows)], ignore_index=True)
    test_v3 = pd.concat([test_v2, pd.DataFrame(new_test_rows)], ignore_index=True)

    # Build complete ingredient vocabulary from all splits
    all_tokens = set()
    for df in (train_v3, val_v3, test_v3):
        for s in df["ingredients_tokens"].dropna():
            for t in str(s).split(";"):
                tc = t.strip().lower()
                if tc:
                    all_tokens.add(tc)

    sorted_tokens = sorted(all_tokens)
    vocab_rows = [
        {"ingredient_id": idx, "ingredient_key": tok, "display_name": tok.replace("_", " ").title()}
        for idx, tok in enumerate(sorted_tokens)
    ]
    vocab_v3 = pd.DataFrame(vocab_rows)

    master_v3.to_csv(os.path.join(V3_DIR, "food_and_nutrition_master_v3.csv"), index=False)
    train_v3.to_csv(os.path.join(V3_DIR, "train.csv"), index=False)
    val_v3.to_csv(os.path.join(V3_DIR, "val.csv"), index=False)
    test_v3.to_csv(os.path.join(V3_DIR, "test.csv"), index=False)
    vocab_v3.to_csv(os.path.join(V3_DIR, "ingredients_vocabulary_v3.csv"), index=False)

    with open(os.path.join(V3_DIR, "dishes_ingredients_map_v3.json"), "w", encoding="utf-8") as f:
        json.dump(dishes_map, f, indent=2)
    with open(os.path.join(V3_DIR, "home_hotel_recipes_v3.json"), "w", encoding="utf-8") as f:
        json.dump(hh_recipes, f, indent=2)

    v3_yaml = {
        "version": 3.0,
        "description": "Ingredient Identification & Comprehensive Nutrition Dataset Version 3",
        "num_foods": len(master_v3),
        "num_ingredients": len(vocab_v3),
        "total_samples": len(train_v3) + len(val_v3) + len(test_v3),
        "train_samples": len(train_v3),
        "val_samples": len(val_v3),
        "test_samples": len(test_v3),
        "nutrition_targets": NUT_COLS,
        "names": {int(r["food_id"]): r["food_name"] for _, r in master_v3.iterrows()},
        "ingredients": {int(r["ingredient_id"]): r["display_name"] for _, r in vocab_v3.iterrows()},
    }
    with open(os.path.join(V3_DIR, "dataset.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(v3_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"Built Version 3 Ingredient Dataset at: {V3_DIR}")
    print(f"  num_foods={len(master_v3)}, num_ingredients={len(vocab_v3)}")
    print(f"  train={len(train_v3)}, val={len(val_v3)}, test={len(test_v3)}")


if __name__ == "__main__":
    build_v3()

