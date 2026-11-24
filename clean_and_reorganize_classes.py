"""
Clean and deduplicate Version 2 classes for computer vision object detection.
Removes:
1. 'Chicken tandoori' (exact inverted duplicate of 'Tandoori chicken')
2. 'Chicken gravy' (identical visual appearance to 'Chicken curry')
3. 'Sundal varieties' (duplicate of 'Sundal')
4. 'Oats' (duplicate of 'Oatmeal')
5. 'Whole milk' (duplicate of 'Milk')
6. 'Whey' (redundant with 'Whey protein')
7. 'Biryani' (generic ambiguous parent of 'Chicken biryani' / 'Mutton biryani')
"""

import os
import json
import yaml
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOODS_MASTER_PATH = os.path.join(BASE_DIR, "Incredient_Identification", "DataSet", "Version2 Used Data", "foods_master.json")
RAW_V2_DIR = os.path.join(BASE_DIR, "Food_Detection", "DataSet", "RawData", "Version2")
USED_V2_DIR = os.path.join(BASE_DIR, "Food_Detection", "DataSet", "Version2 Used Data")

with open(FOODS_MASTER_PATH, "r", encoding="utf-8") as f:
    foods = json.load(f)

foods_dict = {f["food_name"]: f for f in foods}

# Specific removals for computer vision ambiguity / duplicates:
DUPLICATES_TO_REMOVE = {
    "Chicken tandoori": "Duplicate of 'Tandoori chicken'",
    "Chicken gravy": "Visually indistinguishable from 'Chicken curry'",
    "Sundal varieties": "Duplicate of 'Sundal'",
    "Oats": "Duplicate of 'Oatmeal'",
    "Whole milk": "Duplicate of 'Milk'",
    "Whey": "Redundant duplicate of 'Whey protein'",
    "Biryani": "Ambiguous generic class conflicting with 'Chicken biryani' and 'Mutton biryani'"
}

groups_order = [
    # ----------------------------------------------------
    # GROUP 1: FULL VEGETARIAN (Traditional & Modern Veg)
    # ----------------------------------------------------
    ("Full Vegetarian Foods", [
        # South Indian Veg Staples
        "Idli", "Dosa", "Masala dosa", "Ragi dosa", "Ragi koozh", "Pongal", "Upma",
        "Appam", "Puttu", "Idiyappam", "Adai", "Chapati", "Parotta", "Rice",
        "Lemon rice", "Curd rice", "Tomato rice", "Sambar rice", "Rasam rice", "Pulao", "Kichadi",
        # Vegetarian Curries, Dals & Sides
        "Sambar", "Rasam", "Dal", "Kootu", "Poriyal", "Avial", "Kurma",
        "Paneer butter masala", "Palak paneer", "Chana masala", "Rajma",
        "Vegetable curry", "Mushroom curry", "Vegetable pulao", "Vegetable fried rice",
        # Veg Salads
        "Cucumber salad", "Carrot salad", "Beetroot salad", "Onion tomato salad",
        "Mixed vegetable salad", "Sprouts salad", "Chickpea salad", "Protein salad",
        # Sprouts & Legumes
        "Green gram sprouts", "Black gram sprouts", "Chickpea sprouts", "Black chana",
        "Horse gram", "Mixed sprouts", "Boiled chickpeas", "Kidney beans", "Lentils",
        "Soybeans", "Green peas", "Sundal"
    ]),

    # ----------------------------------------------------
    # GROUP 2: FULL NON-VEGETARIAN (Poultry, Meat, Seafood, Eggs)
    # ----------------------------------------------------
    ("Full Non-Vegetarian Foods", [
        # Chicken
        "Chicken curry", "Chicken 65", "Chicken fry", "Chicken biryani",
        "Tandoori chicken", "Chicken tikka", "Grilled chicken", "Chicken kebab",
        "Chicken burger", "Chicken pizza",
        # Mutton
        "Mutton curry", "Mutton biryani",
        # Fish & Seafood
        "Fish curry", "Fish fry", "Grilled fish", "Prawn masala", "Prawn fry", "Crab curry",
        # Eggs
        "Egg boiled", "Egg omelette", "Egg bhurji", "Egg dosa"
    ]),

    # ----------------------------------------------------
    # GROUP 3: FAST FOODS & INTERNATIONAL FOODS
    # ----------------------------------------------------
    ("Fast Foods & International", [
        "Burger", "Veg burger", "Cheese burger",
        "Pizza", "Veg pizza", "Cheese pizza",
        "Sandwich", "Grilled sandwich", "Wraps", "Shawarma", "Tacos",
        "Pasta", "Macaroni", "Noodles", "Fried rice", "French fries", "Nachos", "Hot dogs"
    ]),

    # ----------------------------------------------------
    # GROUP 4: FRESH VEGETABLES
    # ----------------------------------------------------
    ("Fresh Vegetables", [
        "Carrot", "Beetroot", "Cucumber", "Tomato", "Onion", "Potato", "Sweet potato",
        "Broccoli", "Cauliflower", "Spinach", "Amaranth leaves", "Cabbage", "Beans",
        "Capsicum", "Pumpkin", "Bottle gourd", "Bitter gourd", "Drumstick", "Brinjal", "Mushroom"
    ]),

    # ----------------------------------------------------
    # GROUP 5: FRESH FRUITS & FRUIT SALADS
    # ----------------------------------------------------
    ("Fresh Fruits & Salads", [
        "Apple", "Banana", "Mango", "Orange", "Papaya", "Pineapple", "Watermelon",
        "Muskmelon", "Guava", "Grapes", "Pomegranate", "Kiwi", "Dragon fruit",
        "Strawberry", "Blueberry", "Avocado", "Jackfruit", "Dates", "Fig", "Sapota",
        "Custard apple", "Coconut",
        # Fruit Salads
        "Mixed fruit salad", "Apple fruit salad", "Mango fruit salad",
        "Watermelon fruit salad", "Seasonal fruit salad"
    ]),

    # ----------------------------------------------------
    # GROUP 6: NUTS & SEEDS
    # ----------------------------------------------------
    ("Nuts & Seeds", [
        # Nuts
        "Almonds", "Cashews", "Walnuts", "Pistachios", "Peanuts",
        "Hazelnuts", "Brazil nuts", "Macadamia nuts", "Pecans",
        # Seeds
        "Chia seeds", "Flax seeds", "Pumpkin seeds", "Sunflower seeds",
        "Sesame seeds", "Watermelon seeds", "Melon seeds", "Hemp seeds",
        "Basil seeds", "Poppy seeds"
    ]),

    # ----------------------------------------------------
    # GROUP 7: DAIRY PRODUCTS
    # ----------------------------------------------------
    ("Dairy Products", [
        "Toned milk", "Curd", "Greek yogurt",
        "Paneer", "Cheese", "Butter", "Ghee"
    ]),

    # ----------------------------------------------------
    # GROUP 8: BREAKFAST & HEALTHY SNACKS
    # ----------------------------------------------------
    ("Breakfast & Snacks", [
        "Overnight oats", "Oatmeal", "Cornflakes", "Muesli", "Granola",
        "Peanut butter", "Boiled corn", "Popcorn", "Samosa", "Vada", "Bajji",
        "Bonda", "Murukku", "Biscuits"
    ]),

    # ----------------------------------------------------
    # GROUP 9: DESSERTS & SWEETS
    # ----------------------------------------------------
    ("Desserts & Sweets", [
        "Payasam", "Halwa", "Laddu", "Gulab jamun", "Rasgulla", "Kheer",
        "Ice cream", "Cake", "Brownies", "Dark chocolate", "Dates-based sweets"
    ]),

    # ----------------------------------------------------
    # GROUP 10: ALL DRINKS & SMOOTHIES & BEVERAGES
    # ----------------------------------------------------
    ("All Drinks & Beverages", [
        "Water", "Tender coconut water", "Milk", "Skimmed milk", "Buttermilk", "Lassi",
        "Filter coffee", "Tea", "Green tea", "Lemon juice", "Orange juice", "Apple juice",
        "Sugarcane juice", "Fresh fruit juices", "Soft drinks", "Energy drinks", "Recovery shakes"
    ]),

    # ----------------------------------------------------
    # GROUP 11: GYM & SPORTS NUTRITION SUPPLEMENTS
    # ----------------------------------------------------
    ("Gym & Sports Nutrition", [
        "Whey protein", "Whey isolate", "Casein protein", "Plant protein", "Soy protein",
        "Creatine", "Protein bars", "Mass gainers", "Electrolyte drinks", "Pre-workout drinks"
    ])
]

ordered_classes = []
seen = set()

for group_name, item_list in groups_order:
    for item in item_list:
        if item in DUPLICATES_TO_REMOVE:
            continue
        # Check lookup
        f_meta = foods_dict.get(item, {})
        if item not in seen:
            seen.add(item)
            ordered_classes.append({
                "class_name": item,
                "group_category": group_name,
                "original_category": f_meta.get("category", ""),
                "diet_type": f_meta.get("diet_type", ""),
                "fitness_role": f_meta.get("food_type", ""),
                "preparation_method": f_meta.get("preparation_method", ""),
                "food_state": f_meta.get("food_state", "")
            })

for idx, item in enumerate(ordered_classes):
    item["class_id"] = idx

print(f"Total Clean, Deduplicated Classes: {len(ordered_classes)}")
print(f"Removed {len(DUPLICATES_TO_REMOVE)} duplicates:")
for k, v in DUPLICATES_TO_REMOVE.items():
    print(f"  - '{k}': {v}")

# 1. Update classes_v2.txt
classes_txt_path = os.path.join(RAW_V2_DIR, "classes_v2.txt")
with open(classes_txt_path, "w", encoding="utf-8") as f:
    for item in ordered_classes:
        f.write(f"{item['class_name']}\n")

# 2. Update classes_v2.csv
classes_df = pd.DataFrame(ordered_classes)
classes_csv_path = os.path.join(RAW_V2_DIR, "classes_v2.csv")
classes_df.to_csv(classes_csv_path, index=False, encoding="utf-8")

# 3. Update class_mapping_v2.json
classes_json_path = os.path.join(RAW_V2_DIR, "class_mapping_v2.json")
with open(classes_json_path, "w", encoding="utf-8") as f:
    json.dump(ordered_classes, f, indent=2, ensure_ascii=False)

# 4. Update YOLOv8 dataset.yaml
classes_dict = {item["class_id"]: item["class_name"] for item in ordered_classes}
yolo_yaml_data = {
    "path": "Food_Detection/DataSet/Version2 Used Data",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(ordered_classes),
    "names": classes_dict
}

yaml_path = os.path.join(USED_V2_DIR, "dataset.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(yolo_yaml_data, f, sort_keys=False, allow_unicode=True)

print(f"Updated YOLOv8 dataset.yaml with nc={len(ordered_classes)} at: {yaml_path}")

