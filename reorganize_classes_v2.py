"""
Script to reorganize Food Detection Version 2 classes into a wise, categorical group order:
1. Full Vegetarian Foods (South Indian Veg, Curries, Dals, Veg Salads, Sprouts/Legumes)
2. Full Non-Vegetarian Foods (Chicken, Mutton, Fish, Seafood, Eggs)
3. Fast Foods & International (Burgers, Pizzas, Sandwiches, Wraps, Pasta, Noodles, Fries)
4. Fresh Vegetables
5. Fresh Fruits & Fruit Salads
6. Nuts & Seeds
7. Dairy Products
8. Breakfast & Healthy Snacks
9. Desserts & Sweets
10. All Drinks, Smoothies & Beverages
11. Gym & Sports Nutrition Supplements
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

# Lookup dictionary by name
foods_dict = {f["food_name"]: f for f in foods}

# Define the exact categorized groups
groups_order = [
    # ----------------------------------------------------
    # GROUP 1: FULL VEGETARIAN (Traditional & Modern Veg)
    # ----------------------------------------------------
    ("Full Vegetarian Foods", [
        # South Indian Veg Staples
        "Idli", "Dosa", "Masala dosa", "Ragi dosa", "Ragi koozh", "Pongal", "Upma",
        "Appam", "Puttu", "Idiyappam", "Adai", "Chapati", "Parotta", "Rice",
        "Lemon rice", "Curd rice", "Tomato rice", "Sambar rice", "Rasam rice", "Biryani", "Pulao", "Kichadi",
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
        "Soybeans", "Green peas", "Sundal varieties"
    ]),

    # ----------------------------------------------------
    # GROUP 2: FULL NON-VEGETARIAN (Poultry, Meat, Seafood, Eggs)
    # ----------------------------------------------------
    ("Full Non-Vegetarian Foods", [
        # Chicken
        "Chicken curry", "Chicken gravy", "Chicken 65", "Chicken fry", "Chicken biryani",
        "Chicken tandoori", "Tandoori chicken", "Chicken tikka", "Grilled chicken", "Chicken kebab",
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
        "Whole milk", "Toned milk", "Curd", "Greek yogurt",
        "Paneer", "Cheese", "Butter", "Ghee", "Whey"
    ]),

    # ----------------------------------------------------
    # GROUP 8: BREAKFAST & HEALTHY SNACKS
    # ----------------------------------------------------
    ("Breakfast & Snacks", [
        "Oats", "Overnight oats", "Oatmeal", "Cornflakes", "Muesli", "Granola",
        "Peanut butter", "Boiled corn", "Popcorn", "Samosa", "Vada", "Bajji",
        "Bonda", "Sundal", "Murukku", "Biscuits"
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

# Flatten and assign sequential class IDs
ordered_classes = []
seen = set()

for group_name, item_list in groups_order:
    for item in item_list:
        if item in foods_dict:
            if item not in seen:
                seen.add(item)
                f_meta = foods_dict[item]
                ordered_classes.append({
                    "class_name": item,
                    "group_category": group_name,
                    "original_category": f_meta.get("category", ""),
                    "diet_type": f_meta.get("diet_type", ""),
                    "fitness_role": f_meta.get("food_type", ""),
                    "preparation_method": f_meta.get("preparation_method", ""),
                    "food_state": f_meta.get("food_state", "")
                })

# Check if any foods in foods_dict were missed
missing = [f_name for f_name in foods_dict if f_name not in seen]
if missing:
    print(f"Warning: {len(missing)} foods not in groups list: {missing}")
    for item in missing:
        f_meta = foods_dict[item]
        ordered_classes.append({
            "class_name": item,
            "group_category": "Other",
            "original_category": f_meta.get("category", ""),
            "diet_type": f_meta.get("diet_type", ""),
            "fitness_role": f_meta.get("food_type", ""),
            "preparation_method": f_meta.get("preparation_method", ""),
            "food_state": f_meta.get("food_state", "")
        })

print(f"Total Ordered Classes: {len(ordered_classes)}")

# Assign class_id 0 to len-1
for idx, item in enumerate(ordered_classes):
    item["class_id"] = idx

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

print(f"Successfully generated dataset.yaml with {len(ordered_classes)} classes at {yaml_path}")
