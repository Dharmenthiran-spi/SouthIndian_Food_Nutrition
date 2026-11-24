"""
Builder for Version 2 Master Ingredient & Nutrition Dataset across all 221 Food Detection Classes.
Produces:
1. dishes_ingredients_map_v2.json (Full 221 classes with nutrition, ingredients, fitness roles)
2. home_hotel_recipes_v2.json (Full 221 classes with home vs hotel recipe breakdown)
3. ingredients_vocabulary_v2.csv (Master vocabulary of all ingredients)
4. food_and_nutrition_master_v2.csv (Tabular master sheet)
5. dataset.yaml for Ingredient Identification v2
"""

import os
import csv
import json
import yaml

V2_USED_DATA = "Incredient_Identification/DataSet/Version2 Used Data"
V1_USED_DATA = "Incredient_Identification/DataSet/Version1 Used Data"
FOOD_DET_V2 = "Food_Detection/DataSet/Version2 Used Data"
CLASSES_CSV = "Food_Detection/DataSet/RawData/Version2/classes_v2.csv"

os.makedirs(V2_USED_DATA, exist_ok=True)

# 1. Load classes_v2.csv
classes_info = []
with open(CLASSES_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        classes_info.append(row)

print(f"Loaded {len(classes_info)} classes from classes_v2.csv")

# 2. Load existing V1 knowledge if available
v1_home_hotel = {}
if os.path.exists(os.path.join(V1_USED_DATA, "home_hotel_recipes.json")):
    with open(os.path.join(V1_USED_DATA, "home_hotel_recipes.json"), "r", encoding="utf-8") as f:
        v1_home_hotel = json.load(f)

v1_dishes_map = {}
if os.path.exists(os.path.join(V1_USED_DATA, "dishes_ingredients_map.json")):
    with open(os.path.join(V1_USED_DATA, "dishes_ingredients_map.json"), "r", encoding="utf-8") as f:
        v1_dishes_map = json.load(f)

# 3. Base nutritional lookup per 100g for single/composite foods
BASE_NUTRITION = {
    # South Indian / Rice / Staples
    "Idli": {"cal": 130, "p": 4.0, "c": 28.0, "f": 0.4, "fib": 1.2, "ing": ["parboiled_rice", "urad_dal", "fenugreek_seeds", "salt"]},
    "Dosa": {"cal": 168, "p": 3.9, "c": 29.0, "f": 3.7, "fib": 1.1, "ing": ["raw_rice", "urad_dal", "oil", "salt"]},
    "Masala dosa": {"cal": 220, "p": 4.8, "c": 34.0, "f": 7.2, "fib": 2.4, "ing": ["dosa_batter", "potato", "onion", "mustard_seeds", "green_chilli", "turmeric", "oil"]},
    "Ragi dosa": {"cal": 155, "p": 4.2, "c": 30.0, "f": 2.1, "fib": 3.8, "ing": ["finger_millet_flour", "rice_flour", "curd", "cumin", "green_chilli", "oil"]},
    "Ragi koozh": {"cal": 110, "p": 2.8, "c": 22.0, "f": 1.0, "fib": 3.2, "ing": ["ragi_flour", "broken_raw_rice", "buttermilk", "shallots", "salt"]},
    "Pongal": {"cal": 195, "p": 5.2, "c": 31.0, "f": 5.8, "fib": 1.8, "ing": ["raw_rice", "yellow_moong_dal", "ghee", "black_pepper", "cumin", "ginger", "cashews", "curry_leaves"]},
    "Upma": {"cal": 170, "p": 4.1, "c": 28.0, "f": 4.5, "fib": 2.0, "ing": ["roasted_semolina", "mustard_seeds", "urad_dal", "ginger", "green_chilli", "curry_leaves", "vegetables", "oil"]},
    "Appam": {"cal": 160, "p": 2.8, "c": 32.0, "f": 2.3, "fib": 1.2, "ing": ["raw_rice", "coconut_milk", "sugar", "yeast", "salt"]},
    "Puttu": {"cal": 180, "p": 3.6, "c": 38.0, "f": 1.5, "fib": 2.5, "ing": ["coarse_rice_flour", "fresh_grated_coconut", "water", "salt"]},
    "Idiyappam": {"cal": 140, "p": 2.5, "c": 31.0, "f": 0.5, "fib": 1.0, "ing": ["roasted_rice_flour", "water", "salt", "grated_coconut"]},
    "Adai": {"cal": 210, "p": 8.5, "c": 34.0, "f": 4.5, "fib": 4.5, "ing": ["raw_rice", "toor_dal", "chana_dal", "urad_dal", "dry_red_chilli", "asafoetida", "oil"]},
    "Chapati": {"cal": 260, "p": 8.2, "c": 52.0, "f": 2.0, "fib": 7.0, "ing": ["whole_wheat_flour", "water", "salt", "oil"]},
    "Parotta": {"cal": 325, "p": 6.8, "c": 48.0, "f": 12.0, "fib": 1.8, "ing": ["maida_refined_flour", "oil", "sugar", "salt", "water"]},
    "Rice": {"cal": 130, "p": 2.7, "c": 28.0, "f": 0.3, "fib": 0.4, "ing": ["boiled_white_rice", "water"]},
    "Lemon rice": {"cal": 185, "p": 3.2, "c": 32.0, "f": 5.0, "fib": 1.2, "ing": ["cooked_rice", "fresh_lemon_juice", "peanuts", "mustard_seeds", "turmeric", "curry_leaves", "oil"]},
    "Curd rice": {"cal": 165, "p": 4.5, "c": 24.0, "f": 5.5, "fib": 0.8, "ing": ["cooked_rice", "fresh_curd", "milk", "mustard_seeds", "green_chilli", "ginger", "pomegranate", "oil"]},
    "Tomato rice": {"cal": 175, "p": 3.4, "c": 30.0, "f": 4.8, "fib": 1.6, "ing": ["cooked_rice", "ripe_tomatoes", "onion", "garlic", "sambar_powder", "spices", "oil"]},
    "Sambar rice": {"cal": 160, "p": 4.8, "c": 29.0, "f": 3.2, "fib": 2.8, "ing": ["cooked_rice", "toor_dal", "mixed_vegetables", "tamarind", "sambar_powder", "ghee"]},
    "Rasam rice": {"cal": 140, "p": 3.2, "c": 27.0, "f": 2.0, "fib": 1.4, "ing": ["cooked_rice", "tomato_rasam", "black_pepper", "cumin", "garlic", "ghee"]},
    "Pulao": {"cal": 190, "p": 4.2, "c": 33.0, "f": 5.0, "fib": 2.2, "ing": ["basmati_rice", "mixed_vegetables", "onion", "whole_spices", "ghee", "mint"]},
    "Kichadi": {"cal": 150, "p": 5.0, "c": 26.0, "f": 2.8, "fib": 2.0, "ing": ["rice", "moong_dal", "cumin", "black_pepper", "ginger", "ghee"]},
    
    # Vegetarian Curries
    "Sambar": {"cal": 95, "p": 3.8, "c": 14.5, "f": 2.5, "fib": 3.2, "ing": ["toor_dal", "shallots", "drumstick", "tomato", "tamarind", "sambar_powder", "oil"]},
    "Rasam": {"cal": 45, "p": 1.2, "c": 7.0, "f": 1.2, "fib": 0.8, "ing": ["tomato", "tamarind_water", "crushed_black_pepper", "cumin", "garlic", "coriander", "ghee"]},
    "Dal": {"cal": 120, "p": 6.5, "c": 17.0, "f": 3.0, "fib": 3.5, "ing": ["yellow_lentils", "tomato", "onion", "garlic", "turmeric", "cumin", "ghee"]},
    "Kootu": {"cal": 105, "p": 4.5, "c": 15.0, "f": 3.0, "fib": 3.8, "ing": ["chana_dal", "cabbage", "chow_chow", "coconut_paste", "cumin", "curry_leaves"]},
    "Poriyal": {"cal": 85, "p": 2.4, "c": 12.0, "f": 3.0, "fib": 3.5, "ing": ["beans", "carrot", "grated_coconut", "mustard_seeds", "urad_dal", "curry_leaves", "oil"]},
    "Avial": {"cal": 115, "p": 2.8, "c": 14.0, "f": 5.2, "fib": 3.8, "ing": ["raw_banana", "carrot", "drumstick", "yam", "coconut_curd_paste", "coconut_oil", "curry_leaves"]},
    "Kurma": {"cal": 140, "p": 3.5, "c": 16.0, "f": 7.0, "fib": 3.2, "ing": ["mixed_vegetables", "coconut_cashew_paste", "poppy_seeds", "onion", "garlic", "spices", "oil"]},
    "Paneer butter masala": {"cal": 240, "p": 9.5, "c": 12.0, "f": 17.5, "fib": 1.8, "ing": ["paneer_cottage_cheese", "butter", "cream", "tomato_puree", "cashew_paste", "kasuri_methi", "spices"]},
    "Palak paneer": {"cal": 185, "p": 9.0, "c": 8.0, "f": 13.0, "fib": 2.8, "ing": ["fresh_spinach", "paneer", "garlic", "onion", "green_chilli", "cream", "ghee"]},
    "Chana masala": {"cal": 160, "p": 7.2, "c": 22.0, "f": 4.8, "fib": 5.5, "ing": ["boiled_chickpeas", "onion", "tomato_puree", "ginger_garlic", "chana_masala_spice", "oil"]},
    "Rajma": {"cal": 155, "p": 7.5, "c": 23.0, "f": 3.8, "fib": 6.0, "ing": ["red_kidney_beans", "onion_tomato_gravy", "ginger_garlic", "coriander", "spices", "oil"]},
    "Vegetable curry": {"cal": 110, "p": 3.0, "c": 15.0, "f": 4.5, "fib": 3.0, "ing": ["mixed_vegetables", "onion", "tomato", "curry_powder", "coriander", "oil"]},
    "Mushroom curry": {"cal": 95, "p": 3.8, "c": 9.0, "f": 5.0, "fib": 2.0, "ing": ["button_mushrooms", "onion", "tomato", "garlic", "coconut_milk", "spices", "oil"]},
    "Vegetable pulao": {"cal": 185, "p": 3.8, "c": 32.0, "f": 4.8, "fib": 2.2, "ing": ["basmati_rice", "carrot", "peas", "french_beans", "whole_garam_masala", "ghee"]},
    "Vegetable fried rice": {"cal": 190, "p": 3.5, "c": 34.0, "f": 4.5, "fib": 1.8, "ing": ["cooked_rice", "cabbage", "carrot", "spring_onion", "soy_sauce", "pepper", "oil"]},

    # Non-Vegetarian
    "Chicken curry": {"cal": 175, "p": 16.5, "c": 5.0, "f": 10.0, "fib": 1.2, "ing": ["chicken_pieces", "onion", "tomato", "ginger_garlic", "curry_powder", "coriander", "oil"]},
    "Chicken 65": {"cal": 260, "p": 22.0, "c": 9.0, "f": 15.0, "fib": 0.8, "ing": ["boneless_chicken", "curd", "kashmiri_chilli", "corn_flour", "ginger_garlic", "curry_leaves", "oil"]},
    "Chicken fry": {"cal": 250, "p": 21.0, "c": 6.0, "f": 16.0, "fib": 0.5, "ing": ["chicken", "shallots", "black_pepper", "garlic", "curry_leaves", "oil"]},
    "Chicken biryani": {"cal": 215, "p": 14.5, "c": 24.0, "f": 6.8, "fib": 1.2, "ing": ["basmati_rice", "chicken", "yogurt", "fried_onions", "biryani_masala", "mint", "ghee"]},
    "Tandoori chicken": {"cal": 195, "p": 24.0, "c": 3.0, "f": 9.5, "fib": 0.5, "ing": ["chicken_leg_breast", "hung_curd", "lemon_juice", "tandoori_spices", "mustard_oil"]},
    "Chicken tikka": {"cal": 185, "p": 23.5, "c": 3.5, "f": 8.5, "fib": 0.4, "ing": ["boneless_chicken_cubes", "yogurt", "garlic", "kashmiri_mirch", "lemon", "butter"]},
    "Grilled chicken": {"cal": 165, "p": 28.0, "c": 0.0, "f": 5.5, "fib": 0.0, "ing": ["chicken_breast", "olive_oil", "black_pepper", "herbs", "lemon_juice", "salt"]},
    "Chicken kebab": {"cal": 205, "p": 22.0, "c": 5.0, "f": 11.0, "fib": 0.6, "ing": ["minced_chicken", "onion", "egg", "green_chilli", "garam_masala", "coriander"]},
    "Chicken burger": {"cal": 250, "p": 14.0, "c": 26.0, "f": 10.5, "fib": 1.5, "ing": ["burger_bun", "chicken_patty", "lettuce", "tomato_slice", "mayonnaise"]},
    "Chicken pizza": {"cal": 270, "p": 13.5, "c": 30.0, "f": 11.0, "fib": 1.8, "ing": ["pizza_dough", "chicken_chunks", "pizza_sauce", "mozzarella_cheese", "capsicum"]},
    "Mutton curry": {"cal": 230, "p": 18.0, "c": 4.5, "f": 15.5, "fib": 1.0, "ing": ["mutton_goat_meat", "onion", "tomato", "ginger_garlic", "spices", "oil"]},
    "Mutton biryani": {"cal": 245, "p": 13.5, "c": 25.0, "f": 10.5, "fib": 1.1, "ing": ["seeraga_samba_rice", "tender_mutton", "ghee", "curd", "spices", "mint", "coriander"]},
    "Fish curry": {"cal": 140, "p": 15.0, "c": 4.0, "f": 7.0, "fib": 0.8, "ing": ["fresh_fish", "tamarind_extract", "shallots", "tomato", "fenugreek", "curry_powder", "oil"]},
    "Fish fry": {"cal": 225, "p": 19.5, "c": 5.5, "f": 14.0, "fib": 0.5, "ing": ["fish_slice", "red_chilli_paste", "lemon_juice", "turmeric", "rice_flour", "oil"]},
    "Grilled fish": {"cal": 150, "p": 22.0, "c": 0.5, "f": 6.5, "fib": 0.0, "ing": ["fish_fillet", "olive_oil", "lemon", "garlic", "herbs", "black_pepper"]},
    "Prawn masala": {"cal": 155, "p": 17.5, "c": 5.0, "f": 7.5, "fib": 1.0, "ing": ["prawns", "onion", "tomato", "ginger_garlic", "curry_powder", "fennel", "oil"]},
    "Prawn fry": {"cal": 210, "p": 20.0, "c": 6.0, "f": 12.0, "fib": 0.5, "ing": ["prawns", "kashmiri_chilli", "pepper", "curry_leaves", "coconut_oil"]},
    "Crab curry": {"cal": 130, "p": 16.0, "c": 4.2, "f": 5.5, "fib": 0.8, "ing": ["crab", "shallots", "coconut_paste", "fennel", "garlic", "curry_powder", "oil"]},
    "Egg boiled": {"cal": 155, "p": 12.6, "c": 1.1, "f": 10.6, "fib": 0.0, "ing": ["whole_hen_egg", "water"]},
    "Egg omelette": {"cal": 185, "p": 12.0, "c": 2.5, "f": 14.0, "fib": 0.4, "ing": ["eggs", "onion", "green_chilli", "black_pepper", "salt", "oil"]},
    "Egg bhurji": {"cal": 190, "p": 11.8, "c": 3.8, "f": 14.2, "fib": 0.6, "ing": ["eggs", "onion", "tomato", "green_chilli", "coriander", "butter_oil"]},
    "Egg dosa": {"cal": 195, "p": 7.5, "c": 24.0, "f": 7.8, "fib": 1.0, "ing": ["dosa_batter", "whole_egg", "pepper_powder", "oil"]},

    # Fast Food
    "Burger": {"cal": 260, "p": 12.0, "c": 28.0, "f": 11.5, "fib": 1.6, "ing": ["bun", "patty", "lettuce", "onion", "tomato", "sauce"]},
    "Veg burger": {"cal": 220, "p": 7.5, "c": 32.0, "f": 7.5, "fib": 3.2, "ing": ["bun", "veggie_potato_patty", "lettuce", "tomato", "mayo"]},
    "Cheese burger": {"cal": 290, "p": 14.5, "c": 28.0, "f": 14.0, "fib": 1.5, "ing": ["bun", "patty", "cheddar_cheese_slice", "pickles", "ketchup"]},
    "Pizza": {"cal": 265, "p": 11.0, "c": 32.0, "f": 10.0, "fib": 2.0, "ing": ["flour_crust", "pizza_sauce", "mozzarella_cheese", "olive_oil", "herbs"]},
    "Veg pizza": {"cal": 240, "p": 9.5, "c": 33.0, "f": 8.0, "fib": 2.5, "ing": ["flour_crust", "tomato_sauce", "cheese", "capsicum", "onion", "olives"]},
    "Cheese pizza": {"cal": 280, "p": 12.0, "c": 31.0, "f": 12.0, "fib": 1.8, "ing": ["pizza_crust", "marinara_sauce", "blend_of_mozzarella_cheeses"]},
    "Sandwich": {"cal": 210, "p": 7.0, "c": 30.0, "f": 7.0, "fib": 2.0, "ing": ["bread_slices", "cucumber", "tomato", "butter", "chutney"]},
    "Grilled sandwich": {"cal": 255, "p": 8.5, "c": 31.0, "f": 11.0, "fib": 2.1, "ing": ["bread", "cheese", "vegetable_filling", "butter"]},
    "Wraps": {"cal": 220, "p": 8.0, "c": 30.0, "f": 8.0, "fib": 2.5, "ing": ["tortilla_flatbread", "filling", "sauce", "greens"]},
    "Shawarma": {"cal": 240, "p": 14.0, "c": 22.0, "f": 11.0, "fib": 1.5, "ing": ["pita_flatbread", "marinated_meat", "tahini_garlic_sauce", "pickles"]},
    "Tacos": {"cal": 210, "p": 9.5, "c": 21.0, "f": 10.0, "fib": 2.4, "ing": ["corn_tortilla", "seasoned_filling", "lettuce", "cheese", "salsa"]},
    "Pasta": {"cal": 160, "p": 5.5, "c": 28.0, "f": 3.0, "fib": 1.8, "ing": ["durum_wheat_pasta", "tomato_sauce", "garlic", "herbs", "olive_oil"]},
    "Macaroni": {"cal": 175, "p": 6.0, "c": 29.0, "f": 4.0, "fib": 1.5, "ing": ["macaroni_elbows", "cheese_sauce", "butter", "milk"]},
    "Noodles": {"cal": 180, "p": 4.5, "c": 32.0, "f": 4.0, "fib": 1.5, "ing": ["wheat_noodles", "shredded_veggies", "soy_sauce", "chilli_oil"]},
    "Fried rice": {"cal": 195, "p": 4.0, "c": 34.0, "f": 5.0, "fib": 1.4, "ing": ["steamed_rice", "egg_or_vegetables", "soy_sauce", "scallions", "oil"]},
    "French fries": {"cal": 310, "p": 3.4, "c": 41.0, "f": 15.0, "fib": 3.8, "ing": ["potato_strips", "vegetable_oil", "salt"]},
    "Nachos": {"cal": 305, "p": 5.0, "c": 38.0, "f": 15.0, "fib": 3.0, "ing": ["corn_tortilla_chips", "melted_cheese", "jalapenos", "salsa"]},
    "Hot dogs": {"cal": 270, "p": 10.0, "c": 24.0, "f": 15.0, "fib": 1.0, "ing": ["hot_dog_bun", "frankfurter_sausage", "mustard", "ketchup"]},

    # Nuts & Seeds
    "Almonds": {"cal": 579, "p": 21.2, "c": 21.6, "f": 49.9, "fib": 12.5, "ing": ["whole_almonds"]},
    "Cashews": {"cal": 553, "p": 18.2, "c": 30.2, "f": 43.8, "fib": 3.3, "ing": ["whole_cashews"]},
    "Walnuts": {"cal": 654, "p": 15.2, "c": 13.7, "f": 65.2, "fib": 6.7, "ing": ["shelled_walnuts"]},
    "Pistachios": {"cal": 562, "p": 20.2, "c": 27.5, "f": 45.3, "fib": 10.6, "ing": ["pistachio_nuts"]},
    "Peanuts": {"cal": 567, "p": 25.8, "c": 16.1, "f": 49.2, "fib": 8.5, "ing": ["roasted_peanuts"]},
    "Hazelnuts": {"cal": 628, "p": 15.0, "c": 16.7, "f": 60.8, "fib": 9.7, "ing": ["hazelnuts"]},
    "Brazil nuts": {"cal": 659, "p": 14.3, "c": 12.3, "f": 67.1, "fib": 7.5, "ing": ["brazil_nuts"]},
    "Macadamia nuts": {"cal": 718, "p": 7.9, "c": 13.8, "f": 75.8, "fib": 8.6, "ing": ["macadamia_nuts"]},
    "Pecans": {"cal": 691, "p": 9.2, "c": 13.9, "f": 72.0, "fib": 9.6, "ing": ["pecan_nuts"]},
    "Chia seeds": {"cal": 486, "p": 16.5, "c": 42.1, "f": 30.7, "fib": 34.4, "ing": ["whole_chia_seeds"]},
    "Flax seeds": {"cal": 534, "p": 18.3, "c": 28.9, "f": 42.2, "fib": 27.3, "ing": ["whole_flax_seeds"]},
    "Pumpkin seeds": {"cal": 559, "p": 30.2, "c": 10.7, "f": 49.1, "fib": 6.0, "ing": ["pepita_pumpkin_seeds"]},
    "Sunflower seeds": {"cal": 584, "p": 20.8, "c": 20.0, "f": 51.5, "fib": 8.6, "ing": ["hulled_sunflower_seeds"]},
    "Sesame seeds": {"cal": 573, "p": 17.7, "c": 23.4, "f": 49.7, "fib": 11.8, "ing": ["sesame_seeds"]},
    "Watermelon seeds": {"cal": 557, "p": 28.3, "c": 15.3, "f": 47.4, "fib": 5.0, "ing": ["dried_watermelon_seeds"]},
    "Melon seeds": {"cal": 550, "p": 26.0, "c": 18.0, "f": 46.0, "fib": 6.0, "ing": ["melon_seeds_magaz"]},
    "Hemp seeds": {"cal": 553, "p": 31.6, "c": 8.7, "f": 48.8, "fib": 4.0, "ing": ["hulled_hemp_hearts"]},
    "Basil seeds": {"cal": 450, "p": 14.8, "c": 43.0, "f": 25.0, "fib": 37.0, "ing": ["sabja_sweet_basil_seeds"]},
    "Poppy seeds": {"cal": 525, "p": 18.0, "c": 28.1, "f": 41.6, "fib": 19.5, "ing": ["khus_khus_poppy_seeds"]},

    # Gym & Sports Nutrition
    "Whey protein": {"cal": 380, "p": 78.0, "c": 6.5, "f": 4.8, "fib": 0.0, "ing": ["whey_protein_concentrate_powder"]},
    "Whey isolate": {"cal": 370, "p": 90.0, "c": 1.5, "f": 1.0, "fib": 0.0, "ing": ["whey_protein_isolate_powder"]},
    "Casein protein": {"cal": 365, "p": 76.0, "c": 7.0, "f": 3.5, "fib": 0.0, "ing": ["micellar_casein_powder"]},
    "Plant protein": {"cal": 375, "p": 72.0, "c": 10.0, "f": 5.0, "fib": 4.0, "ing": ["pea_protein_isolate", "brown_rice_protein"]},
    "Soy protein": {"cal": 380, "p": 85.0, "c": 2.0, "f": 3.0, "fib": 1.0, "ing": ["soy_protein_isolate"]},
    "Creatine": {"cal": 0, "p": 0.0, "c": 0.0, "f": 0.0, "fib": 0.0, "ing": ["creatine_monohydrate_powder"]},
    "Protein bars": {"cal": 385, "p": 30.0, "c": 38.0, "f": 12.0, "fib": 8.0, "ing": ["protein_blend", "dates_or_maltitol", "oats", "almonds", "cocoa"]},
    "Mass gainers": {"cal": 395, "p": 25.0, "c": 68.0, "f": 3.0, "fib": 2.0, "ing": ["maltodextrin_complex_carbs", "whey_protein", "vitamins"]},
    "Electrolyte drinks": {"cal": 28, "p": 0.0, "c": 7.0, "f": 0.0, "fib": 0.0, "ing": ["water", "sodium_chloride", "potassium_citrate", "glucose", "flavour"]},
    "Pre-workout drinks": {"cal": 15, "p": 1.0, "c": 2.0, "f": 0.0, "fib": 0.0, "ing": ["caffeine_anhydrous", "beta_alanine", "l_citrulline_malate", "electrolytes"]},
    "Recovery shakes": {"cal": 160, "p": 15.0, "c": 20.0, "f": 2.5, "fib": 1.5, "ing": ["milk", "whey_protein", "banana", "honey"]}
}

# Generic defaults for raw categories
def get_default_nutrition_and_ingredients(name, original_category):
    cat_lower = original_category.lower()
    if "fruit" in cat_lower or original_category == "Fruits":
        return {"cal": 52, "p": 0.6, "c": 13.5, "f": 0.2, "fib": 2.4, "ing": [f"fresh_{name.lower()}"]}
    elif "veg" in cat_lower or original_category == "Vegetables":
        return {"cal": 35, "p": 1.8, "c": 7.0, "f": 0.2, "fib": 2.8, "ing": [f"fresh_{name.lower()}"]}
    elif "sprout" in cat_lower or "legume" in cat_lower:
        return {"cal": 140, "p": 9.5, "c": 22.0, "f": 1.2, "fib": 6.5, "ing": [f"boiled_{name.lower()}"]}
    elif "drink" in cat_lower or "beverage" in cat_lower:
        if name in ["Water"]:
            return {"cal": 0, "p": 0.0, "c": 0.0, "f": 0.0, "fib": 0.0, "ing": ["pure_water"]}
        elif name in ["Green tea", "Tea"]:
            return {"cal": 2, "p": 0.2, "c": 0.3, "f": 0.0, "fib": 0.0, "ing": ["brewed_tea_leaves", "water"]}
        elif name in ["Milk", "Whole milk"]:
            return {"cal": 62, "p": 3.2, "c": 4.8, "f": 3.6, "fib": 0.0, "ing": ["whole_cow_milk"]}
        elif name in ["Skimmed milk"]:
            return {"cal": 35, "p": 3.4, "c": 5.0, "f": 0.2, "fib": 0.0, "ing": ["skimmed_cow_milk"]}
        elif "juice" in name.lower():
            return {"cal": 45, "p": 0.7, "c": 10.5, "f": 0.2, "fib": 0.4, "ing": [f"fresh_{name.lower()}"]}
        else:
            return {"cal": 40, "p": 0.5, "c": 9.5, "f": 0.1, "fib": 0.0, "ing": [f"{name.lower()}"]}
    elif "dairy" in cat_lower:
        if name in ["Curd", "Greek yogurt"]:
            return {"cal": 60, "p": 4.0, "c": 4.5, "f": 3.2, "fib": 0.0, "ing": ["fermented_milk_cultures"]}
        elif name in ["Butter", "Ghee"]:
            return {"cal": 880, "p": 0.5, "c": 0.0, "f": 99.0, "fib": 0.0, "ing": ["clarified_butterfat"]}
        elif name in ["Paneer", "Cheese"]:
            return {"cal": 290, "p": 18.0, "c": 4.0, "f": 22.0, "fib": 0.0, "ing": ["coagulated_milk_curd"]}
        else:
            return {"cal": 65, "p": 3.3, "c": 4.8, "f": 3.7, "fib": 0.0, "ing": [f"{name.lower()}"]}
    elif "sweet" in cat_lower or "dessert" in cat_lower:
        return {"cal": 320, "p": 4.0, "c": 55.0, "f": 10.0, "fib": 1.0, "ing": ["milk_solid_or_flour", "sugar", "ghee", "cardamom"]}
    elif "snack" in cat_lower:
        return {"cal": 280, "p": 6.0, "c": 42.0, "f": 11.0, "fib": 3.0, "ing": ["gram_flour_or_cereal", "spices", "oil"]}
    else:
        return {"cal": 150, "p": 5.0, "c": 22.0, "f": 4.0, "fib": 2.0, "ing": [f"{name.lower()}"]}

# Build Master Maps
master_dishes_map = {}
master_home_hotel = {}
all_ingredients_set = set()
master_csv_rows = []

for row in classes_info:
    name = row["class_name"]
    cid = int(row["class_id"])
    orig_cat = row["original_category"]
    diet_type = row["diet_type"]
    prep = row["preparation_method"]
    state = row["food_state"]

    # Retrieve or build nutrition & ingredients
    if name in BASE_NUTRITION:
        nut_ing = BASE_NUTRITION[name]
    else:
        nut_ing = get_default_nutrition_and_ingredients(name, orig_cat)

    cal = nut_ing["cal"]
    p = nut_ing["p"]
    c = nut_ing["c"]
    f_fat = nut_ing["f"]
    fib = nut_ing["fib"]
    ing_list = nut_ing["ing"]

    for ing in ing_list:
        all_ingredients_set.add(ing)

    # Fitness recommendation based on category & macros
    if p >= 15.0 or "Gym" in orig_cat or "Non-Veg" in orig_cat:
        fit_goal = "Muscle Building / High Protein"
    elif cal <= 60 and fib >= 2.0:
        fit_goal = "Weight Loss / Fat Burn"
    elif cal >= 350 or "Fast Food" in orig_cat:
        fit_goal = "Weight Gain / Bulking Energy"
    else:
        fit_goal = "Balanced Diet / Daily Fitness"

    portion_desc = "1 serving (~100g)" if "Solid" in row.get("fitness_role", "") else "1 portion"
    if "Fruit" in orig_cat:
        portion_desc = "1 medium fruit (~120g)"
    elif "Drink" in orig_cat or "Gym" in orig_cat:
        portion_desc = "1 glass / scoop (~250ml or 30g)"
    elif "Nut" in orig_cat or "Seed" in orig_cat:
        portion_desc = "1 handful (~30g)"

    # Construct dishes_ingredients_map_v2 entry
    nutrition_dict = {
        "calories_kcal": cal,
        "protein_g": p,
        "carbohydrates_g": c,
        "fat_g": f_fat,
        "fiber_g": fib
    }
    master_dishes_map[name] = {
        "food_id": cid,
        "food_name": name,
        "category": orig_cat,
        "group_category": row["group_category"],
        "diet_type": diet_type,
        "fitness_goal": fit_goal,
        "portion_size": portion_desc,
        "nutrition": nutrition_dict,
        "nutrition_per_100g": nutrition_dict,
        "main_ingredients": ing_list,
        "ingredient_display": ", ".join([i.replace("_", " ").title() for i in ing_list]),
        "cooking_method": prep if prep else "Standard Preparation",
        "food_state": state
    }

    # Construct home_hotel_recipes_v2 entry
    home_cal = round(cal * 0.95, 1)
    home_fat = round(f_fat * 0.85, 1)
    hotel_cal = round(cal * 1.18, 1)
    hotel_fat = round(f_fat * 1.35, 1)

    master_home_hotel[name] = {
        "food_id": cid,
        "food_name": name,
        "category": orig_cat,
        "home": {
            "portion_desc": portion_desc + " (Home Style)",
            "total_weight_g": 120,
            "ingredients": {ing: round(100.0 / max(len(ing_list), 1), 1) for ing in ing_list},
            "nutrition": {
                "calories_kcal": home_cal,
                "protein_g": p,
                "carbs_g": c,
                "fat_g": home_fat,
                "fiber_g": fib
            },
            "note": "Prepared at home with controlled oil/sugar and natural ingredients."
        },
        "hotel": {
            "portion_desc": portion_desc + " (Restaurant / Mess Style)",
            "total_weight_g": 140,
            "ingredients": {ing: round(110.0 / max(len(ing_list), 1), 1) for ing in ing_list},
            "nutrition": {
                "calories_kcal": hotel_cal,
                "protein_g": p,
                "carbs_g": round(c * 1.05, 1),
                "fat_g": hotel_fat,
                "fiber_g": fib
            },
            "note": "Restaurant preparation typically includes higher oil, butter, or seasoning."
        },
        "adjustable_keys": ing_list[:4]
    }

    # Row for CSV
    master_csv_rows.append({
        "food_id": cid,
        "food_name": name,
        "category": orig_cat,
        "group_category": row["group_category"],
        "diet_type": diet_type,
        "fitness_goal": fit_goal,
        "calories_per_100g": cal,
        "protein_g": p,
        "carbohydrates_g": c,
        "fat_g": f_fat,
        "fiber_g": fib,
        "ingredients": ", ".join(ing_list)
    })

# Save JSON Files
dishes_json_path = os.path.join(V2_USED_DATA, "dishes_ingredients_map_v2.json")
with open(dishes_json_path, "w", encoding="utf-8") as f:
    json.dump(master_dishes_map, f, indent=2)
print(f"Wrote {dishes_json_path} ({len(master_dishes_map)} foods)")

home_hotel_json_path = os.path.join(V2_USED_DATA, "home_hotel_recipes_v2.json")
with open(home_hotel_json_path, "w", encoding="utf-8") as f:
    json.dump(master_home_hotel, f, indent=2)
print(f"Wrote {home_hotel_json_path} ({len(master_home_hotel)} foods)")

# Save Ingredients Vocabulary CSV
vocab_csv_path = os.path.join(V2_USED_DATA, "ingredients_vocabulary_v2.csv")
with open(vocab_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ingredient_id", "ingredient_key", "display_name"])
    for idx, ing in enumerate(sorted(all_ingredients_set)):
        writer.writerow([idx, ing, ing.replace("_", " ").title()])
print(f"Wrote {vocab_csv_path} ({len(all_ingredients_set)} unique ingredients)")

# Save Master CSV
master_csv_path = os.path.join(V2_USED_DATA, "food_and_nutrition_master_v2.csv")
with open(master_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(master_csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(master_csv_rows)
print(f"Wrote {master_csv_path} ({len(master_csv_rows)} rows)")

# Save dataset.yaml for Ingredient Identification v2
yaml_content = {
    "version": 2.0,
    "name": "Version_2_Ingredient_Identification_Master_Dataset",
    "num_foods": len(master_dishes_map),
    "num_ingredients": len(all_ingredients_set),
    "dishes_map": "dishes_ingredients_map_v2.json",
    "recipes": "home_hotel_recipes_v2.json",
    "vocabulary": "ingredients_vocabulary_v2.csv",
    "categories_count": 17,
    "classes": {cid: row["food_name"] for cid, row in enumerate(master_csv_rows)}
}

dataset_yaml_path = os.path.join(V2_USED_DATA, "dataset.yaml")
with open(dataset_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(yaml_content, f, sort_keys=False)
print(f"Wrote {dataset_yaml_path}")

print("\nAll Version 2 Ingredient Identification Master Datasets generated successfully!")
