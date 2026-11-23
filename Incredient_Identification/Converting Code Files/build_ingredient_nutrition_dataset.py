"""
Builder for Model 2: Ingredient Identification & Nutrition Dataset
Mapped directly to the 31 South Indian Food Classes of Food_Detection.

Pipeline:
Food Image -> Food Detection (YOLOv8) -> Food Name -> Ingredient Identification -> Ingredients & Nutrition Facts
"""

import os
import json
import pandas as pd
import yaml

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "DataSet", "Version1 Used Data")
os.makedirs(DATASET_DIR, exist_ok=True)

# 31 Exact Classes from Food_Detection/DataSet/Version1 Used Data/dataset.yaml
DISHES_DATA = [
    {
        "food_id": 0,
        "food_name": "Appam",
        "regional_name": "ஆப்பம் / Palappam",
        "category": "Breakfast / Tiffin",
        "dietary_type": "Vegetarian",
        "portion_size": "2 pieces (150g)",
        "calories_kcal": 240,
        "protein_g": 4.2,
        "carbs_g": 48.0,
        "fat_g": 3.5,
        "fiber_g": 1.8,
        "cooking_method": "Fermented & Steamed/Shallow Cooked in Appachatti",
        "main_ingredients": "raw_rice;parboiled_rice;coconut_milk;sugar;yeast",
        "ingredient_display": "Raw Rice, Parboiled Rice, Fresh Coconut Milk, Sugar, Yeast",
        "health_benefits": "Gluten-Free, Easily Digestible, Fermented Probiotics, Low Sodium"
    },
    {
        "food_id": 1,
        "food_name": "Beetroot poriyal",
        "regional_name": "பீட்ரூட் பொரியல்",
        "category": "Side Dish / Poriyal",
        "dietary_type": "Vegetarian / Vegan",
        "portion_size": "1 cup (150g)",
        "calories_kcal": 110,
        "protein_g": 2.8,
        "carbs_g": 16.5,
        "fat_g": 4.2,
        "fiber_g": 4.5,
        "cooking_method": "Sautéed with Tempering & Grated Coconut",
        "main_ingredients": "beetroot;coconut;mustard_seeds;urad_dal;green_chilli;curry_leaves;vegetable_oil",
        "ingredient_display": "Beetroot, Fresh Grated Coconut, Mustard Seeds, Urad Dal, Green Chilli, Curry Leaves, Oil",
        "health_benefits": "High Dietary Fiber, Rich in Folate & Iron, Lowers Blood Pressure, Antioxidant Betalains"
    },
    {
        "food_id": 2,
        "food_name": "Boiled Egg",
        "regional_name": "அவித்த முட்டை",
        "category": "Protein Side Dish",
        "dietary_type": "Non-Vegetarian (Eggetarian)",
        "portion_size": "1 large egg (50g)",
        "calories_kcal": 78,
        "protein_g": 6.3,
        "carbs_g": 0.6,
        "fat_g": 5.3,
        "fiber_g": 0.0,
        "cooking_method": "Hard Boiled",
        "main_ingredients": "egg;black_pepper",
        "ingredient_display": "Whole Egg, Black Pepper, Salt",
        "health_benefits": "Complete High-Quality Protein, Rich in Choline, Vitamin B12, Zero Glycemic Impact"
    },
    {
        "food_id": 3,
        "food_name": "Carrot poriyal",
        "regional_name": "கேரட் பொரியல்",
        "category": "Side Dish / Poriyal",
        "dietary_type": "Vegetarian / Vegan",
        "portion_size": "1 cup (150g)",
        "calories_kcal": 95,
        "protein_g": 2.1,
        "carbs_g": 14.0,
        "fat_g": 3.8,
        "fiber_g": 4.0,
        "cooking_method": "Sautéed with Mustard Tempering & Coconut",
        "main_ingredients": "carrot;coconut;mustard_seeds;urad_dal;green_chilli;curry_leaves;vegetable_oil",
        "ingredient_display": "Carrot, Grated Coconut, Mustard Seeds, Urad Dal, Green Chilli, Curry Leaves, Oil",
        "health_benefits": "Rich in Beta-Carotene (Vitamin A), Supports Eye Health, High Fiber, Low Calorie"
    },
    {
        "food_id": 4,
        "food_name": "Chicken 65",
        "regional_name": "சிக்கன் 65",
        "category": "Appetizer / Starter",
        "dietary_type": "Non-Vegetarian",
        "portion_size": "1 serving (150g)",
        "calories_kcal": 340,
        "protein_g": 28.5,
        "carbs_g": 8.2,
        "fat_g": 21.0,
        "fiber_g": 1.2,
        "cooking_method": "Marinated & Deep Fried",
        "main_ingredients": "chicken;ginger;garlic;curd;red_chilli_powder;corn_flour;curry_leaves;lemon_juice;vegetable_oil",
        "ingredient_display": "Boneless Chicken, Ginger, Garlic, Curd, Kashmiri Red Chilli, Corn Flour, Curry Leaves, Lemon Juice",
        "health_benefits": "High Muscle-Building Protein, Rich in Niacin, Vitamin B6, Highly Satiating"
    },
    {
        "food_id": 5,
        "food_name": "Chicken briyani",
        "regional_name": "சிக்கன் பிரியாணி / Dindigul Chicken Biryani",
        "category": "Main Course",
        "dietary_type": "Non-Vegetarian",
        "portion_size": "1 plate (350g)",
        "calories_kcal": 560,
        "protein_g": 32.0,
        "carbs_g": 68.0,
        "fat_g": 18.5,
        "fiber_g": 3.5,
        "cooking_method": "Dum Cooking (Slow Steam Infusion)",
        "main_ingredients": "seeraga_samba_rice;chicken;onion;tomato;curd;ginger;garlic;mint_leaves;coriander_leaves;green_chilli;cinnamon;clove;cardamom;ghee;vegetable_oil",
        "ingredient_display": "Seeraga Samba Rice, Tender Chicken, Onion, Tomato, Curd, Ginger-Garlic Paste, Mint, Coriander, Spices, Ghee",
        "health_benefits": "Balanced Macronutrient Profile, Rich in Lean Protein, Energy-Dense, Antioxidant Spices"
    },
    {
        "food_id": 6,
        "food_name": "Dosa",
        "regional_name": "தோசை / Plain Dosa",
        "category": "Breakfast / Tiffin",
        "dietary_type": "Vegetarian",
        "portion_size": "2 plain dosas (160g)",
        "calories_kcal": 265,
        "protein_g": 6.8,
        "carbs_g": 45.2,
        "fat_g": 6.5,
        "fiber_g": 2.6,
        "cooking_method": "Fermented Batter Griddled on Tawa",
        "main_ingredients": "parboiled_rice;raw_rice;urad_dal;fenugreek;sesame_oil",
        "ingredient_display": "Parboiled Rice, Raw Rice, Urad Dal, Fenugreek Seeds, Gingelly / Sesame Oil",
        "health_benefits": "Fermented Probiotic Food, Gut Microbiome Support, Moderate Glycemic Index, Gluten-Free"
    },
    {
        "food_id": 7,
        "food_name": "Idly",
        "regional_name": "இட்லி / Steamed Idli",
        "category": "Breakfast / Tiffin",
        "dietary_type": "Vegetarian",
        "portion_size": "2 pieces (120g)",
        "calories_kcal": 130,
        "protein_g": 4.2,
        "carbs_g": 26.5,
        "fat_g": 0.4,
        "fiber_g": 1.6,
        "cooking_method": "Steam Cooked (Zero Oil)",
        "main_ingredients": "idli_rice;urad_dal;fenugreek",
        "ingredient_display": "Idli Rice (Parboiled), Whole White Urad Dal, Fenugreek Seeds",
        "health_benefits": "Zero Added Oil, Steamed & Heart-Healthy, Easily Digestible, Probiotic Gut Benefits"
    },
    {
        "food_id": 8,
        "food_name": "Kaara chutney",
        "regional_name": "கார சட்னி / Spicy Red Chutney",
        "category": "Accompaniment / Chutney",
        "dietary_type": "Vegetarian / Vegan",
        "portion_size": "3 tablespoons (50g)",
        "calories_kcal": 65,
        "protein_g": 1.2,
        "carbs_g": 7.5,
        "fat_g": 3.6,
        "fiber_g": 1.4,
        "cooking_method": "Sautéed and Ground",
        "main_ingredients": "tomato;onion;red_chilli;garlic;tamarind;mustard_seeds;curry_leaves;sesame_oil",
        "ingredient_display": "Ripe Tomato, Onion, Dried Red Chilli, Garlic, Tamarind, Mustard Seeds, Curry Leaves, Gingelly Oil",
        "health_benefits": "Rich in Lycopene & Allicin, Metabolism Boosting, Digestive Aid"
    },
    {
        "food_id": 9,
        "food_name": "Kali",
        "regional_name": "உளுந்தங்களி / Ulundhu Kali",
        "category": "Traditional Health Food",
        "dietary_type": "Vegetarian",
        "portion_size": "1 cup (150g)",
        "calories_kcal": 320,
        "protein_g": 8.5,
        "carbs_g": 58.0,
        "fat_g": 7.2,
        "fiber_g": 5.8,
        "cooking_method": "Slow Cooked Porridge with Sesame Oil",
        "main_ingredients": "black_urad_dal;rice;jaggery;sesame_oil;cardamom",
        "ingredient_display": "Whole Black Urad Dal, Raw Rice, Organic Jaggery, Cold-Pressed Gingelly Oil, Cardamom",
        "health_benefits": "Strengthens Bones & Spine, High Iron & Calcium, Restores Postpartum Energy"
    },
    {
        "food_id": 10,
        "food_name": "Koozh",
        "regional_name": "கேழ்வரகு கூழ் / Keppai Koozh",
        "category": "Traditional Summer Beverage / Breakfast",
        "dietary_type": "Vegetarian",
        "portion_size": "1 bowl (250ml)",
        "calories_kcal": 180,
        "protein_g": 5.5,
        "carbs_g": 34.0,
        "fat_g": 2.5,
        "fiber_g": 4.8,
        "cooking_method": "Fermented Ragi Porridge with Buttermilk",
        "main_ingredients": "ragi;rice;buttermilk;shallots;green_chilli;curry_leaves",
        "ingredient_display": "Finger Millet (Ragi), Broken Rice, Fermented Sour Buttermilk, Small Onions, Green Chilli",
        "health_benefits": "Natural Body Coolant, Extremely High Calcium, Probiotic Gut Immunity, Low Glycemic Index"
    },
    {
        "food_id": 11,
        "food_name": "Lemon Rice",
        "regional_name": "எலுமிச்சை சாதம் / Chitranna",
        "category": "Variety Rice",
        "dietary_type": "Vegetarian / Vegan",
        "portion_size": "1 plate (200g)",
        "calories_kcal": 310,
        "protein_g": 6.2,
        "carbs_g": 52.0,
        "fat_g": 8.8,
        "fiber_g": 2.8,
        "cooking_method": "Tossed Cooked Rice with Lemon Tempering",
        "main_ingredients": "rice;lemon_juice;peanuts;turmeric;mustard_seeds;green_chilli;curry_leaves;ginger;chana_dal;urad_dal;asafoetida;vegetable_oil",
        "ingredient_display": "Cooked Rice, Fresh Lemon Juice, Roasted Peanuts, Turmeric, Mustard, Green Chilli, Curry Leaves, Lentils",
        "health_benefits": "Rich in Vitamin C, Anti-inflammatory Curcumin, Satiating Healthy Fats from Peanuts"
    },
    {
        "food_id": 12,
        "food_name": "Mushroom briyani",
        "regional_name": "காளான் பிரியாணி",
        "category": "Main Course",
        "dietary_type": "Vegetarian",
        "portion_size": "1 plate (320g)",
        "calories_kcal": 430,
        "protein_g": 11.5,
        "carbs_g": 69.0,
        "fat_g": 12.0,
        "fiber_g": 5.2,
        "cooking_method": "Dum Cooked Rice with Spiced Mushrooms",
        "main_ingredients": "seeraga_samba_rice;mushroom;onion;tomato;curd;mint_leaves;coriander_leaves;ginger;garlic;cinnamon;clove;cardamom;ghee;vegetable_oil",
        "ingredient_display": "Seeraga Samba Rice, Fresh Button Mushrooms, Onion, Tomato, Curd, Mint, Coriander, Spices, Ghee",
        "health_benefits": "Rich in Selenium & Ergothioneine Antioxidants, Vitamin D, Good Source of Plant Protein"
    },
    {
        "food_id": 13,
        "food_name": "Mutton Briyani",
        "regional_name": "மட்டன் பிரியாணி / Chettinad Mutton Biryani",
        "category": "Main Course",
        "dietary_type": "Non-Vegetarian",
        "portion_size": "1 plate (350g)",
        "calories_kcal": 620,
        "protein_g": 36.0,
        "carbs_g": 65.0,
        "fat_g": 24.5,
        "fiber_g": 3.2,
        "cooking_method": "Dum Cooked Aromatic Rice with Tender Goat Meat",
        "main_ingredients": "seeraga_samba_rice;mutton;onion;tomato;curd;ginger;garlic;mint_leaves;coriander_leaves;cinnamon;clove;cardamom;ghee",
        "ingredient_display": "Seeraga Samba Rice, Goat Meat (Mutton), Onion, Tomato, Curd, Ginger-Garlic Paste, Mint, Spices, Pure Ghee",
        "health_benefits": "High Bioavailable Heme Iron, Zinc, Vitamin B12, High Muscle Protein"
    },
    {
        "food_id": 14,
        "food_name": "Nandu masala",
        "regional_name": "நண்டு மசாலா / Chettinad Crab Curry",
        "category": "Seafood Main Dish",
        "dietary_type": "Non-Vegetarian",
        "portion_size": "1 serving (200g)",
        "calories_kcal": 240,
        "protein_g": 26.0,
        "carbs_g": 9.5,
        "fat_g": 11.0,
        "fiber_g": 2.2,
        "cooking_method": "Simmered in Peppery Chettinad Gravy",
        "main_ingredients": "crab;shallots;tomato;ginger;garlic;black_pepper;fennel_seeds;coriander_seeds;turmeric;curry_leaves;coconut_oil",
        "ingredient_display": "Fresh Sea Crab, Small Onions, Tomato, Ginger, Garlic, Fresh Ground Black Pepper, Fennel, Curry Leaves, Coconut Oil",
        "health_benefits": "Rich in Omega-3 Fatty Acids, Excellent Cold & Respiratory Relief, Lean Protein"
    },
    {
        "food_id": 15,
        "food_name": "Nei satham",
        "regional_name": "நெய் சாதம் / Ghee Rice",
        "category": "Special Rice",
        "dietary_type": "Vegetarian",
        "portion_size": "1 plate (200g)",
        "calories_kcal": 380,
        "protein_g": 5.8,
        "carbs_g": 53.0,
        "fat_g": 16.0,
        "fiber_g": 1.5,
        "cooking_method": "Tempered Rice Cooked in Cow Ghee",
        "main_ingredients": "rice;ghee;cashew_nuts;cumin_seeds;black_pepper;curry_leaves",
        "ingredient_display": "Ponni Rice, Pure Cow Ghee, Roasted Cashews, Cumin Seeds, Whole Black Pepper, Curry Leaves",
        "health_benefits": "Butyric Acid for Gut Lining, Fat-Soluble Vitamin Absorption, High Quick Energy"
    },
    {
        "food_id": 16,
        "food_name": "Paal kolukattai",
        "regional_name": "பால் கொழுக்கட்டை",
        "category": "Traditional Dessert",
        "dietary_type": "Vegetarian",
        "portion_size": "1 bowl (180g)",
        "calories_kcal": 290,
        "protein_g": 4.8,
        "carbs_g": 52.0,
        "fat_g": 8.0,
        "fiber_g": 2.0,
        "cooking_method": "Steamed Rice Dumplings Simmered in Sweet Coconut Milk",
        "main_ingredients": "rice_flour;coconut_milk;milk;jaggery;cardamom",
        "ingredient_display": "Rice Flour Dumplings, Thick Coconut Milk, Cow Milk, Jaggery, Cardamom Powder",
        "health_benefits": "Gluten-Free, Unrefined Natural Sweetener (Jaggery), Lauric Acid from Coconut Milk"
    },
    {
        "food_id": 17,
        "food_name": "Paneer briyani",
        "regional_name": "பன்னீர் பிரியாணி",
        "category": "Main Course",
        "dietary_type": "Vegetarian",
        "portion_size": "1 plate (330g)",
        "calories_kcal": 510,
        "protein_g": 18.5,
        "carbs_g": 66.0,
        "fat_g": 19.5,
        "fiber_g": 4.0,
        "cooking_method": "Dum Cooked Rice with Spiced Cottage Cheese",
        "main_ingredients": "seeraga_samba_rice;paneer;onion;tomato;curd;mint_leaves;coriander_leaves;ginger;garlic;cinnamon;clove;cardamom;ghee",
        "ingredient_display": "Seeraga Samba Rice, Fresh Cottage Cheese (Paneer), Onion, Tomato, Curd, Mint, Coriander, Spices, Ghee",
        "health_benefits": "Excellent Vegetarian Protein & Calcium, Sustained Satiety, Wholesome Meal"
    },
    {
        "food_id": 18,
        "food_name": "Panner masala",
        "regional_name": "பன்னீர் மசாலா / Paneer Butter Masala",
        "category": "Curry / Gravy",
        "dietary_type": "Vegetarian",
        "portion_size": "1 cup (200g)",
        "calories_kcal": 360,
        "protein_g": 15.0,
        "carbs_g": 14.0,
        "fat_g": 27.0,
        "fiber_g": 3.2,
        "cooking_method": "Simmered in Rich Tomato-Cashew Gravy",
        "main_ingredients": "paneer;onion;tomato;ginger;garlic;cashew_nuts;butter;garam_masala",
        "ingredient_display": "Paneer Cubes, Onion, Ripe Tomatoes, Ginger, Garlic, Cashew Paste, Butter, Garam Masala",
        "health_benefits": "High Calcium for Bone Density, Quality Dairy Protein, Rich Satiating Fats"
    },
    {
        "food_id": 19,
        "food_name": "Parupu vada",
        "regional_name": "பருப்பு வடை / Masala Vadai",
        "category": "Snack",
        "dietary_type": "Vegetarian / Vegan",
        "portion_size": "2 pieces (80g)",
        "calories_kcal": 210,
        "protein_g": 7.2,
        "carbs_g": 24.5,
        "fat_g": 9.8,
        "fiber_g": 4.8,
        "cooking_method": "Coarsely Ground Bengal Gram Deep Fried",
        "main_ingredients": "chana_dal;onion;green_chilli;ginger;fennel_seeds;curry_leaves;coriander_leaves;asafoetida;vegetable_oil",
        "ingredient_display": "Coarse Chana Dal (Bengal Gram), Chopped Onion, Green Chilli, Ginger, Fennel, Curry Leaves, Oil",
        "health_benefits": "High Dietary Fiber & Plant Protein, Low Glycemic Index Pulse, Crispy Tea-Time Snack"
    },
    {
        "food_id": 20,
        "food_name": "Pidi kolukattai",
        "regional_name": "பிடி கொழுக்கட்டை / Uppu & Vellam Kolukattai",
        "category": "Traditional Snack",
        "dietary_type": "Vegetarian",
        "portion_size": "3 dumplings (120g)",
        "calories_kcal": 220,
        "protein_g": 3.2,
        "carbs_g": 46.0,
        "fat_g": 3.5,
        "fiber_g": 2.5,
        "cooking_method": "Hand Pressed & Steamed",
        "main_ingredients": "rice_flour;jaggery;coconut;cardamom;ghee",
        "ingredient_display": "Rice Flour, Jaggery, Fresh Grated Coconut, Cardamom Powder, Touch of Ghee",
        "health_benefits": "Steam Cooked with Minimal Fat, Clean Fuel from Natural Jaggery, Gluten-Free"
    },
    {
        "food_id": 21,
        "food_name": "Poorna kolukattai",
        "regional_name": "பூரண கொழுக்கட்டை / Modak",
        "category": "Festive Sweet",
        "dietary_type": "Vegetarian",
        "portion_size": "2 pieces (100g)",
        "calories_kcal": 215,
        "protein_g": 3.0,
        "carbs_g": 42.5,
        "fat_g": 4.2,
        "fiber_g": 2.2,
        "cooking_method": "Steamed Rice Dumpling with Sweet Coconut-Jaggery Stuffing",
        "main_ingredients": "rice_flour;coconut;jaggery;cardamom;ghee",
        "ingredient_display": "Steamed Rice Flour Casing, Sweet Filling of Coconut, Jaggery & Cardamom",
        "health_benefits": "Steamed, No Trans Fats, Natural Mineral-Rich Jaggery, Low Sodium"
    },
    {
        "food_id": 22,
        "food_name": "Prawn thokku",
        "regional_name": "இறால் தொக்கு / Chettinad Prawn Masala",
        "category": "Seafood Main Dish",
        "dietary_type": "Non-Vegetarian",
        "portion_size": "1 cup (180g)",
        "calories_kcal": 225,
        "protein_g": 24.5,
        "carbs_g": 8.0,
        "fat_g": 10.5,
        "fiber_g": 2.0,
        "cooking_method": "Thick Sautéed Seafood Gravy",
        "main_ingredients": "prawns;shallots;tomato;ginger;garlic;red_chilli;turmeric;curry_leaves;fennel_seeds;mustard_seeds;sesame_oil",
        "ingredient_display": "Fresh Prawns / Shrimp, Shallots, Tomato, Ginger, Garlic, Red Chilli, Curry Leaves, Fennel, Gingelly Oil",
        "health_benefits": "Lean Protein, Rich in Selenium, Iodine & Astaxanthin Antioxidant, Low Carbohydrates"
    },
    {
        "food_id": 23,
        "food_name": "Puthina Chutney",
        "regional_name": "புதினா சட்னி / Mint Chutney",
        "category": "Accompaniment / Chutney",
        "dietary_type": "Vegetarian / Vegan",
        "portion_size": "3 tablespoons (50g)",
        "calories_kcal": 55,
        "protein_g": 1.5,
        "carbs_g": 5.0,
        "fat_g": 3.2,
        "fiber_g": 1.8,
        "cooking_method": "Sautéed Fresh Herbs & Ground",
        "main_ingredients": "mint_leaves;coriander_leaves;coconut;green_chilli;ginger;tamarind;urad_dal",
        "ingredient_display": "Fresh Mint Leaves, Coriander Leaves, Grated Coconut, Green Chilli, Ginger, Tamarind, Urad Dal",
        "health_benefits": "Potent Digestive Aid (Menthol), Cleanses Palate, Rich in Chlorophyll & Vitamin C"
    },
    {
        "food_id": 24,
        "food_name": "Sambar",
        "regional_name": "சாம்பார் / South Indian Sambar",
        "category": "Lentil Stew / Curry",
        "dietary_type": "Vegetarian / Vegan",
        "portion_size": "1 cup (200ml)",
        "calories_kcal": 135,
        "protein_g": 6.5,
        "carbs_g": 21.0,
        "fat_g": 3.0,
        "fiber_g": 5.2,
        "cooking_method": "Boiled Lentils & Veggies Stewed in Tamarind Broth with Tempering",
        "main_ingredients": "toor_dal;tamarind;shallots;tomato;drumstick;carrot;sambar_powder;turmeric;mustard_seeds;curry_leaves;asafoetida;vegetable_oil",
        "ingredient_display": "Toor Dal, Tamarind Pulp, Shallots, Tomato, Drumstick, Carrot, Sambar Powder, Mustard, Curry Leaves, Hing",
        "health_benefits": "Complete Plant Protein & Fiber, Digestive Spices (Hing/Turmeric), Nutrient-Dense Vegetables"
    },
    {
        "food_id": 25,
        "food_name": "Sambar satham",
        "regional_name": "சாம்பார் சாதம் / Sambar Sadam",
        "category": "One-Pot Meal",
        "dietary_type": "Vegetarian",
        "portion_size": "1 plate (250g)",
        "calories_kcal": 340,
        "protein_g": 9.8,
        "carbs_g": 58.0,
        "fat_g": 7.5,
        "fiber_g": 6.5,
        "cooking_method": "One-Pot Pressure Cooked Rice & Lentil Stew with Ghee",
        "main_ingredients": "rice;toor_dal;shallots;tomato;drumstick;carrot;tamarind;sambar_powder;ghee;mustard_seeds;curry_leaves;asafoetida",
        "ingredient_display": "Ponni Rice, Toor Dal, Shallots, Tomato, Drumstick, Tamarind, Sambar Spices, Tempering, Cow Ghee",
        "health_benefits": "Complete Cereal + Pulse Protein Complementarity, High Satiety, Wholesome Comfort Food"
    },
    {
        "food_id": 26,
        "food_name": "Satham",
        "regional_name": "சாதம் / Steamed Rice",
        "category": "Staple Grain",
        "dietary_type": "Vegetarian / Vegan",
        "portion_size": "1 cup cooked (150g)",
        "calories_kcal": 195,
        "protein_g": 4.0,
        "carbs_g": 44.0,
        "fat_g": 0.4,
        "fiber_g": 0.8,
        "cooking_method": "Boiled / Steamed in Water",
        "main_ingredients": "rice",
        "ingredient_display": "Steamed South Indian Ponni / Sona Masuri Rice",
        "health_benefits": "Easily Digestible Carbohydrate, Zero Cholesterol, Low Sodium, Primary South Indian Energy Staple"
    },
    {
        "food_id": 27,
        "food_name": "Thengai chutney",
        "regional_name": "தேங்காய் சட்னி / Coconut Chutney",
        "category": "Accompaniment / Chutney",
        "dietary_type": "Vegetarian / Vegan",
        "portion_size": "3 tablespoons (50g)",
        "calories_kcal": 90,
        "protein_g": 1.8,
        "carbs_g": 4.2,
        "fat_g": 7.5,
        "fiber_g": 2.4,
        "cooking_method": "Freshly Ground & Tempered with Mustard & Curry Leaves",
        "main_ingredients": "coconut;roasted_gram;green_chilli;ginger;mustard_seeds;curry_leaves;asafoetida;coconut_oil",
        "ingredient_display": "Fresh Grated Coconut, Roasted Chana Dal (Pottukadala), Green Chilli, Ginger, Mustard Tempering",
        "health_benefits": "Medium Chain Triglycerides (MCTs) for Fast Metabolism, Antimicrobial Lauric Acid"
    },
    {
        "food_id": 28,
        "food_name": "Uzhuntha vadai",
        "regional_name": "உளுந்த வடை / Medu Vada",
        "category": "Snack / Breakfast Side",
        "dietary_type": "Vegetarian",
        "portion_size": "2 pieces (90g)",
        "calories_kcal": 245,
        "protein_g": 7.8,
        "carbs_g": 26.0,
        "fat_g": 12.5,
        "fiber_g": 4.2,
        "cooking_method": "Aerated Urad Dal Batter Deep Fried to Golden Crisp",
        "main_ingredients": "urad_dal;onion;green_chilli;black_pepper;ginger;curry_leaves;asafoetida;vegetable_oil",
        "ingredient_display": "Soaked & Fluffed Urad Dal, Finely Chopped Onion, Black Pepper, Green Chilli, Curry Leaves, Oil",
        "health_benefits": "High Pulse Protein, Satiating Healthy Snack, Rich in Potassium & Magnesium"
    },
    {
        "food_id": 29,
        "food_name": "Veg briyani",
        "regional_name": "காய்கறி பிரியாணி / Vegetable Biryani",
        "category": "Main Course",
        "dietary_type": "Vegetarian",
        "portion_size": "1 plate (320g)",
        "calories_kcal": 410,
        "protein_g": 8.5,
        "carbs_g": 68.0,
        "fat_g": 11.5,
        "fiber_g": 6.2,
        "cooking_method": "Dum Cooked Fragrant Rice with Garden Vegetables",
        "main_ingredients": "seeraga_samba_rice;carrot;french_beans;green_peas;potato;onion;tomato;ginger;garlic;mint_leaves;coriander_leaves;curd;cinnamon;clove;cardamom;ghee",
        "ingredient_display": "Seeraga Samba Rice, Carrots, Beans, Green Peas, Potatoes, Onion, Tomato, Curd, Mint, Biryani Spices, Ghee",
        "health_benefits": "Fiber Rich, Broad Spectrum Micronutrients (Vitamins A, C, K), Plant-Based Wholesome Meal"
    },
    {
        "food_id": 30,
        "food_name": "Ven Pongal",
        "regional_name": "வெண் பொங்கல் / Khara Pongal",
        "category": "Breakfast / Tiffin",
        "dietary_type": "Vegetarian",
        "portion_size": "1 plate (200g)",
        "calories_kcal": 340,
        "protein_g": 8.2,
        "carbs_g": 48.0,
        "fat_g": 13.0,
        "fiber_g": 3.5,
        "cooking_method": "Rice & Moong Dal Cooked Soft and Tempered with Ghee, Pepper & Cumin",
        "main_ingredients": "rice;moong_dal;ghee;cumin_seeds;black_pepper;ginger;curry_leaves;cashew_nuts;asafoetida",
        "ingredient_display": "Raw Rice, Split Yellow Moong Dal, Pure Cow Ghee, Black Peppercorns, Cumin Seeds, Ginger, Curry Leaves, Cashews",
        "health_benefits": "Ayurvedic Tridoshic Healing Meal, Complete Amino Acids (Rice + Moong), Easy on Sensitive Stomachs"
    }
]

def main():
    print("=" * 70)
    print("🚀 GENERATING INGREDIENT & NUTRITION DATASET FOR 31 SOUTH INDIAN DISHES")
    print(f"Target Directory: {DATASET_DIR}")
    print("=" * 70)

    # 1. Generate Primary CSV (south_indian_food_ingredients_nutrition.csv)
    df = pd.DataFrame(DISHES_DATA)
    csv_path = os.path.join(DATASET_DIR, "south_indian_food_ingredients_nutrition.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [1/4] Saved Primary CSV: {csv_path} ({len(df)} dishes)")

    # 2. Generate JSON mapping (dishes_ingredients_map.json)
    json_data = {}
    for item in DISHES_DATA:
        key = item["food_name"]
        json_data[key] = {
            "food_id": item["food_id"],
            "regional_name": item["regional_name"],
            "category": item["category"],
            "dietary_type": item["dietary_type"],
            "portion_size": item["portion_size"],
            "nutrition": {
                "calories_kcal": item["calories_kcal"],
                "protein_g": item["protein_g"],
                "carbs_g": item["carbs_g"],
                "fat_g": item["fat_g"],
                "fiber_g": item["fiber_g"]
            },
            "main_ingredients": item["main_ingredients"].split(";"),
            "ingredient_display": item["ingredient_display"],
            "cooking_method": item["cooking_method"],
            "health_benefits": item["health_benefits"]
        }
    
    json_path = os.path.join(DATASET_DIR, "dishes_ingredients_map.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  [2/4] Saved JSON Mapping: {json_path}")

    # 3. Generate Ingredient Vocabulary (ingredients_vocabulary.csv)
    all_ingredients = {}
    for item in DISHES_DATA:
        ings = item["main_ingredients"].split(";")
        for ing in ings:
            all_ingredients[ing] = all_ingredients.get(ing, 0) + 1
    
    ing_vocab_rows = []
    for ing, count in sorted(all_ingredients.items(), key=lambda x: -x[1]):
        ing_vocab_rows.append({
            "ingredient": ing,
            "dishes_count": count
        })
    ing_df = pd.DataFrame(ing_vocab_rows)
    vocab_path = os.path.join(DATASET_DIR, "ingredients_vocabulary.csv")
    ing_df.to_csv(vocab_path, index=False)
    print(f"  [3/4] Saved Ingredient Vocabulary: {vocab_path} ({len(ing_df)} unique ingredients)")

    # 4. Generate dataset.yaml
    yaml_content = {
        "dataset_name": "South Indian Food Ingredient & Nutrition Dataset",
        "version": "1.0.0",
        "num_classes": len(DISHES_DATA),
        "classes": {item["food_id"]: item["food_name"] for item in DISHES_DATA},
        "csv_path": "south_indian_food_ingredients_nutrition.csv",
        "json_path": "dishes_ingredients_map.json",
        "vocabulary_path": "ingredients_vocabulary.csv"
    }
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"  [4/4] Saved dataset.yaml: {yaml_path}")

    # 5. Generate comprehensive README.md
    readme_content = f"""# 🥗 South Indian Food Ingredient & Nutrition Dataset (Version 1)

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
- **Total Unique Ingredients**: {len(ing_df)} canonical culinary ingredients
- **Files Included**:
  - `south_indian_food_ingredients_nutrition.csv`: Master tabular dataset with all nutritional metrics
  - `dishes_ingredients_map.json`: Fast $O(1)$ lookup mapping for real-time app inference
  - `ingredients_vocabulary.csv`: Vocabulary list of ingredients ranked by dish frequency
  - `dataset.yaml`: Dataset configuration and class IDs

## 📋 The 31 Dishes Covered
{', '.join([d['food_name'] for d in DISHES_DATA])}
"""
    readme_path = os.path.join(DATASET_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  [5/5] Saved README.md: {readme_path}")

    print("\n✅ INGREDIENT & NUTRITION DATASET GENERATION COMPLETE!")


if __name__ == "__main__":
    main()
