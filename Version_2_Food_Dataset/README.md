# 🏋️‍♂️ Version 2 – Complete Food & Nutrition Master Dataset
### *Designed for Gym, Fitness, Bodybuilding, Weight Loss, Bulking & Sports Nutrition*

This directory houses the complete **Version 2 Master Food & Nutrition Dataset**, built from ground up to support comprehensive dietary tracking, gym nutrition calculations, bodybuilder macronutrient queries, and multi-cuisine food detection.

---

## 📁 Master Directory Structure

```
Version_2_Food_Dataset/
├── 01_RAW_DATA/                     # Original, unaltered source datasets & manifests
│   ├── Dairy/                       # Whole milk, skimmed, Greek yogurt, paneer, ghee
│   ├── Desserts/                    # Payasam, halwa, laddu, dark chocolate, kheer
│   ├── Drinks/                      # Coconut water, buttermilk, lassi, green tea
│   ├── Fast_Food/                   # Burgers, pizzas, wraps, shawarma, tacos, pastas
│   ├── Fruits/                      # 22 fresh & dried whole fruits
│   ├── Gym_Nutrition/               # Whey isolate, concentrate, casein, creatine, mass gainers
│   ├── Indian_Food/                 # Pan-Indian composite dishes
│   ├── International_Food/          # International benchmark foods
│   ├── Non_Vegetarian/              # Chicken, mutton, fish, prawns, eggs, tandoori
│   ├── Nutrition_Databases/         # Preserved ICMR IFCT 2017 & Archana's Kitchen raw tables
│   ├── Nuts_Seeds/                  # Almonds, walnuts, chia, flax, pumpkin seeds
│   ├── Salads/                      # Sprouts, chicken, chickpea, fruit salads
│   ├── Smoothies/                   # 16 fitness smoothies with complete ingredient quantities
│   ├── Snacks/                      # Oats, granola, peanut butter, protein bars
│   ├── South_Indian_Food/           # 22 traditional tiffins, ragi dishes, rice varieties
│   ├── Sprouts_Legumes/             # Live sprouts, black chana, horse gram, soybeans
│   ├── Vegetables/                  # 20 raw & steamed garden vegetables
│   ├── Vegetarian/                  # Sambar, rasam, kootu, palak paneer, chana masala
│   ├── food_categories_taxonomies.csv/.json
│   └── RAW_DATA_MANIFEST.csv/.json
│
├── 02_PROCESSED_DATA/               # Cleaned, validated, and normalized intermediate tables
│   ├── cleaned_foods.csv/.json
│   ├── cleaned_nutrition.csv
│   ├── cleaned_recipes.json
│   ├── cleaned_ingredients.csv
│   ├── cleaned_food_categories.json
│   ├── duplicate_records.csv
│   ├── invalid_records.csv
│   └── data_quality_report.json
│
├── 03_FINAL_DATA/                   # Application-ready master databases
│   ├── foods_master.csv/.json       # Master table: 229 foods with both per-100g & per-serving macros
│   ├── recipes_master.json          # 78 component-level recipes (smoothies + home/hotel benchmarks)
│   ├── ingredients_master.json      # 281 mapped recipe ingredients with gram weights
│   ├── nutrition_master.json        # Normalized nutritional composition
│   ├── food_categories.json         # Complete 17-category taxonomy
│   ├── serving_sizes.json           # Serving sizes in grams and descriptive measures
│   ├── food_aliases.json            # Tamil and international search aliases
│   ├── nutrition_sources.json       # Documented research sources & licenses
│   └── food_database.sqlite         # High-speed local SQLite database with indexes
│
└── 04_REPORTS/                      # Validation, license, and quality documentation
    ├── dataset_summary.md           # Master executive summary
    ├── category_summary.csv         # Category-wise food counts & diet breakdown
    ├── data_quality_report.md       # Quality audit & Atwater validation
    ├── source_license_report.md     # Official source licenses & terms
    ├── duplicate_report.csv         # Quarantined cross-listed duplicates
    └── processing_log.txt           # Pipeline execution logs
```

---

## 📊 Dataset Metrics at a Glance

* **Total Cleaned Unique Foods**: **229 Foods**
* **Total Formulated Recipes**: **78 Recipes**
* **Total Recipe-Ingredient Mappings**: **281 Ingredients with Gram Weights**
* **Total Covered Categories**: **17 Mandatory Categories**
* **Primary Scientific Sources**:
  1. *ICMR-NIN Indian Food Composition Tables (IFCT 2017)*
  2. *USDA FoodData Central (Foundation Foods & SR Legacy)*
  3. *Open Food Facts Consortium*
  4. *Verified Sports Nutrition Manufacturer Disclosures*
  5. *Archana's Kitchen Culinary Database & RecipeDB (IIIT-Delhi)*

---

## 🗄️ How to Query the SQLite Database (`food_database.sqlite`)

The SQLite database located at `03_FINAL_DATA/food_database.sqlite` comes pre-indexed for high-performance fitness queries:

### 1. Find Top High-Protein Foods for Bodybuilding (>20g Protein/100g):
```sql
SELECT f.food_name, f.category, n.protein_g, n.calories_kcal 
FROM foods f 
JOIN nutrition n ON f.food_id = n.food_id 
WHERE n.protein_g >= 20.0 
ORDER BY n.protein_g DESC;
```

### 2. Search Foods by Common Tamil or English Aliases:
```sql
SELECT a.alias_name, f.food_name, n.calories_kcal, n.protein_g 
FROM aliases a 
JOIN foods f ON a.canonical_name = f.food_name 
JOIN nutrition n ON f.food_id = n.food_id 
WHERE a.alias_name LIKE '%முட்டை%';
```

### 3. Retrieve Complete Ingredients and Quantities for a Recipe:
```sql
SELECT r.recipe_name, ri.ingredient_name, ri.quantity_g 
FROM recipes r 
JOIN recipe_ingredients ri ON r.recipe_id = ri.recipe_id 
WHERE r.recipe_name = 'Banana smoothie';
```

