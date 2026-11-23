# 📘 South Indian Food Inspection & Nutrition AI: Architecture & Usability Guide

This document provides a comprehensive overview of the **South Indian Food Recognition, Ingredient Prediction & Nutrition Inspection System**, including workspace organization, subsystem connectivity, and end-user usability workflows.

---

## 1. 🏗️ High-Level System Architecture

The project solves the complete journey from an **unannotated image of a food plate** to an **exact personalized macronutrient breakdown**:

```
                         📸 FOOD PLATE IMAGE
                                  │
                                  ▼
                 ┌───────────────────────────────────┐
                 │    MODEL 1: Food Detection        │
                 │    (YOLOv8 Small - 31 Classes)    │
                 └───────────────────────────────────┘
                                  │
                                  │ Bounding Boxes & Dish Names
                                  │ (e.g., "Chicken briyani", "Ven Pongal")
                                  ▼
                 ┌───────────────────────────────────┐
                 │ MODEL 2: Ingredient Identification│
                 │   & Nutrition Multi-Task NN       │
                 └───────────────────────────────────┘
                                  │
                                  │ Ingredients, Macro Densities,
                                  │ Cultural Context & Base Recipes
                                  ▼
                 ┌───────────────────────────────────┐
                 │  DYNAMIC HOME VS. HOTEL ENGINE    │
                 └───────────────────────────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      🏨 HOTEL / RESTAURANT STYLE       🏠 HOME MADE STYLE
      • Overall Serving Weight Input    • Typed Individual Ingredient Weights
      • Automatic Proportional Split    • Add Extra Custom Ingredients
      • Commercial Ghee/Oil Factors     • Real-Time Recalculated Macros
```

---

## 2. 📁 Workspace Directory Structure

The project strictly follows a standardized, modular directory layout mirroring both subsystems:

```
d:\Yolo_Food_Inspection\
│
├── 🚀 app.py                         # Production Streamlit Web User Interface
├── ⚡ run_ui.bat                      # 1-Click Windows Batch Launcher for Streamlit
├── 📄 README.md                       # Project root overview
├── 🔒 .gitignore                      # Git ignore file
│
├── 📂 Docs/                           # Official Technical Documentation
│   ├── README.md                      # Documentation Index
│   ├── 01_PROJECT_ARCHITECTURE_AND_USABILITY.md  # (This file) Overview & Usability
│   ├── 02_MODEL_USAGE_AND_API_INTEGRATION.md     # Inputs, Outputs & Reuse Guide
│   └── 03_TRAINING_AND_EVALUATION_GUIDE.md       # Complete Training Pipeline Guide
│
├── 📂 Food_Detection/                 # MODEL 1 SUBSYSTEM
│   ├── Converting Code Files/         # Raw dataset conversion & split preparation scripts
│   │   ├── convert_voc_to_yolo.py     # XML Pascal VOC to YOLO bbox format converter
│   │   ├── convert_khana_to_yolo_cls.py
│   │   └── prepare_huggig_splits.py
│   ├── DataSet/
│   │   ├── RawData/Version1/          # Original multi-source raw images & labels
│   │   └── Version1 Used Data/        # Official curated training/val/test splits (8,010 images)
│   │       ├── dataset.yaml           # YOLO configuration (31 classes)
│   │       ├── images/ (train, val, test)
│   │       └── labels/ (train, val, test)
│   ├── Models/Food Detection/Version1/
│   │   ├── food_detection_v1_best.pt  # Trained YOLOv8s production weights
│   │   └── best.pt                    # Standard alias
│   └── NoteBooks/
│       ├── Basic Setting/basic_setting.ipynb
│       └── Food Detection/Version1/
│           └── food_detection_v1_training.ipynb  # End-to-end Colab training notebook
│
└── 📂 Incredient_Identification/      # MODEL 2 SUBSYSTEM
    ├── ingredient_predictor.py        # Core PyTorch inference & calculation engine
    ├── Converting Code Files/
    │   ├── build_ingredient_nutrition_dataset.py # Generates vocabulary & macro CSVs
    │   └── generate_home_hotel_data.py          # Generates Home vs Hotel recipe database
    ├── DataSet/
    │   ├── RawData/Version1/
    │   └── Version1 Used Data/
    │       ├── dataset.yaml
    │       ├── south_indian_food_ingredients_nutrition.csv # 31 foods × nutrition
    │       ├── ingredients_vocabulary.csv                 # 68 canonical ingredients
    │       ├── dishes_ingredients_map.json                # O(1) fast lookup map
    │       └── home_hotel_recipes.json                    # Dual Home/Hotel recipe weights
    ├── Models/Ingredient Identification/Version1/
    │   ├── ingredient_identification_v1_best.pt  # Trained PyTorch dual-head model
    │   ├── ingredient_identification_v1_last.pt
    │   └── best.pt
    └── NoteBooks/
        ├── Basic Setting/basic_setting.ipynb
        └── Ingredient Identification/Version1/
            └── ingredient_identification_v1_training.ipynb # End-to-end Colab notebook
```

---

## 3. 🎯 31 Supported South Indian Food Classes

Both Model 1 and Model 2 operate across identical 31 canonical food classes:

| ID | Class Name | Tamil Regional Name | Typical Category |
|:---|:---|:---|:---|
| 0 | Appam | ஆப்பம் / Palappam | Breakfast / Tiffin |
| 1 | Beetroot poriyal | பீட்ரூட் பொரியல் | Side Dish / Poriyal |
| 2 | Boiled Egg | அவித்த முட்டை | Side Dish / Protein |
| 3 | Carrot poriyal | கேரட் பொரியல் | Side Dish / Poriyal |
| 4 | Chicken 65 | சிக்கன் 65 | Starter / Appetizer |
| 5 | Chicken briyani | சிக்கன் பிரியாணி | Main Course |
| 6 | Dosa | தோசை | Breakfast / Dinner |
| 7 | Idly | இட்லி | Breakfast / Dinner |
| 8 | Kaara chutney | கார சட்னி | Condiment / Chutney |
| 9 | Kali | உளுந்தங்களி / களி | Traditional / Healthy |
| 10 | Koozh | கேழ்வரகு கூழ் | Traditional / Beverage |
| 11 | Lemon Rice | எலுமிச்சை சாதம் | Variety Rice |
| 12 | Mushroom briyani | காளான் பிரியாணி | Main Course |
| 13 | Mutton Briyani | மட்டன் பிரியாணி | Main Course |
| 14 | Nandu masala | நண்டு மசாலா | Seafood Main Course |
| 15 | Nei satham | நெய் சாதம் | Variety Rice |
| 16 | Paal kolukattai | பால் கொழுக்கட்டை | Traditional Dessert |
| 17 | Paneer briyani | பன்னீர் பிரியாணி | Main Course |
| 18 | Panner masala | பன்னீர் மசாலா | Gravy / Curry |
| 19 | Parupu vada | பருப்பு வடை / ஆமை வடை | Snack / Tiffin |
| 20 | Pidi kolukattai | பிடி கொழுக்கட்டை | Traditional Festival Food |
| 21 | Poorna kolukattai | பூரண கொழுக்கட்டை | Traditional Sweet Dumpling |
| 22 | Prawn thokku | இறால் தொக்கு | Seafood Gravy |
| 23 | Puthina Chutney | புதினா சட்னி | Condiment / Chutney |
| 24 | Sambar | சாம்பார் | Traditional Stew / Curry |
| 25 | Sambar satham | சாம்பார் சாதம் | Variety Rice / Meals |
| 26 | Satham | சாதம் (வெள்ளை சாதம்) | Main Staple Rice |
| 27 | Thengai chutney | தேங்காய் சட்னி | Condiment / Chutney |
| 28 | Uzhuntha vadai | உளுந்த வடை / மெது வடை | Breakfast / Snack |
| 29 | Veg briyani | காய்கறி பிரியாணி | Main Course |
| 30 | Ven Pongal | வெண் பொங்கல் | Breakfast / Tiffin |

---

## 4. 🖥️ Web User Interface Usability (`app.py`)

The Streamlit web application provides an intuitive interactive interface for nutritionists, consumers, and food inspection professionals:

### Tab 1: 📸 Food Plate Image Detection (Primary Workflow)
1. **Upload Food Plate Image**: Upload any camera photo (JPG, PNG, WebP) or pick from test samples.
2. **Visual Bounding Boxes**: Displays original image alongside annotated detection overlay with class names and confidence percentages.
3. **Detected Dish Cards**: Lists identified items with visual cards.
4. **Model 2 Expander per Dish**:
   - Lists constituent ingredients as green chip tags.
   - 5 Key Nutritional Metrics: **Calories (kcal), Protein (g), Carbohydrates (g), Total Fat (g), Dietary Fiber (g)**.
   - Traditional Cooking Method & Health Profile.
5. **Home vs. Hotel Analysis**:
   - **Hotel Mode**: Type overall serving weight $\rightarrow$ auto-splits ingredient weights and scales commercial nutrition.
   - **Home Mode**: Type weights for each constituent ingredient + add extra ingredients from dropdown $\rightarrow$ live dynamic recalculation!

### Tab 2: 🥗 Quick Text Test (Model 2 Direct Lookup)
- Allows typing or selecting any of the 31 dishes without needing an image.
- Instantly benchmarks Model 2 neural network predictions against ICMR ground-truth data.

---

## 5. ⚡ Quick Launch Instructions

### On Windows Local PC:
Double-click `run_ui.bat` or run:
```powershell
streamlit run app.py
```
The browser opens automatically at `http://localhost:8501`.

