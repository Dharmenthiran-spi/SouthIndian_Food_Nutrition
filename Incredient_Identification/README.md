# 🥗 Ingredient Identification Module (Model 2)

Welcome to the **Ingredient Identification** module of the **South Indian Food Recognition & Nutrition Analysis AI** project.

This module is dedicated to **Model 2: Food Image to Multi-label Ingredient Prediction**, training models to identify individual culinary ingredients from South Indian food images.

---

## 📁 Architectural Directory Structure

This module mirrors the exact architecture of the `Food_Detection` module:

```
Incredient_Identification/
│
├── Converting Code Files/
│   ├── README.md
│   └── Archana_Kitchen_Recipes/
│       └── convert_recipes_to_ingredient_dataset.py   # Dataset crawler, dHash deduplicator & normalizer
│
├── DataSet/
│   ├── RawData/
│   │   └── Version1/
│   │       ├── README.md
│   │       └── Archana_Kitchen_Recipes/
│   │           └── README.md                          # Provenance & license documentation
│   │
│   └── Version1 Used Data/
│       ├── README.md                                  # Dataset statistics & specs
│       ├── dataset.yaml                               # Multi-label class mappings & paths
│       ├── images/
│       │   ├── train/                                 # 815 training images
│       │   ├── valid/                                 # 174 validation images
│       │   └── test/                                  # 176 test images
│       ├── annotations/
│       │   ├── train.csv                              # image,food,ingredients
│       │   ├── valid.csv                              # image,food,ingredients
│       │   └── test.csv                               # image,food,ingredients
│       └── metadata/
│           ├── sources.csv                            # Source URLs, image URLs, licenses, hashes
│           ├── ingredients.csv                        # 98 canonical ingredients with counts & categories
│           └── dataset_report.json                    # Full JSON summary report
│
├── Models/
│   └── Version1/
│       └── README.md                                  # Checkpoint storage (`best.pt`, `last.pt`)
│
└── NoteBooks/
    ├── Basic Setting/
    │   └── basic_setting.ipynb                        # Hardware check, Colab & 2-way Git sync
    └── Version1/
        └── ingredient_identification_v1_training.ipynb# Model 2 multi-label training pipeline
```

---

## 📊 Dataset Overview (Version 1)

- **Total Images**: 1,165 verified food photographs
- **Vocabulary**: 98 canonical South Indian ingredients (e.g. `toor_dal`, `urad_dal`, `curry_leaves`, `mustard_seeds`, `tamarind`, `asafoetida`, `coconut`, `ghee`)
- **Format**: Multi-label CSV (`image,food,ingredients`)
- **Split**: 70% Train (815) / 15% Valid (174) / 15% Test (176)
- **License**: CC BY-SA 4.0

---

## 🚀 Quick Start

### 1. Ingest & Regenerate Dataset
```bash
python "Incredient_Identification/Converting Code Files/Archana_Kitchen_Recipes/convert_recipes_to_ingredient_dataset.py" --workers 12
```

### 2. Colab Setup & Synchronization
Open `Incredient_Identification/NoteBooks/Basic Setting/basic_setting.ipynb` in Google Colab to mount Google Drive and synchronize with GitHub.

### 3. Model Training
Open `Incredient_Identification/NoteBooks/Version1/ingredient_identification_v1_training.ipynb` to inspect data loaders, model architecture, and configure training runs.

