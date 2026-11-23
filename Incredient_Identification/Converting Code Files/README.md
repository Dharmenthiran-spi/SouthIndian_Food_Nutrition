# 🔄 Ingredient Identification - Dataset Converting Code Files

This directory contains the pipeline and converter scripts for **Model 2: Food Image to Ingredient Prediction (Multi-label Classification)**.

Each script ingests verified raw culinary recipes and image links from `Incredient_Identification/DataSet/RawData/Version1`, validates image integrity, removes perceptual duplicates via difference hashing (dHash), normalizes ingredient labels into standardized snake_case terms, and generates multi-label annotations for `Incredient_Identification/DataSet/Version1 Used Data`.

---

## 📁 Directory Structure

```
Converting Code Files/
├── README.md
│
└── Archana_Kitchen_Recipes/
    └── convert_recipes_to_ingredient_dataset.py   # Ingests, verifies, and builds multi-label dataset
```

---

## ⚙️ How to Run

To regenerate or update the dataset:
```bash
python "Incredient_Identification/Converting Code Files/Archana_Kitchen_Recipes/convert_recipes_to_ingredient_dataset.py" --workers 12
```

Arguments:
- `--workers`: Number of concurrent download and verification threads (default: 8)
- `--max_recipes`: Limit number of recipes to process for quick testing (optional)

