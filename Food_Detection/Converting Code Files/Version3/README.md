# Food Detection Version 3 (Multi-Dish Platter, Banana Leaf & 223 Categories)

Version 3 expands Version 2 by solving **multi-dish plate detection** using real curated `assorted_meal_platter` and `banana_leaf_foods` images (no synthetic plates) and adding **Poori (221)** and **Chutney (222)**.

## Dataset Composition (`223` Classes — Real Photos Only)
1. **Version 2 Base Splits**: `11,050` single-dish images across `221` classes (`8,840 train / 1,105 val / 1,105 test`).
2. **New Classes**: `50` Poori (`221`) + `47` Chutney (`222`) images.
3. **Curated Multi-Dish Platters & Banana Leaf Meals**: Real `assorted_meal_platter` and `banana_leaf_foods` photos (`80/10/10` split).

## Pipeline Scripts
- `prepare_dataset_v3.py` — Builds `Food_Detection/DataSet/Version3 Used Data/` (`9,793 train / 1,224 val / 1,226 test`).
- `add_poori_chutney_v3.py` — Collects Poori & Chutney images and updates `dataset.yaml` + nutrition JSONs.
- `train_food_detection_v3.py` — Trains YOLOv8 on Version 3 (`223` classes).
- `inference_food_detection_v3.py` — CLI inference on food plates.
