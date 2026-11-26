# Food Detection Version 3 (Multi-Dish Thali & 221 Categories)

Version 3 expands Version 2 by specifically solving **multi-dish plate detection** (Thali sets, Banana leaf meals, South Indian breakfast combos, etc.).

## Pipeline Scripts
1. `download_multidish_v3.py` — Downloads 500 authentic multi-dish meal images into `Food_Detection/DataSet/RawData/Version3/multi_dish_images/`.
2. `prepare_dataset_v3.py` — Combines:
   - All **11,050** Version 2 single-dish images (221 classes)
   - **500** auto-annotated real multi-dish images
   - **600** synthetic multi-dish combo plates (4 dishes per plate with exact bounding boxes)
   Generates `Food_Detection/DataSet/Version3 Used Data/dataset.yaml`.
3. `train_food_detection_v3.py` — Fine-tunes YOLOv8 from `food_detection_v2_best.pt` on the Version 3 dataset.
4. `inference_food_detection_v3.py` — Runs CLI inference on food plates.

