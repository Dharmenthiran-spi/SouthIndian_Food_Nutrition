# 🎯 Food Detection Version 2 Used Data (YOLOv8)

This directory contains the master configuration and dataset structure for the **Version 2 YOLOv8 Food Detection Model**.

---

## 📋 Configuration Specifications
* **Dataset Config**: [`dataset.yaml`](dataset.yaml)
* **Number of Classes (`nc`)**: **229**
* **Train Directory**: `images/train/` (labels in `labels/train/`)
* **Validation Directory**: `images/val/` (labels in `labels/val/`)
* **Test Directory**: `images/test/` (labels in `labels/test/`)

---

## 🏷️ Category Coverage in Version 2:
1. **South Indian Foods** (Idli, Dosa, Pongal, Ragi Koozh, Parotta, etc.)
2. **Vegetarian Dishes** (Sambar, Rasam, Dal, Palak Paneer, Rajma, etc.)
3. **Non-Vegetarian Dishes** (Chicken Curry, Tandoori, Biryani, Fish Fry, Boiled Egg, etc.)
4. **Fast Foods & International** (Burgers, Pizzas, Shawarma, Tacos, Pasta, etc.)
5. **Fitness Smoothies & Shakes** (Banana Peanut Butter, Chocolate Protein, Green Smoothie, etc.)
6. **Fresh Fruits & Vegetables** (Apple, Banana, Avocado, Broccoli, Spinach, etc.)
7. **Nuts, Seeds & Sprouts** (Almonds, Walnuts, Chia, Pumpkin Seeds, Moong Sprouts, etc.)
8. **Gym & Sports Nutrition** (Whey Isolate, Casein, Creatine, Mass Gainers, Shakers)

To start training with Ultralytics YOLOv8:
```bash
yolo detect train data="Food_Detection/DataSet/Version2 Used Data/dataset.yaml" model=yolov8s.pt epochs=100 imgsz=640 batch=16
```
