# 📸 Food Detection Version 2 – Image Dataset Sources & Training Guide

This document specifies the open-source image repositories, computer vision benchmarks, and data preparation pipelines for training the **Version 2 YOLOv8 Model (229 Classes)**.

---

## 🎯 Target Scope
* **Total Food Classes**: **229 Classes**
* **Cuisines**: South Indian Traditional, North Indian Curries, International Fast Foods, Fresh Fruits, Vegetables, Nuts, Seeds, Fitness Smoothies, and Sports Nutrition Products.
* **Target Architecture**: Ultralytics YOLOv8s / YOLOv8m Object Detection.

---

## 🌐 Primary Image Datasets & Repositories

### 1. 🍛 Khana Indian Food Dataset
* **URL**: [https://github.com/KhanaDataset/Khana](https://github.com/KhanaDataset/Khana)
* **Scope**: 131,000+ images of Indian dishes (Idli, Dosa, Pongal, Biryani, Curries, Poriyal, Sambar, Rasam).
* **Usage in V2**: Used as the primary training pool for South Indian and traditional Indian categories (Classes 0 to 45).

### 2. 🍔 Food-101 (ETH Zurich / Vision Lab)
* **URL**: [https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
* **Kaggle Link**: [https://www.kaggle.com/datasets/dansbecker/food-101](https://www.kaggle.com/datasets/dansbecker/food-101)
* **Scope**: 101,000 images across 101 international food categories (Burgers, Pizzas, French Fries, Tacos, Macaroni, Omelette, Club Sandwich, Ice Cream, Cake).
* **Usage in V2**: Used for International Fast Food categories (Classes 46 to 75).

### 3. 🍎 Fruits-360 Dataset
* **Kaggle Link**: [https://www.kaggle.com/datasets/moltean/fruits](https://www.kaggle.com/datasets/moltean/fruits)
* **Scope**: 90,000+ high-quality images of whole fresh fruits, vegetables, and nuts against neutral and complex backgrounds.
* **Usage in V2**: Used for whole raw fruits, vegetables, and tree nuts (Classes 76 to 140).

### 4. 🏋️‍♂️ Roboflow Sports Nutrition & Bottle Detection
* **URL**: [https://universe.roboflow.com/search?q=protein+powder](https://universe.roboflow.com/search?q=protein+powder)
* **Scope**: Pre-annotated bounding boxes for protein powder tubs (Optimum Nutrition, MuscleBlaze), shaker bottles, and protein bars.
* **Usage in V2**: Used for Gym & Sports Nutrition items (Classes 218 to 228).

---

## 🚀 Automated Downloader Script
Use the accompanying script `download_v2_food_detection_data.py` to automate downloading and formatting annotations into standard YOLO format:

```bash
python Food_Detection/DataSet/RawData/Version2/download_v2_food_detection_data.py --subset sample
```
