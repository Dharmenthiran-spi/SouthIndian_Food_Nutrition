# Food Detection Version 2 - Converting Code Files

This directory contains the Python conversion and training pipeline for **Food Detection Version 2** (221 food categories).

## 📁 Directory Structure
- `train_food_detection_v2.py`: End-to-end YOLOv8 training script with auto dataset resolution, metrics tracking, loss curve plotting, and ONNX export.
- `prepare_dataset_v2.py`: Synchronizes raw clean images (11,050 photos across 221 classes) into train, val, and test splits with complete bounding box labels and `dataset.yaml`.
- `inference_food_detection_v2.py`: Standalone evaluation and inference script to test trained models on new images.

## 🚀 How to Run

### 1. Prepare Dataset Splits
```bash
python prepare_dataset_v2.py
```

### 2. Train YOLOv8 Model (Local or Google Colab)
```bash
python train_food_detection_v2.py --epochs 50 --batch 16 --imgsz 640 --device 0
```
On CPU:
```bash
python train_food_detection_v2.py --epochs 30 --batch 8 --device cpu
```

### 3. Run Inference on Test Images
```bash
python inference_food_detection_v2.py --source "Food_Detection/DataSet/Version2 Used Data/images/test" --conf 0.25
```

