# 🔄 Food Detection - Dataset Converting Code Files

This directory contains the dataset converter scripts organized by dataset name. Each script transforms raw datasets from `Food_Detection/DataSet/RawData/Version1` into ready-to-train YOLO datasets.

---

## 📁 Directory Structure

```
Converting Code Files/
├── README.md
│
├── SouthIndian_Food/
│   └── convert_voc_to_yolo.py       # Converts Pascal VOC XML to YOLO Object Detection -> Version1 Used Data
│
├── Khana_DataSet/
│   ├── convert_to_yolo_detect.py    # Converts classification images to YOLO detection bounding boxes
│   └── convert_khana_to_yolo_cls.py # Reference pipeline for raw -> YOLO classification format
│
└── Huggig_Real_Image_Dataset/
    └── prepare_huggig_splits.py     # Splits flat 377 images & labels into train/val
```

---

## 🚀 Converter Usage & Outputs

### 1. South Indian Food (Main Version 1 Food Detection Dataset)
* **Script**: `Food_Detection/Converting Code Files/SouthIndian_Food/convert_voc_to_yolo.py`
* **Input**: `Food_Detection/DataSet/RawData/Version1/SouthIndian_Food` (8,010 images + XML)
* **Output**: `Food_Detection/DataSet/Version1 Used Data` (8,010 images, 31 classes, train/val/test splits, `dataset.yaml`)
* **Run**:
  ```bash
  python "D:\Yolo_Food_Inspection\Food_Detection\Converting Code Files\SouthIndian_Food\convert_voc_to_yolo.py"
  ```

### 2. Khana Dataset (Single-Dish Images to YOLO Detection)
* **Script**: `Food_Detection/Converting Code Files/Khana_DataSet/convert_to_yolo_detect.py`
* **Input**: `Food_Detection/DataSet/RawData/Version1/Khana_DataSet`
* **Output**: `Food_Detection/DataSet/Version1 Used Data/Khana_YOLO_Detect`
* **Run**:
  ```bash
  python "D:\Yolo_Food_Inspection\Food_Detection\Converting Code Files\Khana_DataSet\convert_to_yolo_detect.py"
  ```

### 3. Huggig Multi-Dish Dataset (Train/Val Split)
* **Script**: `Food_Detection/Converting Code Files/Huggig_Real_Image_Dataset/prepare_huggig_splits.py`
* **Input**: `Food_Detection/DataSet/RawData/Version1/Huggig_Real_Image_Dataset`
* **Output**: `Food_Detection/DataSet/Version1 Used Data/Huggig_YOLO`
* **Run**:
  ```bash
  python "D:\Yolo_Food_Inspection\Food_Detection\Converting Code Files\Huggig_Real_Image_Dataset\prepare_huggig_splits.py"
  ```
