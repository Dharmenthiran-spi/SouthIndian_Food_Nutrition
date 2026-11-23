# 🍽️ South Indian Food Nutrition & Detection

An end-to-end computer vision and deep learning project for **South Indian Food Detection**, **Nutritional Estimation**, and **Ingredient Identification** using Ultralytics YOLO models.

---

## 📁 Repository Structure

```
.
├── Food_Detection/
│   ├── Converting Code Files/
│   │   ├── README.md
│   │   ├── SouthIndian_Food/
│   │   │   └── convert_voc_to_yolo.py       # Converts Pascal VOC XML to YOLO format -> Version1 Used Data
│   │   ├── Khana_DataSet/
│   │   │   ├── convert_to_yolo_detect.py    # Converts classification images to YOLO detection bounding boxes
│   │   │   └── convert_khana_to_yolo_cls.py # Pipeline reference for raw -> YOLO classification format
│   │   └── Huggig_Real_Image_Dataset/
│   │       └── prepare_huggig_splits.py     # Splits multi-dish images & labels into train/val
│   │
│   └── DataSet/
│       ├── RawData/
│       │   └── Version1/
│       │       ├── SouthIndian_Food/         # 8,010 images with Pascal VOC XML annotations (31 classes)
│       │       ├── Khana_DataSet/            # 131,819 images across 80 Indian food categories
│       │       └── Huggig_Real_Image_Dataset/# 377 multi-dish annotated food plates
│       └── Version1 Used Data/               # Ready-to-train YOLO Object Detection dataset (8,010 images)
│
├── Incredient_Identification/                # Ingredient identification module
├── .gitignore
└── README.md
```

---

## 🎯 Version 1 Food Detection Model

The primary dataset for training the Version 1 model is prepared in **`Food_Detection/DataSet/Version1 Used Data`**:

- **Total Images**: 8,010 images
- **Bounding Boxes**: 8,054 instances
- **Classes**: 31 South Indian food categories
  * *Appam, Beetroot poriyal, Boiled Egg, Carrot poriyal, Chicken 65, Chicken briyani, Dosa, Idly, Kaara chutney, Kali, Koozh, Lemon Rice, Mushroom briyani, Mutton Briyani, Nandu masala, Nei satham, Paal kolukattai, Paneer briyani, Panner masala, Parupu vada, Pidi kolukattai, Poorna kolukattai, Prawn thokku, Puthina Chutney, Sambar, Sambar satham, Satham, Thengai chutney, Uzhuntha vadai, Veg briyani, Ven Pongal*
- **Splits**:
  * **Train**: 6,403 images
  * **Val**: 812 images
  * **Test**: 795 images

---

## 🚀 Quick Start

### 1. Requirements

```bash
pip install ultralytics torch torchvision pyyaml
```

### 2. Dataset Conversion (if rebuilding from raw)

```bash
python "Food_Detection/Converting Code Files/SouthIndian_Food/convert_voc_to_yolo.py"
```

### 3. Training YOLOv8 / YOLO11

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8s.pt')

# Train
results = model.train(
    data='Food_Detection/DataSet/Version1 Used Data/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)

# Test evaluation
metrics = model.val(split='test')
```

Or via CLI:

```bash
yolo detect train data="Food_Detection/DataSet/Version1 Used Data/dataset.yaml" model=yolov8s.pt epochs=100 imgsz=640 batch=16
```

