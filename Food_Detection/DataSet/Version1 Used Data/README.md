# 🍽️ Version 1 Food Detection Dataset (YOLO Format)

This dataset is the official **Version 1 training-ready food detection dataset** for YOLOv8 / YOLO11 object detection models.

---

## 📊 Dataset Statistics

- **Total Annotated Images**: 8,010 images
- **Total Bounding Boxes**: 8,054 object instances
- **Classes**: 31 South Indian food categories
- **Train Split**: 6,403 images (6,437 boxes)
- **Validation Split**: 812 images (815 boxes)
- **Test Split**: 795 images (802 boxes)
- **Annotation Format**: Ultralytics YOLO (`class_id x_center y_center width height` normalized)

---

## 📁 Directory Structure

```
Version1 Used Data/
├── images/
│   ├── train/  # 6,403 images
│   ├── val/    # 812 images
│   └── test/   # 795 images
├── labels/
│   ├── train/  # 6,403 label txt files
│   ├── val/    # 812 label txt files
│   └── test/   # 795 label txt files
├── dataset.yaml
└── README.md
```

---

## 🏷️ Class List (31 Classes)

| ID | Class Name | ID | Class Name | ID | Class Name | ID | Class Name |
|---|---|---|---|---|---|---|---|
| 0 | Appam | 8 | Kaara chutney | 16 | Paal kolukattai | 24 | Sambar |
| 1 | Beetroot poriyal | 9 | Kali | 17 | Paneer briyani | 25 | Sambar satham |
| 2 | Boiled Egg | 10 | Koozh | 18 | Panner masala | 26 | Satham |
| 3 | Carrot poriyal | 11 | Lemon Rice | 19 | Parupu vada | 27 | Thengai chutney |
| 4 | Chicken 65 | 12 | Mushroom briyani | 20 | Pidi kolukattai | 28 | Uzhuntha vadai |
| 5 | Chicken briyani | 13 | Mutton Briyani | 21 | Poorna kolukattai | 29 | Veg briyani |
| 6 | Dosa | 14 | Nandu masala | 22 | Prawn thokku | 30 | Ven Pongal |
| 7 | Idly | 15 | Nei satham | 23 | Puthina Chutney | | |

---

## 🚀 How to Train

### Python API

```python
from ultralytics import YOLO

# 1. Load pretrained YOLO object detection model
model = YOLO('yolov8s.pt')  # or 'yolo11s.pt', 'yolov8n.pt'

# 2. Train on Version 1 Used Data
results = model.train(
    data=r'D:\Yolo_Food_Inspection\Food_Detection\DataSet\Version1 Used Data\dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # or 'cpu'
)

# 3. Evaluate on test set
metrics = model.val(split='test')
```

### CLI Command

```bash
yolo detect train data="D:/Yolo_Food_Inspection/Food_Detection/DataSet/Version1 Used Data/dataset.yaml" model=yolov8s.pt epochs=100 imgsz=640 batch=16
```

