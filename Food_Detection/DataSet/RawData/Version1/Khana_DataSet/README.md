# 🍛 Food Classification Dataset (YOLO Format)

This dataset is organized in standard **Ultralytics YOLO Classification** format with **100% English class names and file names**.

---

## 📊 Dataset Statistics

- **Total Classes**: 80 Food Classes (English)
- **Total Images**: 131,819 verified images
- **Train Split**: 105,423 images (80%)
- **Validation Split**: 26,396 images (20%)
- **Format**: Ultralytics YOLO Classification (`train/` and `val/`)
- **Naming Pattern**: `{class_name}_{split}_{index:05d}.jpg` (e.g. `potato_cauliflower_train_00001.jpg`)

---

## 📁 Directory Structure

```
Khana_DataSet/
├── train/
│   ├── potato_cauliflower/
│   │   ├── potato_cauliflower_train_00001.jpg
│   │   ├── potato_cauliflower_train_00002.jpg
│   │   └── ...
│   ├── potato_fenugreek/
│   └── ... (80 class folders)
├── val/
│   ├── potato_cauliflower/
│   │   ├── potato_cauliflower_val_00001.jpg
│   │   └── ...
│   └── ... (80 class folders)
├── dataset.yaml
├── labels.txt
└── README.md
```

---

## 🚀 Training with Ultralytics YOLO

### Python API

```python
from ultralytics import YOLO

# Load a pretrained YOLO classification model
model = YOLO('yolov8n-cls.pt')

# Train
results = model.train(
    data=r'D:\Yolo_Food_Inspection\DataSet\Khana_DataSet',
    epochs=50,
    imgsz=224,
    batch=64,
    device=0  # or 'cpu'
)

# Validate
metrics = model.val()
```

### CLI Command

```bash
yolo classify train data="D:/Yolo_Food_Inspection/DataSet/Khana_DataSet" model=yolov8n-cls.pt epochs=50 imgsz=224 batch=64
```

---

## 🏷️ Class List (80 English Classes)

| ID | Class Name | ID | Class Name | ID | Class Name | ID | Class Name |
|---|---|---|---|---|---|---|---|
| 0 | potato_cauliflower | 20 | spiced_potato_burger | 40 | margherita_pizza | 60 | vegetable_curry_with_bread_roll |
| 1 | potato_fenugreek | 21 | lentil_rice_porridge | 41 | spiced_potato_crepe | 61 | pepperoni_pizza |
| 2 | potato_peas | 22 | steamed_chickpea_cake | 42 | spiced_lentil_cracker | 62 | ground_rice_pudding |
| 3 | potato_stuffed_flatbread | 23 | sweet_vermicelli_dessert | 43 | fried_lentil_donut | 63 | flattened_rice_stir_fry |
| 4 | stuffed_leavened_flatbread | 24 | fish_curry | 44 | spicy_sprouts_with_bread_roll | 64 | savory_rice_lentil_porridge |
| 5 | egg_curry | 25 | carrot_pudding | 45 | sweet_rice_dumpling | 65 | puffed_bread_with_potato_curry |
| 6 | glazed_flaky_doughnut | 26 | garlic_bread | 46 | yellow_lentil_pudding | 66 | kidney_bean_curry_with_rice |
| 7 | banana_chips | 27 | garlic_flatbread | 47 | crispy_spiral_snack | 67 | cottage_cheese_balls_in_syrup |
| 8 | chickpea_flour_sweet_ball | 28 | honeycomb_sweet_cake | 48 | gram_flour_ghee_fudge | 68 | semolina_crepe |
| 9 | spiced_okra | 29 | grilled_sandwich | 49 | mixed_vegetable_nut_curry | 69 | tapioca_pearl_pilaf |
| 10 | spiced_rice | 30 | sweet_fried_dumpling | 50 | delicate_rice_crepe | 70 | tapioca_pearl_fritters |
| 11 | sweet_gram_pearl_ball | 31 | sweet_milk_balls_in_syrup | 51 | onion_fritters | 71 | savory_stuffed_pastry |
| 12 | buttermilk | 32 | spinach_pea_cutlet | 52 | spinach_cottage_cheese | 72 | grilled_minced_skewers |
| 13 | chickpea_curry | 33 | steamed_rice_noodles | 53 | cottage_cheese_curry | 73 | fluffy_rice_crepe |
| 14 | wheat_flatbread | 34 | steamed_rice_cake | 54 | cottage_cheese_pizza | 74 | crisp_wafer_potato_snack |
| 15 | chicken_pizza | 35 | crispy_sweet_spirals | 55 | crisp_water_balls | 75 | coconut_kokum_drink |
| 16 | chicken_wings | 36 | cashew_fudge | 56 | steamed_rice_dumpling_balls | 76 | steamed_dumplings |
| 17 | peanut_brittle | 37 | crispy_thin_flatbread | 57 | crisp_wafer_yogurt_snack | 77 | assorted_meal_platter |
| 18 | savory_flattened_rice_mix | 38 | sweet_rice_pudding | 58 | taro_leaf_rolls | 78 | noodle_soup |
| 19 | chickpeas_with_fried_bread | 39 | condensed_milk_ice_cream | 59 | vegetable_curry_with_bread_roll | 79 | potato_fritter_in_bun |
