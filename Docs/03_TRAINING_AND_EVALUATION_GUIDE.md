# 🎓 Model Training, Optimization & Evaluation Guide

This guide documents the complete end-to-end training methodology, mathematical formulations, loss functions, hyperparameter choices, and evaluation metrics for both **Model 1 (Food Detection)** and **Model 2 (Ingredient Identification & Nutrition)**.

---

## 1. 🍛 Model 1 Training Pipeline: Food Plate Detection (YOLOv8)

### Notebook Location:
- [food_detection_v1_training.ipynb](file:///D:/Yolo_Food_Inspection/Food_Detection/NoteBooks/Food%20Detection/Version1/food_detection_v1_training.ipynb)
- Open in Google Colab directly via the badge at the top of the notebook.

### 1.1 Dataset Specification
- **Total Annotated Images**: 8,010 images of real-world South Indian food dishes.
- **Split Distribution**:
  - `Train` split: 6,403 images (80%)
  - `Validation` split: 812 images (10%)
  - `Test` split: 795 images (10%)
- **Classes**: 31 canonical classes defined in `Food_Detection/DataSet/Version1 Used Data/dataset.yaml`.
- **Annotation Format**: YOLO bounding box format:
  $$\text{class\_id} \quad x_{\text{center}} \quad y_{\text{center}} \quad \text{width} \quad \text{height}$$
  *(Normalized coordinates between 0.0 and 1.0).*

### 1.2 Training Hyperparameters
| Parameter | Value | Rationale |
|:---|:---|:---|
| **Base Architecture** | `yolov8s.pt` | Balances mAP accuracy with fast inference (~4.4ms on GPU, 100+ FPS on edge). |
| **Input Resolution (`imgsz`)** | `640 × 640` | Preserves fine texture detail of South Indian gravies, rice grains, and garnishes. |
| **Batch Size (`batch`)** | `16` | Optimal GPU memory utilization without gradient instability. |
| **Epochs (`epochs`)** | `100` | Sufficient iterations for convergence from COCO pretrained weights. |
| **Patience (`patience`)** | `20` | Early stopping halts training if validation mAP does not improve for 20 epochs. |
| **Optimizer** | `AdamW` (Auto) | Weight decay prevents overfitting on repeated plate backgrounds. |
| **Save Period** | `10` | Milestone checkpoints saved every 10 epochs (`epoch10.pt`, `epoch20.pt`, etc.). |

### 1.3 Loss Functions in YOLOv8
Training minimizes a composite multi-task objective:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{VFL}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$
- **$\mathcal{L}_{\text{CIoU}}$ (Complete IoU Loss)**: Enforces overlap, aspect ratio, and center distance alignment of bounding boxes.
- **$\mathcal{L}_{\text{VFL}}$ (Varifocal Loss)**: Asymmetric classification loss prioritizing high-quality positive detections.
- **$\mathcal{L}_{\text{DFL}}$ (Distribution Focal Loss)**: Fine-grained regressor modeling bounding box boundaries as probability distributions.

### 1.4 Evaluation Metrics
- **$\text{mAP}@0.50$**: Mean Average Precision at IoU threshold of 0.50.
- **$\text{mAP}@0.50:0.95$**: Strict metric averaging across IoU thresholds from 0.50 to 0.95 in 0.05 steps.
- **Precision ($P$)**: Ratio of true positive detections over total detected bounding boxes:
  $$P = \frac{TP}{TP + FP}$$
- **Recall ($R$)**: Ratio of true positive detections over ground truth annotations:
  $$R = \frac{TP}{TP + FN}$$
- **Confusion Matrix**: 31 × 31 matrix detailing cross-class confusion (e.g., distinguishing between Idly and Kolukattai).

---

## 2. 🌿 Model 2 Training Pipeline: Ingredient & Nutrition Multi-Task NN

### Notebook Location:
- [ingredient_identification_v1_training.ipynb](file:///D:/Yolo_Food_Inspection/Incredient_Identification/NoteBooks/Ingredient%20Identification/Version1/ingredient_identification_v1_training.ipynb)

### 2.1 Neural Network Architecture
Model 2 is a custom PyTorch Multi-Task Deep Neural Network designed for simultaneous multi-label classification (ingredients) and continuous multi-target regression (nutrition):

```
                        Input Food ID (0 to 30)
                                  │
                                  ▼
                     Embedding Layer (31 ➔ 64)
                                  │
                                  ▼
                       Shared MLP Trunk (128)
                     [Linear ➔ BatchNorm ➔ ReLU ➔ Dropout(0.2)]
                     [Linear ➔ BatchNorm ➔ ReLU ➔ Dropout(0.2)]
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      Ingredient Output Head            Nutritional Output Head
      Linear(128 ➔ 96) ➔ ReLU           Linear(128 ➔ 64) ➔ ReLU
      Linear(96 ➔ 68)                   Linear(64 ➔ 5)
                 │                                 │
                 ▼                                 ▼
       Sigmoid Probabilities             Denormalized & Clamped
      (68 Canonical Ingredients)          (Cal, Prot, Carb, Fat, Fib)
```

### 2.2 Mathematical Formulation & Multi-Task Loss
Because ingredient prediction is multi-label binary classification and nutrition prediction is continuous regression, the training loss combines Binary Cross-Entropy with Mean Squared Error:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{ing}}, y_{\text{ing}}) + \lambda \cdot \mathcal{L}_{\text{MSE}}(\hat{y}_{\text{nut}}, y_{\text{nut}})$$

Where:
- **$\mathcal{L}_{\text{BCE}}$**: Evaluated with `nn.BCEWithLogitsLoss()` across all 68 ingredient vocabulary logits.
- **$\mathcal{L}_{\text{MSE}}$**: Mean squared error between normalized predicted nutrients and target ground truth.
- **$\lambda$ ($= 2.0$)**: Loss weighting factor balancing the regression head with the classification head.

### 2.3 Ground-Truth Normalization & Physical Constraint
1. **Z-Score Normalization during Training**:
   Nutritional values vary by orders of magnitude (Calories ~100–700 kcal, Protein ~2–40g). Target nutrients are standardized during training:
   $$y_{\text{norm}} = \frac{y - \mu}{\sigma}$$
   Where $\mu$ and $\sigma$ are computed from the ICMR-NIN ground-truth dataset.
2. **Denormalization during Inference**:
   $$\hat{y} = (\hat{y}_{\text{norm}} \times \sigma) + \mu$$
3. **Physical Reality Constraint (Non-Negative Clamping)**:
   Linear regression near zero can slightly undershoot (e.g. producing $-0.2$g fat for fat-free steamed Idly). Model 2 enforces physical validity:
   $$\hat{y}_{\text{final}} = \max(0.0, \hat{y})$$

---

## 3. 🔄 Google Colab & GitHub Workflow

Both Model 1 and Model 2 notebooks adhere to the following workflow:

```
  [GitHub Repository]
          │
          │ 1. Click 'Open in Colab' Badge
          ▼
   [Google Colab]
   • Freshly clones repo onto local fast SSD (/content/)
   • Trains on NVIDIA T4/A100 GPU
   • Produces results.png, confusion_matrix.png, best.pt, last.pt
          │
          │ 2. Step 10: 1-Click Browser Download (files.download)
          ▼
   [Local Computer Downloads Folder]
   • food_detection_v1_best.pt (~22.5 MB)
   • ingredient_identification_v1_best.pt (~1.2 MB)
          │
          │ 3. Move into official project Models folder
          ▼
   [Local Streamlit Web Application (app.py)]
```

### Resuming Training from `last.pt`:
If a Google Colab session disconnects or reaches maximum runtime limits, run the resume cell in either notebook:
```python
from ultralytics import YOLO
resume_model = YOLO("Food_Detection/Models/Food Detection/Version1/last.pt")
resume_model.train(resume=True)
```
Training seamlessly picks up from the exact last saved epoch with all optimizer states preserved.

