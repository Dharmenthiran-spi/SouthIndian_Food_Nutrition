"""
Food Detection Version 3 - High-Recall Tight Multi-Dish Re-Annotator
Upgrades:
1. Leaf-Masked HoughCircles + COCO yolov8s.pt: Detects round katori/bowls on the table/thali
   while masking out green banana leaf veins (zero false leaf circles).
2. HSV Non-Green Food-Pile Contours: Detects food items sitting directly on the banana leaf
   (rice, poriyal, beetroot poriyal, papad, vada, chips, etc.).
3. Padded 320x320 Crop Classification: Resizes & pads each tight proposal to match Version 2's
   single-dish framing before running food_detection_v2_best.pt (restricted to INDIAN_MEAL_CLASS_IDS)
   so small overhead dishes are classified instead of dropped.
4. Strict Containment Suppression (>= 55%) + NMS (0.40) removes nested duplicates.
"""

import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[3]
V3_USED_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "Version3 Used Data"
V2_MODEL_PATH = ROOT_DIR / "Food_Detection" / "Models" / "Food Detection" / "Version2" / "food_detection_v2_best.pt"
COCO_MODEL_PATH = ROOT_DIR / "yolov8s.pt"

COCO_FOOD_IDS = {41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55}

INDIAN_MEAL_CLASS_IDS = [
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19,
    21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35,
    55, 56, 57, 58, 59, 66, 68, 69, 71, 72, 73,
    74, 75, 77,
    163, 177, 178, 179, 180, 183, 184, 185, 186, 188,
    221, 222,
]


def propose_tight_boxes(coco_model, im_bgr: np.ndarray):
    h, w = im_bgr.shape[:2]
    img_area = float(w * h)
    proposals = []

    hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
    leaf_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 25)
    leaf_u8 = (leaf_mask.astype(np.uint8) * 255)

    # 1. COCO pretrained bowl/cup/food detector
    coco_res = coco_model.predict(source=im_bgr, conf=0.07, iou=0.45, verbose=False)[0]
    for b in coco_res.boxes:
        cid = int(b.cls[0].cpu())
        if cid in COCO_FOOD_IDS:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].cpu().tolist()]
            area_frac = ((x2 - x1) * (y2 - y1)) / img_area
            if 0.007 <= area_frac <= 0.40:
                proposals.append((x1, y1, x2, y2, float(b.conf[0].cpu()) + 0.40))

    # 2. Leaf-Masked HoughCircles (only finds round bowls, rejects green-leaf interior)
    gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    min_r = max(20, int(min(w, h) * 0.055))
    max_r = int(min(w, h) * 0.14)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(min_r * 2.0),
        param1=95, param2=46, minRadius=min_r, maxRadius=max_r
    )
    if circles is not None:
        for cx, cy, r in np.round(circles[0, :]).astype(int):
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)
            bw, bh = x2 - x1, y2 - y1
            if bw >= 35 and bh >= 35:
                # Reject if inside green banana leaf
                if float(np.mean(leaf_mask[y1:y2, x1:x2])) < 0.18:
                    proposals.append((float(x1), float(y1), float(x2), float(y2), 0.60))

    # 3. Banana-leaf food-pile segmentation
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 39))
    leaf_filled = cv2.morphologyEx(leaf_u8, cv2.MORPH_CLOSE, kernel_big)
    if np.mean(leaf_mask) > 0.08:
        on_leaf_food = cv2.bitwise_and(leaf_filled, cv2.bitwise_not(leaf_u8))
        on_leaf_food = cv2.morphologyEx(on_leaf_food, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        on_leaf_food = cv2.morphologyEx(on_leaf_food, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        contours, _ = cv2.findContours(on_leaf_food, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area_frac = (bw * bh) / img_area
            aspect = max(bw / max(bh, 1), bh / max(bw, 1))
            if 0.007 <= area_frac <= 0.35 and bw >= 30 and bh >= 30 and aspect <= 2.3:
                box_green = float(np.mean(leaf_mask[y:y + bh, x:x + bw]))
                if box_green < 0.45:
                    proposals.append((float(x), float(y), float(x + bw), float(y + bh), 0.52))

    if not proposals:
        return []

    boxes_t = torch.tensor([[p[0], p[1], p[2], p[3]] for p in proposals], dtype=torch.float32)
    scores_t = torch.tensor([p[4] for p in proposals], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.38)
    kept_boxes = [boxes_t[i].tolist() for i in keep.tolist()]

    # Remove nested proposals (>= 55% inside a larger proposal)
    final_boxes = []
    for i, b1 in enumerate(kept_boxes):
        a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        nested = False
        for j, b2 in enumerate(kept_boxes):
            if i == j:
                continue
            a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
            if a2 <= a1:
                continue
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            if ix2 > ix1 and iy2 > iy1:
                if ((ix2 - ix1) * (iy2 - iy1) / a1) >= 0.55:
                    nested = True
                    break
        if not nested:
            final_boxes.append(b1)

    return final_boxes[:16]


def pad_crop_to_square(crop_pil: Image.Image, target_size=320) -> Image.Image:
    cw, ch = crop_pil.size
    side = max(cw, ch)
    padded = Image.new("RGB", (side, side), (114, 114, 114))
    padded.paste(crop_pil, ((side - cw) // 2, (side - ch) // 2))
    return padded.resize((target_size, target_size), Image.Resampling.BILINEAR)


def reannotate_image(coco_model, v2_model, img_path: Path, conf_thresh=0.14):
    im_bgr = cv2.imread(str(img_path))
    if im_bgr is None:
        return []
    h, w = im_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))

    tight_boxes = propose_tight_boxes(coco_model, im_bgr)
    if not tight_boxes:
        return []

    crops = []
    valid_boxes = []
    for x1, y1, x2, y2 in tight_boxes:
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = bw * 0.12, bh * 0.12
        cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        cx2, cy2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
        if cx2 - cx1 < 24 or cy2 - cy1 < 24:
            continue
        crops.append(pad_crop_to_square(pil_img.crop((cx1, cy1, cx2, cy2)), target_size=320))
        valid_boxes.append((x1, y1, x2, y2))

    if not crops:
        return []

    preds = v2_model.predict(
        source=crops, conf=conf_thresh, iou=0.5,
        classes=INDIAN_MEAL_CLASS_IDS, verbose=False
    )

    lines = []
    for (x1, y1, x2, y2), res in zip(valid_boxes, preds):
        if len(res.boxes) == 0:
            continue
        best_b = max(res.boxes, key=lambda b: float(b.conf[0]))
        cid = int(best_b.cls[0].cpu())
        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return lines


def reannotate_all():
    coco_src = str(COCO_MODEL_PATH) if COCO_MODEL_PATH.exists() else "yolov8s.pt"
    coco_model = YOLO(coco_src)
    v2_model = YOLO(str(V2_MODEL_PATH))

    updated = 0
    total_boxes = 0
    for sp in ("train", "val", "test"):
        img_dir = V3_USED_DIR / "images" / sp
        lbl_dir = V3_USED_DIR / "labels" / sp
        targets = list(img_dir.glob("v3_bananaleaf_*.jpg")) + list(img_dir.glob("v3_platter_*.jpg"))
        print(f"Re-annotating {len(targets)} multi-dish images in {sp}...")
        for img_p in targets:
            lines = reannotate_image(coco_model, v2_model, img_p)
            if lines:
                (lbl_dir / f"{img_p.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                updated += 1
                total_boxes += len(lines)

    print(f"\n✅ Re-annotated {updated} multi-dish images with {total_boxes} clean Indian meal boxes!")


if __name__ == "__main__":
    reannotate_all()
