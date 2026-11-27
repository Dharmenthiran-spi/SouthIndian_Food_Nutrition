"""
Food Detection Version 3 - Offline ResNet50 Visual Prototype + YOLO Annotator (No API Key)
1. Extracts 2048-dim L2-normalized ResNet50 embeddings from verified Version 2/3
   single-dish training images across Indian Meal / Banana Leaf / Thali classes.
2. Proposes tight dish & bowl boxes using:
   - COCO yolov8s.pt (bowls, cups, food)
   - Leaf-masked & flower-filtered HoughCircles (catches metal/ceramic katori bowls)
   - HSV non-green leaf food-pile segmentation (catches rice, poriyal, avial, chutney, banana)
3. Classifies every tight crop using ResNet50 Prototype Cosine Similarity + YOLO V2 boost.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
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
    13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40,
    55, 56, 57, 58, 59, 66, 67, 68, 69, 71, 72, 73,
    74, 75, 77, 117, 163, 177, 178, 179, 180, 183, 184, 185, 186, 188,
    221, 222,
]

PRESERVED_STEMS = {"v3_bananaleaf_0318", "v3_bananaleaf_0241", "v3_platter_0314"}


def build_resnet_extractor():
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    base = models.resnet50(weights=weights)
    extractor = nn.Sequential(*list(base.children())[:-1]).eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return extractor, transform


def build_prototype_bank(extractor, transform, samples_per_class=12):
    print("🧠 Building ResNet50 visual prototype bank from verified single-dish training images...")
    img_dir = V3_USED_DIR / "images" / "train"
    lbl_dir = V3_USED_DIR / "labels" / "train"
    allowed_set = set(INDIAN_MEAL_CLASS_IDS)

    class_files = {cid: [] for cid in INDIAN_MEAL_CLASS_IDS}
    for im_p in sorted(img_dir.glob("*.jpg")):
        if im_p.name.startswith(("v3_bananaleaf_", "v3_platter_", "v3_synth_")):
            continue
        lbl_p = lbl_dir / f"{im_p.stem}.txt"
        if not lbl_p.exists():
            continue
        first = lbl_p.read_text(encoding="utf-8").strip().splitlines()
        if not first:
            continue
        cid = int(first[0].split()[0])
        if cid in allowed_set and len(class_files[cid]) < samples_per_class:
            class_files[cid].append(im_p)

    proto_cids = []
    proto_vecs = []
    with torch.no_grad():
        for cid, files in class_files.items():
            if not files:
                continue
            tensors = []
            for fp in files:
                try:
                    with Image.open(fp) as im:
                        tensors.append(transform(im.convert("RGB")))
                except Exception:
                    continue
            if not tensors:
                continue
            batch = torch.stack(tensors)
            feats = extractor(batch).squeeze(-1).squeeze(-1)
            feats = feats / feats.norm(dim=1, keepdim=True).clamp(min=1e-6)
            centroid = feats.mean(dim=0)
            centroid = centroid / centroid.norm().clamp(min=1e-6)
            proto_cids.append(cid)
            proto_vecs.append(centroid)

    proto_matrix = torch.stack(proto_vecs)  # (C, 2048)
    print(f"✅ Built prototype bank for {len(proto_cids)} Indian meal classes!")
    return proto_cids, proto_matrix


def is_flower_corner(im_hsv: np.ndarray, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> bool:
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    if (xc < 0.15 or xc > 0.85) and (yc < 0.15 or yc > 0.85):
        roi = im_hsv[y1:y2, x1:x2]
        if roi.size > 0:
            flower_mask = (roi[:, :, 0] >= 8) & (roi[:, :, 0] <= 32) & (roi[:, :, 1] > 155) & (roi[:, :, 2] > 155)
            if float(np.mean(flower_mask)) > 0.30:
                return True
    return False


def propose_tight_boxes(coco_model, im_bgr: np.ndarray):
    h, w = im_bgr.shape[:2]
    img_area = float(w * h)
    proposals = []

    hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
    leaf_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 25)
    leaf_u8 = (leaf_mask.astype(np.uint8) * 255)

    # 1. COCO pretrained bowl/cup/food
    coco_res = coco_model.predict(source=im_bgr, conf=0.06, iou=0.45, verbose=False)[0]
    for b in coco_res.boxes:
        cid = int(b.cls[0].cpu())
        if cid in COCO_FOOD_IDS:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].cpu().tolist()]
            area_frac = ((x2 - x1) * (y2 - y1)) / img_area
            if 0.006 <= area_frac <= 0.42:
                proposals.append((x1, y1, x2, y2, float(b.conf[0].cpu()) + 0.40))

    # 2. Leaf-masked HoughCircles for metal/ceramic katori bowls
    gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    min_r = max(18, int(min(w, h) * 0.050))
    max_r = int(min(w, h) * 0.16)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(min_r * 1.9),
        param1=90, param2=44, minRadius=min_r, maxRadius=max_r
    )
    if circles is not None:
        for cx, cy, r in np.round(circles[0, :]).astype(int):
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)
            if (x2 - x1) >= 32 and (y2 - y1) >= 32:
                if float(np.mean(leaf_mask[y1:y2, x1:x2])) < 0.20:
                    if not is_flower_corner(hsv, x1, y1, x2, y2, w, h):
                        proposals.append((float(x1), float(y1), float(x2), float(y2), 0.62))

    # 3. Banana-leaf food-pile segmentation (catches small Poriyal, Avial, Chutney, Rice, Banana)
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    leaf_filled = cv2.morphologyEx(leaf_u8, cv2.MORPH_CLOSE, kernel_big)
    if np.mean(leaf_mask) > 0.08:
        on_leaf_food = cv2.bitwise_and(leaf_filled, cv2.bitwise_not(leaf_u8))
        on_leaf_food = cv2.morphologyEx(on_leaf_food, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        on_leaf_food = cv2.morphologyEx(on_leaf_food, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        contours, _ = cv2.findContours(on_leaf_food, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area_frac = (bw * bh) / img_area
            aspect = max(bw / max(bh, 1), bh / max(bw, 1))
            if 0.004 <= area_frac <= 0.36 and bw >= 24 and bh >= 24 and aspect <= 2.6:
                box_green = float(np.mean(leaf_mask[y:y + bh, x:x + bw]))
                if box_green < 0.48 and not is_flower_corner(hsv, x, y, x + bw, y + bh, w, h):
                    proposals.append((float(x), float(y), float(x + bw), float(y + bh), 0.55))

    if not proposals:
        return []

    boxes_t = torch.tensor([[p[0], p[1], p[2], p[3]] for p in proposals], dtype=torch.float32)
    scores_t = torch.tensor([p[4] for p in proposals], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.38)
    kept_boxes = [boxes_t[i].tolist() for i in keep.tolist()]

    # Remove nested proposals (>= 55% inside a larger box)
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


def annotate_with_prototypes(coco_model, v2_model, extractor, transform, proto_cids, proto_matrix, img_path: Path):
    im_bgr = cv2.imread(str(img_path))
    if im_bgr is None:
        return []
    h, w = im_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))

    tight_boxes = propose_tight_boxes(coco_model, im_bgr)
    if not tight_boxes:
        return []

    crops_pil = []
    crops_tensor = []
    valid_boxes = []
    for x1, y1, x2, y2 in tight_boxes:
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, int(x1 - bw * 0.06)), max(0, int(y1 - bh * 0.06))
        cx2, cy2 = min(w, int(x2 + bw * 0.06)), min(h, int(y2 + bh * 0.06))
        if cx2 - cx1 < 22 or cy2 - cy1 < 22:
            continue
        crop = pil_img.crop((cx1, cy1, cx2, cy2))
        crops_pil.append(crop)
        crops_tensor.append(transform(crop))
        valid_boxes.append((x1, y1, x2, y2))

    if not valid_boxes:
        return []

    # 1. ResNet50 prototype cosine similarities
    with torch.no_grad():
        batch = torch.stack(crops_tensor)
        feats = extractor(batch).squeeze(-1).squeeze(-1)
        feats = feats / feats.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sims = torch.matmul(feats, proto_matrix.T)  # (N, C)

    # 2. YOLO V2 boost on padded crops
    yolo_preds = v2_model.predict(source=crops_pil, conf=0.10, iou=0.5, classes=INDIAN_MEAL_CLASS_IDS, verbose=False)

    cid_to_col = {cid: idx for idx, cid in enumerate(proto_cids)}
    lines = []
    for i, ((x1, y1, x2, y2), y_res) in enumerate(zip(valid_boxes, yolo_preds)):
        scores = sims[i].clone()
        # Add YOLO confidence boost if YOLO agrees on a dish
        for b in y_res.boxes:
            ycid = int(b.cls[0].cpu())
            yconf = float(b.conf[0].cpu())
            if ycid in cid_to_col:
                scores[cid_to_col[ycid]] += 0.25 * yconf

        best_idx = int(torch.argmax(scores).item())
        best_score = float(sims[i, best_idx].item())
        if best_score < 0.50:
            continue

        cid = proto_cids[best_idx]
        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return lines


def main():
    coco_model = YOLO(str(COCO_MODEL_PATH) if COCO_MODEL_PATH.exists() else "yolov8s.pt")
    v2_model = YOLO(str(V2_MODEL_PATH))
    extractor, transform = build_resnet_extractor()
    proto_cids, proto_matrix = build_prototype_bank(extractor, transform)

    updated = 0
    total_boxes = 0
    for sp in ("train", "val", "test"):
        img_dir = V3_USED_DIR / "images" / sp
        lbl_dir = V3_USED_DIR / "labels" / sp
        targets = sorted(list(img_dir.glob("v3_bananaleaf_*.jpg")) + list(img_dir.glob("v3_platter_*.jpg")))
        print(f"\n🔄 Annotating {len(targets)} multi-dish images in {sp}...")
        for idx, img_p in enumerate(targets, 1):
            if img_p.stem in PRESERVED_STEMS:
                continue
            lines = annotate_with_prototypes(coco_model, v2_model, extractor, transform, proto_cids, proto_matrix, img_p)
            if lines:
                (lbl_dir / f"{img_p.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                updated += 1
                total_boxes += len(lines)
            if idx % 100 == 0:
                print(f"  [{sp}] {idx}/{len(targets)} | Updated: {updated} | Total boxes: {total_boxes}")

    print("\n" + "=" * 65)
    print(f"✅ Offline ResNet50 + YOLO Multi-Dish Annotation Complete!")
    print(f"   Images Updated : {updated} (+3 preserved)")
    print(f"   Total Boxes    : {total_boxes}")
    print("=" * 65)


if __name__ == "__main__":
    main()

