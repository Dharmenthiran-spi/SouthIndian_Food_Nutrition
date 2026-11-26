"""
Food Detection Version 3 - Dataset Preparation & Multi-Dish Integration Script
Combines:
1. All Version 2 single-dish images & labels (11,050 images across 221 classes).
2. All 500 downloaded real multi-dish images (Thali, Banana Leaf, Combo Plates),
   auto-annotated with multi-object YOLO bounding boxes using Version 2 model
   with 5-region overlapping tile scanning + NMS.
3. Synthetic multi-dish combo plates (600 stitched 2x2 / 3-dish / 4-dish plates)
   generated from Version 2 dishes with 100% exact multi-object bounding boxes.

Output: Food_Detection/DataSet/Version3 Used Data/ (images, labels, dataset.yaml)
"""

import os
import shutil
import random
import yaml
import torch
import torchvision
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[2]
V2_USED_DIR = BASE_DIR / "DataSet" / "Version2 Used Data"
RAW_V3_DIR = BASE_DIR / "DataSet" / "RawData" / "Version3" / "multi_dish_images"
V3_USED_DIR = BASE_DIR / "DataSet" / "Version3 Used Data"
V2_MODEL_PATH = BASE_DIR / "Models" / "Food Detection" / "Version2" / "food_detection_v2_best.pt"


def auto_annotate_multidish(model, img_path: Path, conf_thresh=0.20, iou_thresh=0.45):
    """
    Uses trained Version 2 YOLO model + 5 overlapping plate tiles + NMS
    to generate multi-object YOLO bounding box annotations for real multi-dish images.
    Returns list of strings: "cls_id xc yc w h"
    """
    with Image.open(img_path) as im:
        pil_img = im.convert("RGB")
    w, h = pil_img.size

    all_boxes = []
    all_scores = []
    all_clses = []

    # Full-image pass
    full_res = model.predict(source=pil_img, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
    for b in full_res.boxes:
        all_boxes.append(b.xyxy[0].cpu())
        all_scores.append(float(b.conf[0].cpu()))
        all_clses.append(int(b.cls[0].cpu()))

    # 5 overlapping tiles for bowls & side dishes
    if w >= 160 and h >= 160:
        crop_specs = [
            (0, 0, int(w * 0.60), int(h * 0.60)),
            (int(w * 0.40), 0, w, int(h * 0.60)),
            (0, int(h * 0.40), int(w * 0.60), h),
            (int(w * 0.40), int(h * 0.40), w, h),
            (int(w * 0.20), int(h * 0.20), int(w * 0.80), int(h * 0.80)),
        ]
        crops = [pil_img.crop(box) for box in crop_specs]
        tile_res = model.predict(source=crops, conf=conf_thresh, iou=iou_thresh, verbose=False)
        for res, (ox, oy, _, _) in zip(tile_res, crop_specs):
            for b in res.boxes:
                xyxy = b.xyxy[0].cpu().clone()
                xyxy[0] += ox
                xyxy[1] += oy
                xyxy[2] += ox
                xyxy[3] += oy
                all_boxes.append(xyxy)
                all_scores.append(float(b.conf[0].cpu()))
                all_clses.append(int(b.cls[0].cpu()))

    if not all_boxes:
        return []

    boxes_t = torch.stack(all_boxes)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    cls_t = torch.tensor(all_clses, dtype=torch.int64)

    keep = torchvision.ops.batched_nms(boxes_t, scores_t, cls_t, iou_threshold=iou_thresh)
    boxes_t, scores_t, cls_t = boxes_t[keep], scores_t[keep], cls_t[keep]

    keep_cross = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.70)
    boxes_t, scores_t, cls_t = boxes_t[keep_cross], scores_t[keep_cross], cls_t[keep_cross]

    lines = []
    for box, cid in zip(boxes_t, cls_t):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        if bw < 0.04 or bh < 0.04:
            continue
        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        lines.append(f"{int(cid.item())} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return lines


def generate_synthetic_plates(v2_split_imgs, v2_split_lbls, out_img_dir, out_lbl_dir, split_name, num_plates):
    """
    Stitches 4 random Version 2 dish crops onto a 640x640 plate/banana-leaf style
    canvas with exact ground-truth multi-object YOLO bounding boxes.
    """
    if len(v2_split_imgs) < 4:
        return 0

    created = 0
    canvas_size = 640
    bg_colors = [
        (34, 85, 34),    # Banana leaf dark green
        (46, 105, 42),   # Banana leaf medium green
        (225, 220, 210), # Ceramic plate off-white
        (205, 195, 180), # Steel/stone plate neutral
    ]

    for i in range(num_plates):
        picks = random.sample(v2_split_imgs, 4)
        bg = random.choice(bg_colors)
        canvas = Image.new("RGB", (canvas_size, canvas_size), bg)
        labels = []

        # 2x2 quadrants with slight padding & random jitter so boxes don't overlap
        quads = [
            (12, 12, 308, 308),
            (332, 12, 628, 308),
            (12, 332, 308, 628),
            (332, 332, 628, 628),
        ]

        for img_p, (qx1, qy1, qx2, qy2) in zip(picks, quads):
            lbl_p = v2_split_lbls / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue
            first_line = lbl_p.read_text(encoding="utf-8").strip().splitlines()
            if not first_line:
                continue
            cls_id = int(first_line[0].split()[0])

            try:
                with Image.open(img_p) as dish_im:
                    dish_rgb = dish_im.convert("RGB")
            except Exception:
                continue

            qw = qx2 - qx1
            qh = qy2 - qy1
            scale = random.uniform(0.82, 0.98)
            tw = int(qw * scale)
            th = int(qh * scale)
            dish_resized = dish_rgb.resize((tw, th), Image.Resampling.BILINEAR)

            ox = qx1 + random.randint(0, max(0, qw - tw))
            oy = qy1 + random.randint(0, max(0, qh - th))
            canvas.paste(dish_resized, (ox, oy))

            xc = (ox + tw / 2.0) / canvas_size
            yc = (oy + th / 2.0) / canvas_size
            bw = tw / canvas_size
            bh = th / canvas_size
            labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if len(labels) >= 2:
            out_stem = f"v3_synth_plate_{split_name}_{i+1:04d}"
            canvas.save(out_img_dir / f"{out_stem}.jpg", "JPEG", quality=90)
            (out_lbl_dir / f"{out_stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            created += 1

    return created


def prepare_v3_dataset():
    random.seed(42)

    if not V2_USED_DIR.exists():
        raise FileNotFoundError(f"Version 2 dataset not found at {V2_USED_DIR}")
    if not RAW_V3_DIR.exists():
        raise FileNotFoundError(f"Version 3 raw multi-dish images not found at {RAW_V3_DIR}")

    print("=" * 65)
    print("🚀 PREPARING FOOD DETECTION VERSION 3 DATASET")
    print("=" * 65)

    for split in ["train", "val", "test"]:
        (V3_USED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (V3_USED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Copy / link Version 2 dataset (11,050 images across 221 classes)
    v2_counts = {}
    for split in ["train", "val", "test"]:
        src_img_dir = V2_USED_DIR / "images" / split
        src_lbl_dir = V2_USED_DIR / "labels" / split
        dst_img_dir = V3_USED_DIR / "images" / split
        dst_lbl_dir = V3_USED_DIR / "labels" / split

        imgs = list(src_img_dir.glob("*.jpg"))
        for im in imgs:
            dst_im = dst_img_dir / im.name
            if not dst_im.exists():
                shutil.copy2(im, dst_im)
            lbl = src_lbl_dir / f"{im.stem}.txt"
            if lbl.exists():
                dst_lbl = dst_lbl_dir / lbl.name
                if not dst_lbl.exists():
                    shutil.copy2(lbl, dst_lbl)
        v2_counts[split] = len(imgs)
    print(f"✅ Copied Version 2 base splits: {v2_counts} (Total: {sum(v2_counts.values())})")

    # 2. Auto-annotate 500 real multi-dish images using Version 2 model
    raw_imgs = sorted(RAW_V3_DIR.glob("*.jpg"))
    random.shuffle(raw_imgs)
    n_raw = len(raw_imgs)
    n_tr = int(n_raw * 0.80)
    n_va = int(n_raw * 0.10)
    raw_splits = {
        "train": raw_imgs[:n_tr],
        "val": raw_imgs[n_tr:n_tr + n_va],
        "test": raw_imgs[n_tr + n_va:],
    }

    if V2_MODEL_PATH.exists():
        print(f"🔍 Auto-annotating {n_raw} real multi-dish images with {V2_MODEL_PATH.name}...")
        v2_model = YOLO(str(V2_MODEL_PATH))
    else:
        print("⚠️ Version 2 weights not found locally; skipping auto-annotation.")
        v2_model = None

    real_multi_counts = {"train": 0, "val": 0, "test": 0}
    annotated_boxes_total = 0
    for split, files in raw_splits.items():
        dst_img_dir = V3_USED_DIR / "images" / split
        dst_lbl_dir = V3_USED_DIR / "labels" / split
        for img_p in files:
            lines = auto_annotate_multidish(v2_model, img_p) if v2_model else []
            if not lines:
                # Skip unannotated images or save empty background label
                continue
            dst_im = dst_img_dir / img_p.name
            if not dst_im.exists():
                shutil.copy2(img_p, dst_im)
            (dst_lbl_dir / f"{img_p.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            real_multi_counts[split] += 1
            annotated_boxes_total += len(lines)

    print(f"✅ Added auto-annotated real multi-dish images: {real_multi_counts} (Total boxes: {annotated_boxes_total})")

    # 3. Generate synthetic multi-dish combo plates (480 train / 60 val / 60 test = 600 plates)
    synth_targets = {"train": 480, "val": 60, "test": 60}
    synth_counts = {}
    for split, target_n in synth_targets.items():
        v2_imgs = list((V2_USED_DIR / "images" / split).glob("*.jpg"))
        v2_lbls = V2_USED_DIR / "labels" / split
        dst_img_dir = V3_USED_DIR / "images" / split
        dst_lbl_dir = V3_USED_DIR / "labels" / split
        synth_counts[split] = generate_synthetic_plates(v2_imgs, v2_lbls, dst_img_dir, dst_lbl_dir, split, target_n)
    print(f"✅ Generated synthetic multi-dish plates: {synth_counts} (Total: {sum(synth_counts.values())})")

    # 4. Generate Version 3 dataset.yaml
    v2_yaml = yaml.safe_load((V2_USED_DIR / "dataset.yaml").read_text(encoding="utf-8"))
    v3_yaml_data = {
        "path": str(V3_USED_DIR).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": v2_yaml["nc"],
        "names": v2_yaml["names"],
    }
    yaml_path = V3_USED_DIR / "dataset.yaml"
    yaml_path.write_text(yaml.dump(v3_yaml_data, sort_keys=False), encoding="utf-8")

    # Summary
    total_counts = {s: len(list((V3_USED_DIR / "images" / s).glob("*.jpg"))) for s in ["train", "val", "test"]}
    print("\n" + "=" * 65)
    print("🎉 VERSION 3 DATASET READY!")
    print(f"   Train : {total_counts['train']} images")
    print(f"   Val   : {total_counts['val']} images")
    print(f"   Test  : {total_counts['test']} images")
    print(f"   Total : {sum(total_counts.values())} images (221 classes)")
    print(f"   YAML  : {yaml_path}")
    print("=" * 65)


if __name__ == "__main__":
    prepare_v3_dataset()

