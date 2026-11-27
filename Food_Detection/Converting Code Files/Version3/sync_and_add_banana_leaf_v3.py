"""
Food Detection Version 3 - Sync User Deletions & Add 300 New Verified Banana Leaf Photos
1. Syncs any user deletions from RawData/Version3/banana_leaf_foods/ to Version3 Used Data/.
2. Downloads 300 NEW strictly-verified Banana Leaf Foods photos (HSV green >= 10%,
   white <= 28%, >= 2 YOLO food boxes, perceptual dHash deduplication against existing).
3. Splits the 300 new photos 80/10/10 (240 train / 30 val / 30 test) with multi-dish
   YOLO labels into Version3 Used Data/.
"""

import cv2
import random
import shutil
import hashlib
import numpy as np
import torch
import torchvision
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from bing_image_downloader import downloader

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_BANANA_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "RawData" / "Version3" / "banana_leaf_foods"
TMP_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "RawData" / "Version3" / "_tmp_banana_300"
V3_USED_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "Version3 Used Data"
V2_MODEL_PATH = ROOT_DIR / "Food_Detection" / "Models" / "Food Detection" / "Version2" / "food_detection_v2_best.pt"

NEW_TARGET = 300

QUERIES = [
    "banana leaf rice meal South India",
    "Kerala sadya banana leaf feast",
    "Tamil Nadu banana leaf meals lunch",
    "South Indian banana leaf tiffin",
    "Chettinad banana leaf non veg meals",
    "Andhra meals banana leaf",
    "Onam sadya banana leaf",
    "banana leaf biryani meal",
    "banana leaf fish curry rice",
    "banana leaf parotta chicken curry",
    "banana leaf dosa sambar chutney",
    "banana leaf idli vada pongal",
    "malaysian banana leaf rice curry",
    "banana leaf meals chennai madurai",
    "banana leaf mutton curry meals",
    "banana leaf chicken 65 meals",
    "banana leaf full veg meals papad",
    "traditional banana leaf lunch kerala",
    "vishu sadya banana leaf meal",
    "kalyana virundhu banana leaf feast",
    "banana leaf pongal vada sambar",
    "banana leaf appam stew meal",
    "banana leaf curd rice pickle",
    "banana leaf lemon rice meal",
    "banana leaf sambar rice poriyal",
    "banana leaf seafood meal india",
    "banana leaf meals karnataka",
    "indian thali on banana leaf",
    "banana leaf combo meal restaurant",
    "south indian breakfast on banana leaf",
    "banana leaf mutton biryani",
    "banana leaf chicken curry rice",
    "banana leaf vegetable curry rice",
    "banana leaf festival meal india",
    "banana leaf dinner plate south india",
]


def dhash(pil_img: Image.Image, hash_size=8) -> int:
    gray = pil_img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum(1 << i for i, v in enumerate(diff.flatten()) if v)


def hamming_dist(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def annotate_and_verify_food(model, pil_img: Image.Image, conf_thresh=0.22, iou_thresh=0.45):
    w, h = pil_img.size
    all_boxes, all_scores, all_clses = [], [], []

    full_res = model.predict(source=pil_img, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
    for b in full_res.boxes:
        all_boxes.append(b.xyxy[0].cpu())
        all_scores.append(float(b.conf[0].cpu()))
        all_clses.append(int(b.cls[0].cpu()))

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
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw < 0.04 or bh < 0.04:
            continue
        xc, yc = ((x1 + x2) / 2.0) / w, ((y1 + y2) / 2.0) / h
        lines.append(f"{int(cid.item())} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return lines


def main():
    random.seed(123)
    RAW_BANANA_DIR.mkdir(parents=True, exist_ok=True)
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    # 1. Sync user deletions from RawData/Version3/banana_leaf_foods -> Version3 Used Data
    raw_stems = {p.stem for p in RAW_BANANA_DIR.glob("*.jpg")}
    synced_removed = 0
    for sp in ("train", "val", "test"):
        img_dir = V3_USED_DIR / "images" / sp
        lbl_dir = V3_USED_DIR / "labels" / sp
        for f in list(img_dir.glob("v3_bananaleaf_*.jpg")):
            if f.stem not in raw_stems:
                f.unlink(missing_ok=True)
                (lbl_dir / f"{f.stem}.txt").unlink(missing_ok=True)
                synced_removed += 1

    print("=" * 65)
    print(f"🧹 Synced user deletions: removed {synced_removed} deleted images from Version3 Used Data")
    print(f"📁 Existing clean raw Banana Leaf images: {len(raw_stems)}")
    print(f"🎯 Downloading {NEW_TARGET} NEW strictly-verified Banana Leaf images...")
    print("=" * 65)

    # Build perceptual hash index of existing images so we never re-download duplicates
    seen_hashes = []
    max_num = 0
    for p in sorted(RAW_BANANA_DIR.glob("*.jpg")):
        try:
            num_part = int(p.stem.split("_")[-1])
            if num_part > max_num:
                max_num = num_part
        except Exception:
            pass
        try:
            with Image.open(p) as im:
                seen_hashes.append(dhash(im))
        except Exception:
            pass

    model = YOLO(str(V2_MODEL_PATH)) if V2_MODEL_PATH.exists() else None
    new_files = []
    labels_cache = {}

    for qi, query in enumerate(QUERIES, 1):
        if len(new_files) >= NEW_TARGET:
            break
        q_dir = TMP_DIR / f"q_{qi}"
        q_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Query {qi}/{len(QUERIES)}] '{query}' (New saved: {len(new_files)}/{NEW_TARGET})")
        try:
            downloader.download(
                query,
                limit=45,
                output_dir=str(q_dir),
                adult_filter_off=True,
                force_replace=True,
                timeout=8,
                verbose=False,
            )
        except Exception as e:
            print(f"  Warning ({query}): {e}")
            continue

        dl_sub = q_dir / query
        if not dl_sub.exists():
            continue

        for f in sorted(dl_sub.iterdir()):
            if len(new_files) >= NEW_TARGET:
                break
            if not f.is_file() or f.stat().st_size < 7000:
                continue

            im_bgr = cv2.imread(str(f))
            if im_bgr is None or im_bgr.shape[0] < 200 or im_bgr.shape[1] < 200:
                continue
            h_px, w_px = im_bgr.shape[:2]
            if max(w_px / h_px, h_px / w_px) > 2.5:
                continue

            # Strict Banana Leaf Green & Real-Photo check
            hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
            green_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 25)
            green_ratio = float(np.mean(green_mask))
            white_mask = (hsv[:, :, 1] < 20) & (hsv[:, :, 2] > 235)
            white_ratio = float(np.mean(white_mask))
            color_std = float(np.std(im_bgr))

            if green_ratio < 0.10 or white_ratio > 0.28 or color_std < 28.0:
                continue

            try:
                with Image.open(f) as im:
                    rgb = im.convert("RGB")
            except Exception:
                continue

            h_val = dhash(rgb)
            if any(hamming_dist(h_val, prev) <= 4 for prev in seen_hashes):
                continue

            # Strict Multi-Dish YOLO verification (>= 2 food boxes)
            if model is not None:
                lbl_lines = annotate_and_verify_food(model, rgb, conf_thresh=0.22)
                if len(lbl_lines) < 2:
                    continue
            else:
                lbl_lines = []

            seen_hashes.append(h_val)
            max_num += 1
            out_name = f"v3_bananaleaf_{max_num:04d}.jpg"
            out_path = RAW_BANANA_DIR / out_name
            rgb.save(out_path, "JPEG", quality=92)
            new_files.append(out_path)
            labels_cache[out_name] = lbl_lines

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    print(f"\n✅ Downloaded & verified {len(new_files)} NEW Banana Leaf images!")

    # Split new files 80/10/10 into Version3 Used Data
    random.shuffle(new_files)
    n = len(new_files)
    n_tr = int(n * 0.80)
    n_va = int(n * 0.10)
    splits = {
        "train": new_files[:n_tr],
        "val": new_files[n_tr:n_tr + n_va],
        "test": new_files[n_tr + n_va:],
    }

    added_counts = {"train": 0, "val": 0, "test": 0}
    total_boxes = 0
    for sp, files in splits.items():
        img_out = V3_USED_DIR / "images" / sp
        lbl_out = V3_USED_DIR / "labels" / sp
        for img_p in files:
            lines = labels_cache.get(img_p.name, [])
            if not lines:
                continue
            shutil.copy2(img_p, img_out / img_p.name)
            (lbl_out / f"{img_p.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            added_counts[sp] += 1
            total_boxes += len(lines)

    final_counts = {s: len(list((V3_USED_DIR / "images" / s).glob("*.jpg"))) for s in ["train", "val", "test"]}
    total_raw = len(list(RAW_BANANA_DIR.glob("*.jpg")))
    print("\n" + "=" * 65)
    print(f"🎉 ADDED {sum(added_counts.values())} NEW BANANA LEAF IMAGES ({total_boxes} boxes): {added_counts}")
    print(f"📁 Total Raw Banana Leaf Images : {total_raw}")
    print(f"📊 Final Version 3 Used Data    : Train={final_counts['train']}, Val={final_counts['val']}, Test={final_counts['test']} (Total={sum(final_counts.values())})")
    print("=" * 65)


if __name__ == "__main__":
    main()

