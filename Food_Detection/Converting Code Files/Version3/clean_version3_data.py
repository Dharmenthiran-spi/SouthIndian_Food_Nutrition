"""
Food Detection Version 3 - Strict Quality & Relevance Cleaner
Analyzes all Banana Leaf images in RawData/Version3/banana_leaf_foods and Version3 Used Data,
and removes:
1. Near-duplicate / re-encoded images (64-bit perceptual dHash)
2. Non-banana-leaf images (HSV green leaf ratio < 8%)
3. Clipart / white-background graphics / posters (white ratio > 30% or low color variance)
4. Single-box / weak / non-multi-dish images (< 2 food bounding boxes in YOLO labels)
5. Any orphan images or labels in Version3 Used Data
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_BANANA_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "RawData" / "Version3" / "banana_leaf_foods"
V3_USED_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "Version3 Used Data"


def dhash(pil_img: Image.Image, hash_size=8) -> int:
    gray = pil_img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum(1 << i for i, v in enumerate(diff.flatten()) if v)


def hamming_dist(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def find_label_boxes_count(stem: str) -> int:
    for sp in ("train", "val", "test"):
        lbl = V3_USED_DIR / "labels" / sp / f"{stem}.txt"
        if lbl.exists():
            lines = [ln.strip() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return len(lines)
    return 0


def remove_from_used_data(stem: str) -> int:
    removed = 0
    for sp in ("train", "val", "test"):
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            p = V3_USED_DIR / sub / sp / f"{stem}{ext}"
            if p.exists():
                p.unlink(missing_ok=True)
                removed += 1
    return removed


def clean_banana_leaf():
    if not RAW_BANANA_DIR.exists():
        print(f"Directory not found: {RAW_BANANA_DIR}")
        return

    imgs = sorted(RAW_BANANA_DIR.glob("*.jpg"))
    print("=" * 65)
    print(f"🔍 ANALYZING {len(imgs)} BANANA LEAF IMAGES FOR UNWANTED DATA")
    print("=" * 65)

    seen_hashes = []
    reasons = {
        "corrupt": 0,
        "low_green_not_leaf": 0,
        "high_white_or_flat_graphic": 0,
        "too_few_food_boxes": 0,
        "near_duplicate": 0,
    }
    kept = 0
    used_removed = 0

    for p in imgs:
        im_bgr = cv2.imread(str(p))
        if im_bgr is None or im_bgr.shape[0] < 180 or im_bgr.shape[1] < 180:
            reasons["corrupt"] += 1
            p.unlink(missing_ok=True)
            used_removed += remove_from_used_data(p.stem)
            continue

        hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
        green_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 25)
        green_ratio = float(np.mean(green_mask))

        white_mask = (hsv[:, :, 1] < 20) & (hsv[:, :, 2] > 235)
        white_ratio = float(np.mean(white_mask))

        color_std = float(np.std(im_bgr))

        if green_ratio < 0.08:
            reasons["low_green_not_leaf"] += 1
            p.unlink(missing_ok=True)
            used_removed += remove_from_used_data(p.stem)
            continue

        if white_ratio > 0.30 or color_std < 28.0:
            reasons["high_white_or_flat_graphic"] += 1
            p.unlink(missing_ok=True)
            used_removed += remove_from_used_data(p.stem)
            continue

        box_cnt = find_label_boxes_count(p.stem)
        if box_cnt < 2:
            reasons["too_few_food_boxes"] += 1
            p.unlink(missing_ok=True)
            used_removed += remove_from_used_data(p.stem)
            continue

        with Image.open(p) as pil_im:
            h = dhash(pil_im)
        if any(hamming_dist(h, prev) <= 4 for prev in seen_hashes):
            reasons["near_duplicate"] += 1
            p.unlink(missing_ok=True)
            used_removed += remove_from_used_data(p.stem)
            continue

        seen_hashes.append(h)
        kept += 1

    # Sync check across all Version3 Used Data (remove any orphan images or orphan labels)
    orphans_removed = 0
    for sp in ("train", "val", "test"):
        img_dir = V3_USED_DIR / "images" / sp
        lbl_dir = V3_USED_DIR / "labels" / sp
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        img_stems = {f.stem for f in img_dir.glob("*.jpg")}
        lbl_stems = {f.stem for f in lbl_dir.glob("*.txt")}

        for s in img_stems - lbl_stems:
            (img_dir / f"{s}.jpg").unlink(missing_ok=True)
            orphans_removed += 1
        for s in lbl_stems - img_stems:
            (lbl_dir / f"{s}.txt").unlink(missing_ok=True)
            orphans_removed += 1

    print("\n🧹 Cleanup Breakdown:")
    for k, v in reasons.items():
        print(f"   - Removed ({k}): {v}")
    print(f"   - Total unwanted raw images removed : {sum(reasons.values())}")
    print(f"   - Corresponding Used Data files removed: {used_removed}")
    print(f"   - Orphan files cleaned              : {orphans_removed}")
    print(f"✅ Clean verified Banana Leaf images remaining: {kept}")

    final_counts = {s: len(list((V3_USED_DIR / "images" / s).glob("*.jpg"))) for s in ["train", "val", "test"]}
    print("\n" + "=" * 65)
    print(f"📊 FINAL CLEAN VERSION 3 DATASET COUNTS:")
    print(f"   Train : {final_counts['train']}")
    print(f"   Val   : {final_counts['val']}")
    print(f"   Test  : {final_counts['test']}")
    print(f"   Total : {sum(final_counts.values())} images")
    print("=" * 65)


if __name__ == "__main__":
    clean_banana_leaf()

