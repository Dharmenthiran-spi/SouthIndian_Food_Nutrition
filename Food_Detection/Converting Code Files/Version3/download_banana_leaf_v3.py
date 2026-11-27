"""
Food Detection Version 3 - Banana Leaf Foods Downloader & Auto-Annotator
Downloads 500 genuine "Banana Leaf Foods" images (using Google & Bing crawlers),
filters out any non-food/corrupted/duplicate images using food_detection_v2_best.pt,
saves them to Food_Detection/DataSet/RawData/Version3/banana_leaf_foods/,
and adds them (80/10/10) with multi-dish YOLO labels to Version3 Used Data/.
"""

import os
import time
import shutil
import random
import hashlib
import torch
import torchvision
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from bing_image_downloader import downloader

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_BANANA_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "RawData" / "Version3" / "banana_leaf_foods"
TMP_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "RawData" / "Version3" / "_tmp_banana"
V3_USED_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "Version3 Used Data"
V2_MODEL_PATH = ROOT_DIR / "Food_Detection" / "Models" / "Food Detection" / "Version2" / "food_detection_v2_best.pt"

TARGET_COUNT = 500

BANANA_LEAF_QUERIES = [
    "Banana Leaf Foods",
    "South Indian banana leaf meal",
    "Kerala sadya banana leaf food",
    "Tamil Nadu banana leaf meals",
    "Indian food served on banana leaf",
    "banana leaf rice meal South India",
    "traditional banana leaf lunch India",
    "veg meals on banana leaf",
    "non veg meals on banana leaf",
    "Onam sadya banana leaf meal",
    "Chettinad banana leaf meal",
    "Andhra meals on banana leaf",
    "banana leaf biryani meal",
    "banana leaf breakfast idli dosa",
    "fish fry banana leaf meal",
    "wedding feast banana leaf South India",
    "nasi daun pisang banana leaf meal",
    "kalyana virundhu banana leaf",
    "banana leaf meals chennai",
    "banana leaf meals madurai",
    "banana leaf thali restaurant",
    "banana leaf parotta chicken curry",
    "banana leaf dosa sambar chutney",
    "banana leaf idli vada pongal",
    "banana leaf chicken fry meals",
    "banana leaf mutton curry meals",
    "malaysian banana leaf rice curry",
    "indian banana leaf platter food",
    "south indian leaf meals rice sambar",
    "vishu sadya banana leaf food",
    "pongal feast banana leaf meal",
    "banana leaf seafood meal india",
    "banana leaf lunch thali veg",
    "banana leaf dinner south indian",
    "banana leaf full meals rice papad",
    "banana leaf meals kerala style",
    "banana leaf meals tamil style",
    "banana leaf meals karnataka style",
    "banana leaf tiffin south indian",
    "banana leaf appam stew meal",
    "banana leaf puttu kadala meal",
    "banana leaf curd rice pickle",
    "banana leaf lemon rice meal",
    "banana leaf sambar rice poriyal",
    "banana leaf feast payasam",
]


def is_valid_image(img_path: Path, min_size=200):
    try:
        if img_path.stat().st_size < 7000:
            return False
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            w, h = im.size
            if w < min_size or h < min_size:
                return False
            if max(w / h, h / w) > 2.6:
                return False
        return True
    except Exception:
        return False


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def annotate_and_verify_food(model, pil_img: Image.Image, conf_thresh=0.22, iou_thresh=0.45):
    """
    Verifies the image actually contains food dishes and returns multi-dish YOLO label lines.
    """
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
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        if bw < 0.04 or bh < 0.04:
            continue
        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        lines.append(f"{int(cid.item())} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return lines


def main():
    random.seed(42)
    RAW_BANANA_DIR.mkdir(parents=True, exist_ok=True)
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    print("=" * 65)
    print("🌿 DOWNLOADING 500 VERIFIED 'BANANA LEAF FOODS' IMAGES")
    print("=" * 65)

    model = YOLO(str(V2_MODEL_PATH)) if V2_MODEL_PATH.exists() else None
    seen_hashes = set()
    labels_cache = {}

    for p in sorted(RAW_BANANA_DIR.glob("*.jpg")):
        if is_valid_image(p):
            seen_hashes.add(file_md5(p))
        else:
            p.unlink(missing_ok=True)

    saved_count = len(list(RAW_BANANA_DIR.glob("*.jpg")))
    print(f"Existing Banana Leaf images: {saved_count} / {TARGET_COUNT}")

    # Crawl with bing_image_downloader to reach 500 clean Banana Leaf images
    for qi, query in enumerate(BANANA_LEAF_QUERIES, 1):
        if saved_count >= TARGET_COUNT:
            break
        q_dir = TMP_DIR / f"q_{qi}"
        q_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Query {qi}/{len(BANANA_LEAF_QUERIES)}] '{query}' (Current: {saved_count}/{TARGET_COUNT})")
        try:
            downloader.download(
                query,
                limit=50,
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
            if saved_count >= TARGET_COUNT:
                break
            if not f.is_file() or not is_valid_image(f):
                continue
            digest = file_md5(f)
            if digest in seen_hashes:
                continue

            try:
                with Image.open(f) as im:
                    rgb = im.convert("RGB")
            except Exception:
                continue

            # Verify real food content with YOLO model
            if model is not None:
                lbl_lines = annotate_and_verify_food(model, rgb, conf_thresh=0.20)
                if len(lbl_lines) < 1:
                    continue
            else:
                lbl_lines = []

            seen_hashes.add(digest)
            saved_count += 1
            out_name = f"v3_bananaleaf_{saved_count:04d}.jpg"
            out_path = RAW_BANANA_DIR / out_name
            rgb.save(out_path, "JPEG", quality=92)
            labels_cache[out_name] = lbl_lines

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    all_banana = sorted(RAW_BANANA_DIR.glob("*.jpg"))
    print(f"\n✅ Total verified Banana Leaf images collected: {len(all_banana)}")

    # Add into Version3 Used Data (80/10/10 split)
    random.shuffle(all_banana)
    n = len(all_banana)
    n_tr = int(n * 0.80)
    n_va = int(n * 0.10)
    splits = {
        "train": all_banana[:n_tr],
        "val": all_banana[n_tr:n_tr + n_va],
        "test": all_banana[n_tr + n_va:],
    }

    added_counts = {"train": 0, "val": 0, "test": 0}
    total_boxes = 0
    for sp, files in splits.items():
        img_out = V3_USED_DIR / "images" / sp
        lbl_out = V3_USED_DIR / "labels" / sp
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_p in files:
            if img_p.name in labels_cache:
                lines = labels_cache[img_p.name]
            elif model is not None:
                with Image.open(img_p) as im:
                    lines = annotate_and_verify_food(model, im.convert("RGB"), conf_thresh=0.22)
            else:
                lines = []

            if not lines:
                continue

            shutil.copy2(img_p, img_out / img_p.name)
            (lbl_out / f"{img_p.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            added_counts[sp] += 1
            total_boxes += len(lines)

    print(f"✅ Added Banana Leaf images to Version3 Used Data: {added_counts} (Total boxes: {total_boxes})")
    final_counts = {s: len(list((V3_USED_DIR / "images" / s).glob("*.jpg"))) for s in ["train", "val", "test"]}
    print(f"📊 Updated Version 3 Totals: Train={final_counts['train']}, Val={final_counts['val']}, Test={final_counts['test']} (Total={sum(final_counts.values())})")


if __name__ == "__main__":
    main()

