"""
Food Detection Version 3 - Offline Exact-Label Multi-Dish Builder (No API Key Needed)
1. Cleans corner/flower false positives from real Banana Leaf & Platter labels.
2. Generates 1,800 realistic multi-dish Banana Leaf & Thali plates (4 to 6 dishes per plate)
   with 100% EXACT ground-truth YOLO bounding boxes using verified Version 2 + Poori + Chutney
   single-dish images (1,440 train / 180 val / 180 test).
"""

import cv2
import yaml
import random
import numpy as np
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
V2_USED_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "Version2 Used Data"
V3_USED_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "Version3 Used Data"

# Prioritize South Indian Banana Leaf / Thali dishes when sampling synthetic plates
PRIORITY_CLASS_IDS = {
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 32, 56, 57, 58, 59, 66, 68, 69, 74, 75, 77,
    117, 163, 178, 179, 180, 183, 221, 222
}


def clean_corner_flower_boxes():
    """Removes corner decorative flower false positives from v3_bananaleaf_* labels."""
    removed_boxes = 0
    for sp in ("train", "val", "test"):
        img_dir = V3_USED_DIR / "images" / sp
        lbl_dir = V3_USED_DIR / "labels" / sp
        for lbl_p in lbl_dir.glob("v3_bananaleaf_*.txt"):
            img_p = img_dir / f"{lbl_p.stem}.jpg"
            if not img_p.exists():
                continue
            im = cv2.imread(str(img_p))
            if im is None:
                continue
            h, w = im.shape[:2]
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            kept_lines = []
            for ln in lbl_p.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                cid = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])

                # Check if box is in extreme corner with high bright-marigold yellow/orange saturation
                in_corner = (xc < 0.14 or xc > 0.86) and (yc < 0.14 or yc > 0.86)
                if in_corner:
                    x1, y1 = max(0, int((xc - bw / 2) * w)), max(0, int((yc - bh / 2) * h))
                    x2, y2 = min(w, int((xc + bw / 2) * w)), min(h, int((yc + bh / 2) * h))
                    roi = hsv[y1:y2, x1:x2]
                    if roi.size > 0:
                        # Bright marigold orange/yellow flowers: H in [8..32], S > 160, V > 160
                        flower_mask = (roi[:, :, 0] >= 8) & (roi[:, :, 0] <= 32) & (roi[:, :, 1] > 160) & (roi[:, :, 2] > 160)
                        if float(np.mean(flower_mask)) > 0.35:
                            removed_boxes += 1
                            continue
                kept_lines.append(ln)

            if kept_lines:
                lbl_p.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    print(f"🧹 Cleaned {removed_boxes} decorative flower false boxes from Banana Leaf labels.")


def create_leaf_background(size=640):
    """Creates a realistic textured banana-leaf + wooden-table background."""
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    # Dark wooden table border
    bg[:, :] = (42, 62, 88)  # BGR warm brown
    # Green banana leaf polygon/rounded rect in center/bottom
    leaf_top = random.randint(70, 130)
    leaf_bot = size - random.randint(25, 60)
    leaf_left = random.randint(20, 45)
    leaf_right = size - random.randint(20, 45)
    green_tone = (random.randint(25, 45), random.randint(105, 145), random.randint(45, 75))
    cv2.rectangle(bg, (leaf_left, leaf_top), (leaf_right, leaf_bot), green_tone, -1)
    # Subtle horizontal leaf vein lines
    mid_y = (leaf_top + leaf_bot) // 2
    cv2.line(bg, (leaf_left, mid_y), (leaf_right, mid_y), (25, 90, 40), 4)
    for vx in range(leaf_left + 15, leaf_right, 22):
        cv2.line(bg, (vx, leaf_top), (vx + 10, leaf_bot), (28, 100, 45), 1)
    return Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))


def build_realistic_plates():
    random.seed(42)
    targets = {"train": 1440, "val": 180, "test": 180}
    canvas_size = 640

    # Predefined realistic 5-item & 6-item layouts (x1, y1, x2, y2) on 640x640 banana leaf
    layouts_5 = [
        [(60, 20, 195, 155), (245, 20, 380, 155), (435, 20, 570, 155), (65, 290, 275, 515), (330, 285, 575, 535)],
        [(50, 25, 185, 160), (235, 25, 370, 160), (425, 25, 560, 160), (75, 220, 255, 400), (310, 310, 565, 555)],
    ]
    layouts_6 = [
        [(55, 20, 180, 145), (225, 20, 350, 145), (400, 20, 525, 145), (75, 195, 225, 345), (265, 195, 415, 345), (210, 385, 460, 585)],
        [(50, 18, 175, 143), (220, 18, 345, 143), (420, 18, 545, 143), (65, 330, 245, 525), (295, 330, 545, 555), (435, 190, 575, 315)],
    ]

    for sp, count in targets.items():
        img_dir = V3_USED_DIR / "images" / sp
        lbl_dir = V3_USED_DIR / "labels" / sp

        # Remove old 2x2 synthetic plates so we replace with realistic 5-6 dish banana leaf plates
        for old_f in img_dir.glob("v3_synth_plate_*.jpg"):
            old_f.unlink(missing_ok=True)
            (lbl_dir / f"{old_f.stem}.txt").unlink(missing_ok=True)

        # Gather single-dish images (prioritize Indian meal dishes)
        priority_pool = []
        general_pool = []
        for im_p in img_dir.glob("*.jpg"):
            if im_p.name.startswith(("v3_bananaleaf_", "v3_platter_", "v3_synth_")):
                continue
            lbl_p = lbl_dir / f"{im_p.stem}.txt"
            if not lbl_p.exists():
                continue
            lines = [ln.strip() for ln in lbl_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                continue
            cid = int(lines[0].split()[0])
            if cid in PRIORITY_CLASS_IDS:
                priority_pool.append((im_p, cid))
            else:
                general_pool.append((im_p, cid))

        created = 0
        for i in range(count):
            layout = random.choice(layouts_5 if i % 2 == 0 else layouts_6)
            canvas = create_leaf_background(canvas_size)
            labels = []

            # Sample 85% from South Indian meal dishes, 15% from general pool
            picks = []
            for _ in range(len(layout)):
                if priority_pool and (random.random() < 0.85 or not general_pool):
                    picks.append(random.choice(priority_pool))
                else:
                    picks.append(random.choice(general_pool))

            for (im_p, cid), (qx1, qy1, qx2, qy2) in zip(picks, layout):
                try:
                    with Image.open(im_p) as dish_im:
                        dish_rgb = dish_im.convert("RGB")
                except Exception:
                    continue

                qw, qh = qx2 - qx1, qy2 - qy1
                scale = random.uniform(0.86, 1.0)
                tw, th = int(qw * scale), int(qh * scale)
                dish_resized = dish_rgb.resize((tw, th), Image.Resampling.BILINEAR)
                ox = qx1 + random.randint(0, max(0, qw - tw))
                oy = qy1 + random.randint(0, max(0, qh - th))
                canvas.paste(dish_resized, (ox, oy))

                xc = (ox + tw / 2.0) / canvas_size
                yc = (oy + th / 2.0) / canvas_size
                bw, bh = tw / canvas_size, th / canvas_size
                labels.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            if len(labels) >= 4:
                stem = f"v3_synth_plate_{sp}_{i+1:04d}"
                canvas.save(img_dir / f"{stem}.jpg", "JPEG", quality=90)
                (lbl_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
                created += 1

        print(f"✅ [{sp}] Built {created} exact-label multi-dish Banana Leaf plates")

    totals = {s: len(list((V3_USED_DIR / "images" / s).glob("*.jpg"))) for s in ("train", "val", "test")}
    print("\n" + "=" * 65)
    print(f"🎉 FINAL VERSION 3 DATASET COUNTS (223 classes):")
    print(f"   Train : {totals['train']}")
    print(f"   Val   : {totals['val']}")
    print(f"   Test  : {totals['test']}")
    print(f"   Total : {sum(totals.values())} images")
    print("=" * 65)


if __name__ == "__main__":
    clean_corner_flower_boxes()
    build_realistic_plates()

