"""
Food Detection Version 3 - Gemini Vision Precision Auto-Annotator
Uses Gemini Vision REST API (gemini-3-flash-preview with fallbacks to
gemini-3.5-flash-lite and gemini-3.1-flash-lite) to accurately detect and label
every food dish and bowl on all Banana Leaf (v3_bananaleaf_*) and Assorted Meal
Platter (v3_platter_*) images across train, val, and test splits.

Usage (PowerShell):
    $env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
    python "Food_Detection/Converting Code Files/Version3/gemini_annotate_v3.py"
"""

import os
import io
import json
import time
import base64
import yaml
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
V3_USED_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "Version3 Used Data"
YAML_PATH = V3_USED_DIR / "dataset.yaml"

MODELS_CASCADE = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
]

INDIAN_MEAL_IDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 117, 163, 168, 177, 178, 179, 180, 183, 184, 185, 186, 188,
    221, 222,
]

PRESERVED_STEMS = {"v3_bananaleaf_0318", "v3_bananaleaf_0241", "v3_platter_0314"}


def load_class_menu():
    names = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))["names"]
    menu_lines = [f"{cid}: {names[cid]}" for cid in INDIAN_MEAL_IDS if cid in names]
    return "\n".join(menu_lines), names


def encode_image_b64(img_path: Path, max_side=1024) -> str:
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def annotate_with_gemini(api_key: str, img_path: Path, class_menu: str, valid_ids: set):
    prompt = f"""Detect every visible Indian food dish, rice pile, curry bowl, sweet, or fruit in this meal image.
Return ONLY a JSON array of objects, each with:
- "class_id": integer from the allowed list below
- "box_2d": [ymin, xmin, ymax, xmax] normalized from 0 to 1000

Allowed class_ids:
{class_menu}

Rules:
- Only label real food items/bowls. Do NOT label flowers, empty banana leaf, spoons, or tables.
- Draw tight bounding boxes around each individual bowl or food pile.
- Output pure JSON array only."""

    b64_img = encode_image_b64(img_path)
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/jpeg", "data": b64_img}},
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
        },
    }

    last_err = None
    for model_name in MODELS_CASCADE:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=35)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text.startswith("```"):
                    text = text.strip("`").replace("json", "", 1).strip()
                items = json.loads(text)
                yolo_lines = []
                for item in items:
                    cid = int(item["class_id"])
                    if cid not in valid_ids:
                        continue
                    ymin, xmin, ymax, xmax = [float(v) / 1000.0 for v in item["box_2d"]]
                    ymin, ymax = max(0.0, min(ymin, 1.0)), max(0.0, min(ymax, 1.0))
                    xmin, xmax = max(0.0, min(xmin, 1.0)), max(0.0, min(xmax, 1.0))
                    bw, bh = xmax - xmin, ymax - ymin
                    if bw < 0.025 or bh < 0.025:
                        continue
                    xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                    yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                return yolo_lines
            elif resp.status_code in (429, 503):
                last_err = f"{model_name} HTTP {resp.status_code}"
                time.sleep(2.0)
                continue
            else:
                last_err = f"{model_name} HTTP {resp.status_code}: {resp.text[:150]}"
        except Exception as e:
            last_err = f"{model_name} exc: {e}"
            continue

    raise RuntimeError(last_err or "All Gemini models failed")


def main():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Missing GEMINI_API_KEY environment variable!", flush=True)
        return

    class_menu, names = load_class_menu()
    valid_ids = set(names.keys())

    done_log = ROOT_DIR / "Food_Detection" / "Converting Code Files" / "Version3" / "_gemini_done.txt"
    already_done = set(done_log.read_text(encoding="utf-8").splitlines()) if done_log.exists() else set()
    already_done.update(PRESERVED_STEMS)

    lock = threading.Lock()
    stats = {"done": 0, "boxes": 0, "checked": 0}

    def process_one(sp: str, img_p: Path, lbl_dir: Path):
        for attempt in range(3):
            try:
                lines = annotate_with_gemini(api_key, img_p, class_menu, valid_ids)
                if lines:
                    (lbl_dir / f"{img_p.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                with lock:
                    if lines:
                        stats["done"] += 1
                        stats["boxes"] += len(lines)
                    with open(done_log, "a", encoding="utf-8") as df:
                        df.write(img_p.stem + "\n")
                    already_done.add(img_p.stem)
                return
            except Exception as e:
                err_s = str(e)
                if "429" in err_s or "503" in err_s:
                    time.sleep(8 * (attempt + 1))
                else:
                    print(f"  Warning ({img_p.name}): {err_s}", flush=True)
                    time.sleep(1.0)
                    return

    # Annotate val and test first, then train
    for sp in ("val", "test", "train"):
        img_dir = V3_USED_DIR / "images" / sp
        lbl_dir = V3_USED_DIR / "labels" / sp
        all_targets = sorted(list(img_dir.glob("v3_bananaleaf_*.jpg")) + list(img_dir.glob("v3_platter_*.jpg")))
        pending = [p for p in all_targets if p.stem not in already_done]
        print(f"\n🚀 [{sp}] Total: {len(all_targets)} | Pending: {len(pending)}", flush=True)

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(process_one, sp, img_p, lbl_dir) for img_p in pending]
            for i, fut in enumerate(as_completed(futures), 1):
                fut.result()
                if i % 20 == 0 or i == len(pending):
                    print(f"  [{sp}] {i}/{len(pending)} finished | Newly annotated: {stats['done']} | Total boxes: {stats['boxes']}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("✅ Gemini Vision Annotation Complete!", flush=True)
    print(f"   Images Updated : {stats['done']}", flush=True)
    print(f"   Total Boxes    : {stats['boxes']}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
