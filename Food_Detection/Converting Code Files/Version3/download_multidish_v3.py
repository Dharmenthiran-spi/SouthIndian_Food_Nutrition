"""
Food Detection Version 3 - Multi-Dish Raw Image Downloader
Downloads 500 authentic multi-dish food images (Thali sets, Banana leaf meals,
South Indian breakfast combos, etc.) into Food_Detection/DataSet/RawData/Version3/multi_dish_images.
Includes MD5 deduplication and corrupted/tiny image filtering.
"""

import os
import hashlib
import shutil
from pathlib import Path
from PIL import Image
from icrawler.builtin import BingImageCrawler

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_V3_DIR = BASE_DIR / "DataSet" / "RawData" / "Version3" / "multi_dish_images"
TEMP_DL_DIR = BASE_DIR / "DataSet" / "RawData" / "Version3" / "_temp_dl"

TARGET_COUNT = 500

QUERIES = [
    "South Indian banana leaf meal",
    "South Indian thali full meals",
    "Indian vegetarian thali plate",
    "idli vada sambar chutney plate",
    "masala dosa vada sambar chutney",
    "Kerala sadya banana leaf",
    "Tamil Nadu lunch meals plate",
    "South Indian mini tiffin platter",
    "Indian non veg thali platter",
    "North Indian thali combo plate",
    "chicken biryani raita gravy plate",
    "pongal vada idli sambar plate",
    "Indian breakfast combo plate",
    "Andhra meals banana leaf",
    "parotta salna chicken fry plate",
    "poori masala sambar vada plate",
    "curd rice pickle appalam thali",
    "fish curry meals banana leaf",
    "Chettinad non veg meals banana leaf",
    "veg thali restaurant india",
    "mutton biryani combo plate",
    "paneer butter masala naan dal thali",
    "Gujarati thali platter",
    "Rajasthani thali meal",
    "Punjabi thali lunch",
    "Maharashtrian thali food",
    "Bengali thali platter",
    "Indian buffet plate multiple dishes",
    "south indian tiffin combo plate",
    "rice sambar rasam poriyal kootu plate",
    "chapati kurma rice plate",
    "tandoori chicken platter naan",
    "mixed rice combo plate indian",
    "traditional banana leaf feast india",
    "hotel meals south indian plate",
    "special veg meals banana leaf",
    "festival sadya meal plate",
    "egg biryani chicken 65 combo",
]


def is_valid_image(img_path: Path, min_size=200):
    try:
        if img_path.stat().st_size < 8000:
            return False
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            w, h = img.size
            if w < min_size or h < min_size:
                return False
            # Filter extreme aspect ratios (banners)
            ratio = max(w / h, h / w)
            if ratio > 3.0:
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


def download_multidish_images(target_count=TARGET_COUNT):
    RAW_V3_DIR.mkdir(parents=True, exist_ok=True)
    if TEMP_DL_DIR.exists():
        shutil.rmtree(TEMP_DL_DIR, ignore_errors=True)

    seen_hashes = set()
    existing = sorted(RAW_V3_DIR.glob("*.jpg"))
    for p in existing:
        if is_valid_image(p):
            seen_hashes.add(file_md5(p))
        else:
            p.unlink(missing_ok=True)

    saved_count = len(list(RAW_V3_DIR.glob("*.jpg")))
    print(f"Existing valid multi-dish images: {saved_count} / {target_count}")

    per_query = 55
    for q_idx, query in enumerate(QUERIES, 1):
        if saved_count >= target_count:
            break

        q_dir = TEMP_DL_DIR / f"q_{q_idx}"
        q_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{q_idx}/{len(QUERIES)}] Crawling: '{query}' (Current: {saved_count}/{target_count})")

        try:
            crawler = BingImageCrawler(
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=4,
                storage={"root_dir": str(q_dir)},
                log_level=40,
            )
            crawler.crawl(keyword=query, max_num=per_query)
        except Exception as e:
            print(f"  Crawl warning for '{query}': {e}")
            continue

        for img_file in sorted(q_dir.iterdir()):
            if saved_count >= target_count:
                break
            if not img_file.is_file():
                continue
            if not is_valid_image(img_file):
                continue

            digest = file_md5(img_file)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            saved_count += 1
            out_name = f"v3_multidish_{saved_count:04d}.jpg"
            out_path = RAW_V3_DIR / out_name
            try:
                with Image.open(img_file) as im:
                    im.convert("RGB").save(out_path, "JPEG", quality=92)
            except Exception:
                saved_count -= 1

    if TEMP_DL_DIR.exists():
        shutil.rmtree(TEMP_DL_DIR, ignore_errors=True)

    final_count = len(list(RAW_V3_DIR.glob("*.jpg")))
    print("\n" + "=" * 65)
    print(f"✅ Downloaded {final_count} clean multi-dish images to:")
    print(f"   {RAW_V3_DIR}")
    print("=" * 65)
    return final_count


if __name__ == "__main__":
    download_multidish_images(TARGET_COUNT)

