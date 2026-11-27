"""
Food Detection Version 3 - Add Poori (221) & Chutney (222)
1. Downloads 50 clean images each for Poori and Chutney into RawData/Version3/new_classes/
2. Splits them 80/10/10 (40 train, 5 val, 5 test) into Version3 Used Data/
3. Updates Version3 Used Data/dataset.yaml to nc: 223
4. Adds Poori and Chutney nutrition & Home/Hotel recipes to Version 2/3 JSON maps
"""

import os
import json
import yaml
import random
import hashlib
import shutil
from pathlib import Path
from PIL import Image
from icrawler.builtin import BingImageCrawler

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_NEW_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "RawData" / "Version3" / "new_classes"
V3_USED_DIR = ROOT_DIR / "Food_Detection" / "DataSet" / "Version3 Used Data"
DISH_MAP_PATH = ROOT_DIR / "Incredient_Identification" / "DataSet" / "Version2 Used Data" / "dishes_ingredients_map_v2.json"
HOME_HOTEL_PATH = ROOT_DIR / "Incredient_Identification" / "DataSet" / "Version2 Used Data" / "home_hotel_recipes_v2.json"

CLASSES_TO_ADD = {
    221: {
        "name": "Poori",
        "queries": ["poori masala south indian", "puri bhaji indian food", "poori plate"],
        "target": 50,
    },
    222: {
        "name": "Chutney",
        "queries": [
            "coconut chutney south indian",
            "tomato chutney idli dosa",
            "mint chutney bowl",
            "coriander chutney indian",
            "south indian red chutney",
            "peanut chutney south indian",
            "idli coconut chutney bowl",
            "chutney indian condiment"
        ],
        "target": 50,
    },
}


def is_valid_image(img_path: Path, min_size=150):
    try:
        if img_path.stat().st_size < 5000:
            return False
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            w, h = im.size
            if w < min_size or h < min_size:
                return False
            if max(w / h, h / w) > 3.0:
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


def download_class_images(cls_name: str, queries: list, target: int):
    cls_dir = RAW_NEW_DIR / cls_name
    tmp_dir = RAW_NEW_DIR / f"_tmp_{cls_name}"
    cls_dir.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    seen = set()
    for p in cls_dir.glob("*.jpg"):
        if is_valid_image(p):
            seen.add(file_md5(p))
        else:
            p.unlink(missing_ok=True)

    saved = len(list(cls_dir.glob("*.jpg")))
    print(f"\n📸 Collecting images for '{cls_name}' (Existing: {saved}/{target})")

    for qi, q in enumerate(queries, 1):
        if saved >= target:
            break
        q_dir = tmp_dir / f"q_{qi}"
        q_dir.mkdir(parents=True, exist_ok=True)
        try:
            crawler = BingImageCrawler(
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=4,
                storage={"root_dir": str(q_dir)},
                log_level=40,
            )
            crawler.crawl(keyword=q, max_num=35)
        except Exception as e:
            print(f"  Crawl warning ({q}): {e}")
            continue

        for f in sorted(q_dir.iterdir()):
            if saved >= target:
                break
            if not f.is_file() or not is_valid_image(f):
                continue
            digest = file_md5(f)
            if digest in seen:
                continue
            seen.add(digest)
            saved += 1
            out_p = cls_dir / f"{cls_name.lower()}_{saved:02d}.jpg"
            try:
                with Image.open(f) as im:
                    im.convert("RGB").save(out_p, "JPEG", quality=92)
            except Exception:
                saved -= 1

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"✅ '{cls_name}' collected: {len(list(cls_dir.glob('*.jpg')))} images")
    return sorted(cls_dir.glob("*.jpg"))


def update_nutrition_databases():
    # 1. dishes_ingredients_map_v2.json
    if DISH_MAP_PATH.exists():
        data = json.loads(DISH_MAP_PATH.read_text(encoding="utf-8"))
        data["Poori"] = {
            "food_id": 221,
            "food_name": "Poori",
            "category": "South Indian",
            "group_category": "Full Vegetarian Foods",
            "diet_type": "Vegetarian",
            "fitness_goal": "Energy & Balanced Diet",
            "portion_size": "2 pieces (~120g)",
            "nutrition": {
                "calories_kcal": 310, "protein_g": 6.5, "carbohydrates_g": 38.0, "fat_g": 14.5, "fiber_g": 2.8,
                "sugar_g": 1.0, "net_carbs_g": 35.2, "saturated_fat_g": 2.4, "monounsaturated_fat_g": 7.5,
                "polyunsaturated_fat_g": 4.2, "trans_fat_g": 0.0, "cholesterol_mg": 0.0, "sodium_mg": 210.0,
                "potassium_mg": 145.0, "calcium_mg": 22.0, "iron_mg": 1.9, "magnesium_mg": 38.0,
                "zinc_mg": 0.9, "phosphorus_mg": 95.0, "vitamin_a_mcg": 5.0, "vitamin_c_mg": 0.0,
                "vitamin_d_mcg": 0.0, "vitamin_e_mg": 1.8, "vitamin_b1_thiamine_mg": 0.18, "vitamin_b6_mg": 0.14,
                "vitamin_b9_folate_mcg": 24.0, "vitamin_b12_mcg": 0.0, "water_moisture_g": 42.0
            },
            "main_ingredients": ["whole_wheat_flour", "cooking_oil", "semolina", "salt", "water"],
            "ingredient_display": "Whole Wheat Flour (Atta), Cooking Oil, Semolina (Rava), Salt",
            "cooking_method": "Deep Fried",
            "food_state": "Cooked/Fried"
        }
        data["Chutney"] = {
            "food_id": 222,
            "food_name": "Chutney",
            "category": "South Indian",
            "group_category": "Accompaniments & Sides",
            "diet_type": "Vegetarian",
            "fitness_goal": "Balanced Diet / Daily Fitness",
            "portion_size": "1 small bowl (~80g)",
            "nutrition": {
                "calories_kcal": 115, "protein_g": 2.4, "carbohydrates_g": 6.5, "fat_g": 9.2, "fiber_g": 2.6,
                "sugar_g": 1.5, "net_carbs_g": 3.9, "saturated_fat_g": 6.8, "monounsaturated_fat_g": 1.2,
                "polyunsaturated_fat_g": 0.6, "trans_fat_g": 0.0, "cholesterol_mg": 0.0, "sodium_mg": 190.0,
                "potassium_mg": 160.0, "calcium_mg": 18.0, "iron_mg": 1.1, "magnesium_mg": 24.0,
                "zinc_mg": 0.5, "phosphorus_mg": 45.0, "vitamin_a_mcg": 12.0, "vitamin_c_mg": 4.5,
                "vitamin_d_mcg": 0.0, "vitamin_e_mg": 0.3, "vitamin_b1_thiamine_mg": 0.06, "vitamin_b6_mg": 0.08,
                "vitamin_b9_folate_mcg": 16.0, "vitamin_b12_mcg": 0.0, "water_moisture_g": 58.0
            },
            "main_ingredients": ["fresh_coconut", "roasted_chana_dal", "green_chili", "curry_leaves", "mustard_seeds", "cooking_oil"],
            "ingredient_display": "Fresh Coconut, Roasted Chana Dal, Green Chili, Curry Leaves, Mustard Seeds",
            "cooking_method": "Ground & Tempered",
            "food_state": "Fresh/Tempered"
        }
        DISH_MAP_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✅ Updated {DISH_MAP_PATH.name} ({len(data)} dishes)")

    # 2. home_hotel_recipes_v2.json
    if HOME_HOTEL_PATH.exists():
        hh = json.loads(HOME_HOTEL_PATH.read_text(encoding="utf-8"))
        hh["Poori"] = {
            "home": {
                "portion_desc": "2 home-made whole wheat pooris (~110g)",
                "total_weight_g": 110,
                "ingredients": {"whole_wheat_flour": 65, "cooking_oil": 12, "semolina": 5, "salt": 1},
                "nutrition": {"calories_kcal": 290, "protein_g": 6.2, "carbohydrates_g": 37.0, "fat_g": 12.8, "fiber_g": 2.9},
                "note": "Home preparation drains excess oil well on paper towels."
            },
            "hotel": {
                "portion_desc": "2-3 restaurant pooris (~140g)",
                "total_weight_g": 140,
                "ingredients": {"whole_wheat_flour": 70, "maida": 15, "cooking_oil": 22, "salt": 2},
                "nutrition": {"calories_kcal": 395, "protein_g": 7.0, "carbohydrates_g": 46.0, "fat_g": 20.5, "fiber_g": 2.1},
                "note": "Hotel pooris often mix maida for puffiness and retain higher frying oil."
            }
        }
        hh["Chutney"] = {
            "home": {
                "portion_desc": "1 home bowl coconut/tomato chutney (~80g)",
                "total_weight_g": 80,
                "ingredients": {"fresh_coconut": 35, "roasted_chana_dal": 12, "green_chili": 3, "cooking_oil": 3},
                "nutrition": {"calories_kcal": 115, "protein_g": 2.4, "carbohydrates_g": 6.5, "fat_g": 9.2, "fiber_g": 2.6},
                "note": "Freshly ground coconut with minimal tempering oil."
            },
            "hotel": {
                "portion_desc": "1 restaurant chutney cup (~90g)",
                "total_weight_g": 90,
                "ingredients": {"fresh_coconut": 30, "roasted_chana_dal": 18, "cooking_oil": 7, "water": 35},
                "nutrition": {"calories_kcal": 145, "protein_g": 2.8, "carbohydrates_g": 8.2, "fat_g": 11.4, "fiber_g": 2.2},
                "note": "Hotel chutney uses higher chana dal and tempering oil."
            }
        }
        HOME_HOTEL_PATH.write_text(json.dumps(hh, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✅ Updated {HOME_HOTEL_PATH.name} ({len(hh)} dishes)")


def main():
    random.seed(42)
    yaml_path = V3_USED_DIR / "dataset.yaml"
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = cfg["names"]

    for cls_id, info in CLASSES_TO_ADD.items():
        cls_name = info["name"]
        imgs = download_class_images(cls_name, info["queries"], info["target"])
        random.shuffle(imgs)
        n = len(imgs)
        n_tr = int(n * 0.80)
        n_va = int(n * 0.10)
        splits = {
            "train": imgs[:n_tr],
            "val": imgs[n_tr:n_tr + n_va],
            "test": imgs[n_tr + n_va:],
        }
        slug = cls_name.lower()
        for sp, sp_files in splits.items():
            img_out = V3_USED_DIR / "images" / sp
            lbl_out = V3_USED_DIR / "labels" / sp
            for f in sp_files:
                dst_im = img_out / f"v3_{slug}_{f.name}"
                shutil.copy2(f, dst_im)
                (lbl_out / f"v3_{slug}_{f.stem}.txt").write_text(
                    f"{cls_id} 0.500000 0.500000 0.900000 0.900000\n", encoding="utf-8"
                )
        names[cls_id] = cls_name
        print(f"📦 Added {n} '{cls_name}' images (Class {cls_id}) -> train:{len(splits['train'])}, val:{len(splits['val'])}, test:{len(splits['test'])}")

    cfg["nc"] = len(names)
    cfg["names"] = names
    yaml_path.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"\n✅ Updated {yaml_path} to nc: {cfg['nc']}")

    update_nutrition_databases()


if __name__ == "__main__":
    main()

