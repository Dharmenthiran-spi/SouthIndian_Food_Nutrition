"""
Migrate Version 1 YOLOv8 images and labels into Version 2 (221 Classes).
Remaps all 31 Version 1 classes into the new wise categorical order (nc=221).
Leaves out non-matching accessories (e.g. dips that don't exist in V2) to preserve pristine ground truth quality.
"""

import os
import shutil
import yaml
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(BASE_DIR, "Food_Detection", "DataSet", "Version1 Used Data")
V2_DIR = os.path.join(BASE_DIR, "Food_Detection", "DataSet", "Version2 Used Data")

v1_yaml = yaml.safe_load(open(os.path.join(V1_DIR, "dataset.yaml"), encoding="utf-8"))
v1_names = v1_yaml['names']
if isinstance(v1_names, list):
    v1_dict = {i: name for i, name in enumerate(v1_names)}
else:
    v1_dict = {int(k): v for k, v in v1_names.items()}

v2_yaml = yaml.safe_load(open(os.path.join(V2_DIR, "dataset.yaml"), encoding="utf-8"))
v2_names = v2_yaml['names']
v2_inv = {name.lower().strip(): int(idx) for idx, name in v2_names.items()}

# Pristine mapping: Only map when visual ground truth is authentic
CLEAN_MAPPING = {
    "Appam": "Appam",
    "Beetroot poriyal": "Poriyal",
    "Boiled Egg": "Egg boiled",
    "Carrot poriyal": "Poriyal",
    "Chicken 65": "Chicken 65",
    "Chicken briyani": "Chicken biryani",
    "Dosa": "Dosa",
    "Idly": "Idli",
    "Kali": "Ragi koozh",
    "Koozh": "Ragi koozh",
    "Lemon Rice": "Lemon rice",
    "Mushroom briyani": "Vegetable pulao",
    "Mutton Briyani": "Mutton biryani",
    "Nandu masala": "Crab curry",
    "Nei satham": "Rice",
    "Paal kolukattai": "Payasam",
    "Paneer briyani": "Vegetable pulao",
    "Panner masala": "Paneer butter masala",
    "Parupu vada": "Vada",
    "Pidi kolukattai": "Idiyappam",
    "Poorna kolukattai": "Laddu",
    "Prawn thokku": "Prawn masala",
    "Sambar": "Sambar",
    "Sambar satham": "Sambar rice",
    "Satham": "Rice",
    "Uzhuntha vadai": "Vada",
    "Veg briyani": "Vegetable pulao",
    "Ven Pongal": "Pongal"
}

# Build v1_id -> v2_id map
id_map = {}
for v1_id, v1_name in v1_dict.items():
    if v1_name in CLEAN_MAPPING:
        v2_target = CLEAN_MAPPING[v1_name]
        v2_id = v2_inv.get(v2_target.lower().strip())
        if v2_id is not None:
            id_map[v1_id] = (v2_id, v2_target)

print(f"Mapped {len(id_map)} / {len(v1_dict)} classes from V1 to V2.")

total_images_copied = 0
total_labels_processed = 0
total_boxes_remapped = 0
remapped_counter = Counter()

for split in ["train", "val", "test"]:
    v1_img_split = os.path.join(V1_DIR, "images", split)
    v1_lbl_split = os.path.join(V1_DIR, "labels", split)
    
    v2_img_split = os.path.join(V2_DIR, "images", split)
    v2_lbl_split = os.path.join(V2_DIR, "labels", split)
    
    os.makedirs(v2_img_split, exist_ok=True)
    os.makedirs(v2_lbl_split, exist_ok=True)
    
    if not os.path.exists(v1_img_split):
        continue
        
    img_files = [f for f in os.listdir(v1_img_split) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nProcessing '{split}' split ({len(img_files)} images)...")
    
    for img_name in img_files:
        src_img = os.path.join(v1_img_split, img_name)
        dst_img = os.path.join(v2_img_split, img_name)
        
        # Copy image if not already present
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
        total_images_copied += 1
        
        # Process corresponding label
        base_name = os.path.splitext(img_name)[0]
        src_lbl = os.path.join(v1_lbl_split, base_name + ".txt")
        dst_lbl = os.path.join(v2_lbl_split, base_name + ".txt")
        
        new_lines = []
        if os.path.exists(src_lbl):
            with open(src_lbl, "r", encoding="utf-8") as lf:
                lines = lf.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                v1_cid = int(parts[0])
                if v1_cid in id_map:
                    v2_cid, v2_cname = id_map[v1_cid]
                    new_line = f"{v2_cid} " + " ".join(parts[1:]) + "\n"
                    new_lines.append(new_line)
                    remapped_counter[v2_cname] += 1
                    total_boxes_remapped += 1
                    
        with open(dst_lbl, "w", encoding="utf-8") as out_lf:
            out_lf.writelines(new_lines)
            
        total_labels_processed += 1

print("\n" + "="*50)
print("MIGRATION SUMMARY TO VERSION 2 USED DATA")
print("="*50)
print(f"Total Images Migrated: {total_images_copied}")
print(f"Total Labels Written: {total_labels_processed}")
print(f"Total Bounding Boxes Remapped: {total_boxes_remapped}")
print(f"Unique V2 Classes Seeded: {len(remapped_counter)}")
print("\nTop Seeded Classes in Version 2:")
for cname, count in remapped_counter.most_common(12):
    print(f"  - {cname:22}: {count} bounding boxes")

