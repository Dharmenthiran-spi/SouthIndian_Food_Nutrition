"""
Food Detection Version 2 - Dataset Preparation & Synchronization Script
Prepares and synchronizes 221 clean food classes from RawData/Version2/downloaded_images
into train, val, and test splits with complete YOLO format annotations and dataset.yaml.
"""

import os
import sys
import shutil
import random
import csv
import yaml
from PIL import Image

def prepare_splits(train_ratio=0.80, val_ratio=0.10, test_ratio=0.10):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(base_dir, "DataSet", "RawData", "Version2", "downloaded_images")
    used_dir = os.path.join(base_dir, "DataSet", "Version2 Used Data")
    csv_path = os.path.abspath(os.path.join(base_dir, "..", "Incredient_Identification", "DataSet", "Version2 Used Data", "food_and_nutrition_master_v2.csv"))
    
    if not os.path.exists(raw_dir):
        print(f"❌ Raw directory not found: {raw_dir}")
        return
        
    print("=" * 65)
    print("📁 PREPARING FOOD DETECTION VERSION 2 DATASET SPLITS")
    print("=" * 65)
    print(f"Source Raw Directory: {raw_dir}")
    print(f"Target Used Directory: {used_dir}")
    
    # Load class mapping
    class_map = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                class_map[r['food_name']] = int(r['food_id'])
    else:
        # Fallback to sorted directory names
        folders = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
        class_map = {name: idx for idx, name in enumerate(folders)}
        
    print(f"Loaded {len(class_map)} class definitions.")
    
    # Setup directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(used_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(used_dir, "labels", split), exist_ok=True)
        
    random.seed(42)
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for food_name, class_id in class_map.items():
        food_folder = os.path.join(raw_dir, food_name)
        if not os.path.exists(food_folder):
            continue
            
        imgs = [f for f in os.listdir(food_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not imgs:
            continue
            
        random.shuffle(imgs)
        n_total = len(imgs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]
        
        for split_name, split_files in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            for f in split_files:
                src_p = os.path.join(food_folder, f)
                slug = food_name.lower().replace(' ', '_').replace('-', '_')
                dst_img_name = f"v2_{slug}_{f}"
                dst_lbl_name = f"v2_{slug}_{os.path.splitext(f)[0]}.txt"
                
                dst_img_path = os.path.join(used_dir, "images", split_name, dst_img_name)
                dst_lbl_path = os.path.join(used_dir, "labels", split_name, dst_lbl_name)
                
                if not os.path.exists(dst_img_path):
                    shutil.copy2(src_p, dst_img_path)
                    
                # Full frame annotation for canonical food dishes
                if not os.path.exists(dst_lbl_path):
                    with open(dst_lbl_path, 'w', encoding='utf-8') as lf:
                        lf.write(f"{class_id} 0.500000 0.500000 0.900000 0.900000\n")
                        
                stats[split_name] += 1
                
    print(f"\n✅ Dataset Splits Synchronized:")
    print(f"   Train Images : {stats['train']}")
    print(f"   Val Images   : {stats['val']}")
    print(f"   Test Images  : {stats['test']}")
    print(f"   Total Images : {sum(stats.values())}")
    
    # Generate dataset.yaml
    yaml_path = os.path.join(used_dir, "dataset.yaml")
    yaml_data = {
        'path': used_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_map),
        'names': {cid: cname for cname, cid in class_map.items()}
    }
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yaml.dump(yaml_data, yf, sort_keys=False)
        
    print(f"✅ Generated dataset.yaml with {len(class_map)} classes at: {yaml_path}")

if __name__ == "__main__":
    prepare_splits()

