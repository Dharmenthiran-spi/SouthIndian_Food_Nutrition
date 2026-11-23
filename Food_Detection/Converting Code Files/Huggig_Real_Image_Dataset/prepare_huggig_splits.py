"""
Huggig Multi-Dish Dataset -> YOLO Train/Val Split Preparation
Splits flat images and labels into standard train (80%) and val (20%) sets.
"""

import os
import shutil
import random
import yaml

def prepare_splits(
    src_dir=r"D:\Yolo_Food_Inspection\Food_Detection\DataSet\RawData\Version1\Huggig_Real_Image_Dataset",
    dest_dir=r"D:\Yolo_Food_Inspection\Food_Detection\DataSet\Version1 Used Data\Huggig_YOLO",
    train_ratio=0.8,
    seed=42
):
    src_images = os.path.join(src_dir, "images")
    src_labels = os.path.join(src_dir, "labels")

    if not os.path.exists(src_images) or not os.path.exists(src_labels):
        raise FileNotFoundError(f"Images or labels folder not found in {src_dir}")

    images = sorted([f for f in os.listdir(src_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Found {len(images)} images in Huggig dataset.")

    random.seed(seed)
    random.shuffle(images)

    n_train = int(len(images) * train_ratio)
    train_images = images[:n_train]
    val_images = images[n_train:]

    for split_name, img_list in [("train", train_images), ("val", val_images)]:
        dest_img_p = os.path.join(dest_dir, "images", split_name)
        dest_lbl_p = os.path.join(dest_dir, "labels", split_name)

        os.makedirs(dest_img_p, exist_ok=True)
        os.makedirs(dest_lbl_p, exist_ok=True)

        for img_name in img_list:
            stem = os.path.splitext(img_name)[0]
            label_name = f"{stem}.txt"

            shutil.copyfile(os.path.join(src_images, img_name), os.path.join(dest_img_p, img_name))

            src_lbl = os.path.join(src_labels, label_name)
            if os.path.exists(src_lbl):
                shutil.copyfile(src_lbl, os.path.join(dest_lbl_p, label_name))

        print(f"  -> Prepared {len(img_list)} images and labels in '{split_name}' split.")

    # Load classes from original dataset.yaml if present
    classes = {}
    src_yaml = os.path.join(src_dir, "dataset.yaml")
    if os.path.exists(src_yaml):
        with open(src_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if "names" in data:
                classes = data["names"]

    dest_yaml = {
        "path": os.path.abspath(dest_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes
    }
    with open(os.path.join(dest_dir, "dataset.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(dest_yaml, f, sort_keys=False)

    print(f"\nSuccessfully generated Huggig YOLO dataset at:\n  -> {dest_dir}")

if __name__ == "__main__":
    prepare_splits()

