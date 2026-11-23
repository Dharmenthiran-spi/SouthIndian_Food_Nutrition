"""
Khana Dataset -> YOLO Object Detection Converter
Converts Khana single-dish classification dataset to YOLO Object Detection format
with bounding boxes covering the main dish area [class_id, 0.5, 0.5, 0.95, 0.95].
"""

import os
import shutil
import yaml

def convert_to_detection(
    src_dir=r"D:\Yolo_Food_Inspection\Food_Detection\DataSet\RawData\Version1\Khana_DataSet",
    dest_dir=r"D:\Yolo_Food_Inspection\Food_Detection\DataSet\Version1 Used Data\Khana_YOLO_Detect"
):
    labels_file = os.path.join(src_dir, "labels.txt")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"labels.txt not found in {src_dir}")

    with open(labels_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    print(f"Loaded {len(classes)} classes from {labels_file}")

    for split in ["train", "val"]:
        src_split = os.path.join(src_dir, split)
        if not os.path.exists(src_split):
            continue

        dest_img_split = os.path.join(dest_dir, "images", split)
        dest_lbl_split = os.path.join(dest_dir, "labels", split)

        os.makedirs(dest_img_split, exist_ok=True)
        os.makedirs(dest_lbl_split, exist_ok=True)

        class_names = [d for d in os.listdir(src_split) if os.path.isdir(os.path.join(src_split, d))]
        print(f"Processing '{split}' split ({len(class_names)} classes)...")

        for class_name in class_names:
            class_folder = os.path.join(src_split, class_name)
            if class_name not in class_to_idx:
                continue
            cid = class_to_idx[class_name]

            for fname in os.listdir(class_folder):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    continue
                # Copy image
                src_img = os.path.join(class_folder, fname)
                dest_img = os.path.join(dest_img_split, fname)
                shutil.copyfile(src_img, dest_img)

                # Create YOLO bounding box label: class_idx x_center y_center width height
                stem = os.path.splitext(fname)[0]
                label_path = os.path.join(dest_lbl_split, f"{stem}.txt")
                with open(label_path, "w", encoding="utf-8") as lf:
                    lf.write(f"{cid} 0.500000 0.500000 0.950000 0.950000\n")

    # Generate dataset.yaml
    data_yaml = {
        "path": os.path.abspath(dest_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": {i: name for i, name in enumerate(classes)}
    }
    with open(os.path.join(dest_dir, "dataset.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Successfully generated YOLO Detection dataset at:\n  -> {dest_dir}")

if __name__ == "__main__":
    convert_to_detection()
