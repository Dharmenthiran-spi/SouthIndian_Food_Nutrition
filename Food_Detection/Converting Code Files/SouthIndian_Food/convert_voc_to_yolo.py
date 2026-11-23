"""
South Indian Food Dataset -> YOLO Object Detection Converter
Converts Pascal VOC XML annotations from RawData/Version1/SouthIndian_Food
into a complete, verified Ultralytics YOLO dataset in DataSet/Version1 Used Data.
"""

import os
import shutil
import time
import xml.etree.ElementTree as ET
import yaml

def convert_voc_to_yolo(
    src_dir=r"D:\Yolo_Food_Inspection\Food_Detection\DataSet\RawData\Version1\SouthIndian_Food",
    dest_dir=r"D:\Yolo_Food_Inspection\Food_Detection\DataSet\Version1 Used Data"
):
    start_time = time.time()
    splits = ["train", "valid", "test"]

    print(f"Scanning classes across all splits in: {src_dir}")
    classes = set()
    for split in splits:
        split_p = os.path.join(src_dir, split)
        if not os.path.exists(split_p):
            continue
        for f in os.listdir(split_p):
            if f.lower().endswith(".xml"):
                try:
                    tree = ET.parse(os.path.join(split_p, f))
                    root = tree.getroot()
                    for obj in root.findall("object"):
                        cname = obj.find("name").text
                        if cname:
                            classes.add(cname.strip())
                except Exception:
                    pass

    class_list = sorted(list(classes))
    class_to_idx = {name: idx for idx, name in enumerate(class_list)}
    print(f"Discovered {len(class_list)} classes:")
    for idx, cname in enumerate(class_list):
        print(f"  {idx:2d}: {cname}")

    total_converted = 0
    total_boxes = 0

    for split in splits:
        src_split = os.path.join(src_dir, split)
        if not os.path.exists(src_split):
            continue

        # Standard YOLO split names: 'train', 'val', 'test'
        target_split = "val" if split == "valid" else split
        dest_img_split = os.path.join(dest_dir, "images", target_split)
        dest_lbl_split = os.path.join(dest_dir, "labels", target_split)

        os.makedirs(dest_img_split, exist_ok=True)
        os.makedirs(dest_lbl_split, exist_ok=True)

        files = os.listdir(src_split)
        xml_files = [f for f in files if f.lower().endswith(".xml")]
        print(f"\nProcessing '{split}' split ({len(xml_files)} annotations) -> '{target_split}'...")

        split_converted = 0
        split_boxes = 0

        for xml_file in xml_files:
            xml_path = os.path.join(src_split, xml_file)
            stem = os.path.splitext(xml_file)[0]

            # Find matching image file
            img_file = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = f"{stem}{ext}"
                if os.path.exists(os.path.join(src_split, candidate)):
                    img_file = candidate
                    break

            if not img_file:
                continue

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                size = root.find("size")
                if size is not None:
                    width = float(size.find("width").text)
                    height = float(size.find("height").text)
                else:
                    width, height = 640.0, 640.0

                if width <= 0 or height <= 0:
                    width, height = 640.0, 640.0

                yolo_boxes = []
                for obj in root.findall("object"):
                    cname = obj.find("name").text
                    if not cname:
                        continue
                    cname = cname.strip()
                    if cname not in class_to_idx:
                        continue
                    cid = class_to_idx[cname]

                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)

                    # Normalize coordinates
                    box_w = max(0.0001, min(1.0, (xmax - xmin) / width))
                    box_h = max(0.0001, min(1.0, (ymax - ymin) / height))
                    x_center = max(0.0, min(1.0, ((xmin + xmax) / 2.0) / width))
                    y_center = max(0.0, min(1.0, ((ymin + ymax) / 2.0) / height))

                    yolo_boxes.append(f"{cid} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

                # Copy image file
                shutil.copyfile(
                    os.path.join(src_split, img_file),
                    os.path.join(dest_img_split, img_file)
                )

                # Write YOLO format label file (even if empty to indicate negative background)
                label_path = os.path.join(dest_lbl_split, f"{stem}.txt")
                with open(label_path, "w", encoding="utf-8") as lf:
                    if yolo_boxes:
                        lf.write("\n".join(yolo_boxes) + "\n")

                split_converted += 1
                split_boxes += len(yolo_boxes)
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")

        total_converted += split_converted
        total_boxes += split_boxes
        print(f"Finished '{split}': {split_converted} images, {split_boxes} bounding boxes.")

    # Write dataset.yaml
    dataset_yaml = {
        "path": os.path.abspath(dest_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_list),
        "names": {i: name for i, name in enumerate(class_list)}
    }
    yaml_path = os.path.join(dest_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False, default_flow_style=False)

    elapsed = time.time() - start_time
    print(f"\n==========================================")
    print(f"Version 1 Dataset created in {elapsed:.2f} seconds!")
    print(f"Total Images: {total_converted}")
    print(f"Total Bounding Boxes: {total_boxes}")
    print(f"Classes ({len(class_list)}): {class_list}")
    print(f"dataset.yaml saved to: {yaml_path}")
    print(f"==========================================")

if __name__ == "__main__":
    convert_voc_to_yolo()
