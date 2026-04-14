from pathlib import Path
import cv2
import numpy as np
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BUSI"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "yolo_v0"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

CLASS_NAMES = ["tumor"]


def mask_to_bbox(mask: np.ndarray):
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return int(x_min), int(y_min), int(x_max), int(y_max)


def bbox_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox

    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    return x_center, y_center, width, height


def get_image_files(folder: Path):
    return sorted([f for f in folder.glob("*.png") if "_mask" not in f.stem])


def prepare_output_dirs():
    for split in ["train", "val", "test"]:
        (OUTPUT_PATH / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_PATH / split / "labels").mkdir(parents=True, exist_ok=True)


def split_list(items, train_ratio, val_ratio, test_ratio):
    random.shuffle(items)
    total = len(items)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_items = items[:train_end]
    val_items = items[train_end:val_end]
    test_items = items[val_end:]

    return train_items, val_items, test_items


def process_files(image_files, split):
    for img_path in image_files:
        mask_path = img_path.with_name(f"{img_path.stem}_mask.png")

        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        if mask is None:
            print(f"Skipping unreadable mask: {mask_path}")
            continue

        img_h, img_w = image.shape[:2]
        bbox = mask_to_bbox(mask)

        out_img_path = OUTPUT_PATH / split / "images" / img_path.name
        out_label_path = OUTPUT_PATH / split / "labels" / f"{img_path.stem}.txt"

        shutil.copy2(img_path, out_img_path)

        if bbox is None:
            out_label_path.write_text("")
            continue

        x_center, y_center, width, height = bbox_to_yolo(bbox, img_w, img_h)

        class_id = 0
        label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        out_label_path.write_text(label_line)


def create_dataset_yaml():
    yaml_text = f"""path: {OUTPUT_PATH}
train: train/images
val: val/images
test: test/images

names:
  0: {CLASS_NAMES[0]}
"""
    (OUTPUT_PATH / "dataset.yaml").write_text(yaml_text)


def main():
    random.seed(SEED)
    prepare_output_dirs()

    all_train, all_val, all_test = [], [], []

    for class_dir in DATA_PATH.iterdir():
        if not class_dir.is_dir():
            continue

        image_files = get_image_files(class_dir)
        train_files, val_files, test_files = split_list(
            image_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )

        all_train.extend(train_files)
        all_val.extend(val_files)
        all_test.extend(test_files)

    process_files(all_train, "train")
    process_files(all_val, "val")
    process_files(all_test, "test")

    create_dataset_yaml()

    print("YOLO dataset generated successfully!")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()