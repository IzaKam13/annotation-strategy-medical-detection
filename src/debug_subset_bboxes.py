from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BUSI"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "debug_bboxes"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def mask_to_bbox(mask: np.ndarray):
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return int(x_min), int(y_min), int(x_max), int(y_max)


def get_image_files(folder: Path):
    return [
        f for f in folder.glob("*.png")
        if "_mask" not in f.stem
    ]


def process_subset(max_per_class=5):
    for class_dir in DATA_PATH.iterdir():
        if not class_dir.is_dir():
            continue

        image_files = sorted(get_image_files(class_dir))[:max_per_class]

        for img_path in image_files:
            mask_path = img_path.with_name(f"{img_path.stem}_mask.png")

            print(f"\nProcessing: {img_path.name}")
            print(f"Mask: {mask_path.name}")

            image = cv2.imread(str(img_path))
            if image is None:
                print("Could not read image")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print("Could not read mask")
                continue

            bbox = mask_to_bbox(mask)
            print("Bounding box:", bbox)

            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                title = f"{class_dir.name}: bbox"
            else:
                title = f"{class_dir.name}: empty mask"

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image)
            ax.set_title(f"{title}\n{img_path.name}")
            ax.axis("off")

            save_name = f"{class_dir.name}_{img_path.stem}.png"
            save_path = OUTPUT_PATH / save_name
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            print(f"Saved: {save_path}")


if __name__ == "__main__":
    process_subset(max_per_class=5)