from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data/processed/yolo_v0"

def yolo_to_bbox(yolo_line, img_w, img_h):
    class_id, x_center, y_center, width, height = map(float, yolo_line.split())

    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h

    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    return int(class_id), x_min, y_min, x_max, y_max


def visualize_sample(split="train", max_samples=5):
    img_dir = DATASET_PATH / split / "images"
    label_dir = DATASET_PATH / split / "labels"

    image_files = list(img_dir.glob("*.png"))
    image_files = random.sample(image_files, min(max_samples, len(image_files)))

    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        print(f"\nImage: {img_path.name}")

        if label_path.exists():
            lines = label_path.read_text().strip().splitlines()

            if lines:
                for line in lines:
                    class_id, x_min, y_min, x_max, y_max = yolo_to_bbox(line, w, h)
                    print("BBox:", (x_min, y_min, x_max, y_max))

                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            else:
                print("Empty label (normal case)")
        else:
            print("No label file")

        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(img_path.name)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    visualize_sample("train", max_samples=5)