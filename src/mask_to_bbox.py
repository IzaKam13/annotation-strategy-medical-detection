from pathlib import Path
import cv2
import numpy as np

print("=== RUNNING mask_to_bbox.py ===")
print("THIS FILE:", Path(__file__).resolve())
print("CURRENT WORKDIR:", Path.cwd())

def mask_to_bbox(mask: np.ndarray):
    """
    Return bounding box coordinates (x_min, y_min, x_max, y_max)
    from a binary mask. Returns None if mask is empty.
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return int(x_min), int(y_min), int(x_max), int(y_max)


if __name__ == "__main__":
    mask_path = Path("data/raw/BUSI/malignant/malignant (11)_mask.png")

    print("MASK PATH:", mask_path)
    print("ABS MASK PATH:", mask_path.resolve())
    print("MASK EXISTS:", mask_path.exists())

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    print("MASK SHAPE:", None if mask is None else mask.shape)

    bbox = mask_to_bbox(mask)
    print("Bounding box:", bbox)