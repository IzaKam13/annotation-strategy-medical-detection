from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from mask_to_bbox import mask_to_bbox

img_path = Path("data/raw/BUSI/malignant/malignant (11).png")
mask_path = Path("data/raw/BUSI/malignant/malignant (11)_mask.png")
save_path = Path("outputs/debug_bbox.png")

print("=== NEW RUN ===")
print("IMAGE:", img_path)
print("MASK:", mask_path)
print("IMAGE EXISTS:", img_path.exists())
print("MASK EXISTS:", mask_path.exists())

image = cv2.imread(str(img_path))
if image is None:
    raise ValueError(f"Could not read image: {img_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise ValueError(f"Could not read mask: {mask_path}")

bbox = mask_to_bbox(mask)
print("Bounding box:", bbox)

if bbox is not None:
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    title = "With bounding box"
else:
    title = "No object (empty mask)"

plt.close("all")
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image)
ax.set_title(f"{title}\n{img_path.name}")
ax.axis("off")

save_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(save_path, bbox_inches="tight", dpi=150)
print(f"Saved figure to: {save_path.resolve()}")

plt.show()
plt.close(fig)