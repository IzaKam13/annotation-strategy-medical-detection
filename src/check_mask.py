import cv2
from pathlib import Path
import matplotlib.pyplot as plt

img_path = Path("data/raw/BUSI/normal/normal (1).png")
mask_path = Path("data/raw/BUSI/normal/normal (1)_mask.png")

img = cv2.imread(str(img_path))
mask = cv2.imread(str(mask_path), 0)

plt.close('all')  # close previous figures

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Mask")
plt.imshow(mask, cmap="gray")

plt.show()

print("IMAGE:", img_path)
print("MASK:", mask_path)