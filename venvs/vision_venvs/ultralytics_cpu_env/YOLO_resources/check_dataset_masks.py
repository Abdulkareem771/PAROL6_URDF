import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path


current_dir = Path(__file__)
project_dir = current_dir.parent.parent

Image_Path = project_dir / "data" / "U-Net_data" / "U-Net_model_v2" / "train" / "200_jpg.rf.ebad15ed1c423acc7c9d72343985dfd8.jpg"
Mask_Path = project_dir / "data" / "U-Net_data" / "U-Net_model_v2" / "train" / "200_jpg.rf.ebad15ed1c423acc7c9d72343985dfd8_mask.png"

img  = cv2.imread(Image_Path)
mask = cv2.imread(Mask_Path, 0)

plt.subplot(1,2,1); plt.imshow(img[:,:,::-1]); plt.title("Image")
plt.subplot(1,2,2); plt.imshow(mask, cmap="gray"); plt.title("Mask")
plt.show()