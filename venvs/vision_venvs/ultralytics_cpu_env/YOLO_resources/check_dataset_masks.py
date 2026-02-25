import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path


current_dir = Path(__file__)
project_dir = current_dir.parent.parent

Image_Path = project_dir / "data" / "U-Net_data" / "U-Net_model.v1i.png-mask-semantic" / "train" / "N_1_jpg.rf.51d3c1c9a0bd18dad0356a615ac5a7bf.jpg"
Mask_Path = project_dir / "data" / "U-Net_data" / "U-Net_model.v1i.png-mask-semantic" / "train" / "N_1_jpg.rf.51d3c1c9a0bd18dad0356a615ac5a7bf_mask.png"

img  = cv2.imread(Image_Path)
mask = cv2.imread(Mask_Path, 0)

plt.subplot(1,2,1); plt.imshow(img[:,:,::-1]); plt.title("Image")
plt.subplot(1,2,2); plt.imshow(mask, cmap="gray"); plt.title("Mask")
plt.show()