import cv2
import matplotlib.pyplot as plt

img  = cv2.imread("images/img_001.png")
mask = cv2.imread("masks/img_001.png", 0)

plt.subplot(1,2,1); plt.imshow(img[:,:,::-1]); plt.title("Image")
plt.subplot(1,2,2); plt.imshow(mask, cmap="gray"); plt.title("Mask")
plt.show()