# Deep Residual UNet (ResUNet) Pipeline
## For Automated Weld Seam Detection

This folder contains the automated pipeline for extracting the geometric 1-pixel centerline of a weld seam from an RGB image, removing the need for manually drawing red lines.

### Table of Contents
1. [Why ResUNet?](#why-resunet)
2. [Colab Training Guide](#colab-training-guide)
3. [Local Testing Guide](#local-testing-guide)

---

## 1. Why ResUNet?

The initial attempt (`UNet_training.py`) struggled to perfectly map the thin, 1-pixel-wide line of the weld seam. The standard UNet architecture was too shallow (only 3 encoder blocks) and suffered from vanishing gradients on such subtle geometric features.

**ResUNet (Deep Residual UNet)** solves this:
- **Depth**: It is much deeper, downsampling to `1/8th` resolution (512 -> 64), giving it a large receptive field.
- **Residual Blocks**: Each convolution block uses a shortcut identity connection. These "skip" paths let the gradients flow continuously to early layers without getting lost.
- **BCE + Dice Loss**: The model is penalized for missing the thin line (via Dice coefficient), actively combating the class imbalance where 99% of the image is background.

---

## 2. Colab Training Guide

Because `ResUNet` requires GPU memory, training must happen on Google Colab (using an NVIDIA T4 or A100). The dataset is automatically pulled from your Roboflow workspace.

### Steps to Train:
1. Open Google Colab and upload the `ResUNet_training_colab.ipynb` file from this directory.
2. In Colab, go to **Runtime > Change runtime type** and select **T4 GPU**.
3. Run the cells sequentially.
4. The notebook will use your API key (`rf_HrQ6aUiVG3PmJKFOe8pmmXpxol62`) to automatically download the V4 `u-net_model` semantic segmentation dataset. 
5. The notebook tracks `Val Loss` and `Val Dice Coeff` visually using matplotlib.
6. Once complete, it will automatically download `best_resunet_seam.pth` to your host computer.

---

## 3. Local Testing Guide

Once you download the `best_resunet_seam.pth` file from Colab, place it in this directory (`YOLO_resources/`). 

We've provided a simple testing tool that handles preprocessing, inference, and skeletonization (extracting the 1-pixel centerline for the robot).

### Usage:
1. Activate your python environment:
   ```bash
   source /home/kareem/Desktop/PAROL6_URDF/venvs/vision_venvs/ultralytics_cpu_env/bin/activate
   ```
2. Run the tester on an image. The script accepts a single image (ideally cropped from YOLO Stage 1).
   ```bash
   python weld_seam_tester.py --image test_images/sample1.jpg --model best_resunet_seam.pth
   ```

### Output
The script generates a 3-panel visualization showing:
1. The **Original Image**
2. The **Predicted Binary Mask** (thick white line)
3. The **Skeleton Centerline Overlay** (thin green line, which mimics the robot's required path vector)

It also automatically saves the overlays to a `test_results/` folder so you can review them after.
