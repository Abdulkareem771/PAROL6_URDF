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

## 3. Local GUI Testing Guide

Once you download the `best_resunet_seam.pth` file from Colab, place it in this directory (`vision_work/resunet_training/`).

We have provided a fully interactive **WeldVision GUI** (`weld_seam_gui.py`) that handles inference, visualization, and centerline extraction. It makes tuning the threshold for the robotic path planner incredibly easy.

### Usage:
1. Ensure you have the required dependencies:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu scikit-image matplotlib opencv-python Pillow
   ```
2. Launch the tester GUI:
   ```bash
   python3 weld_seam_gui.py
   ```

### GUI Features
- **Browse Image**: Opens a standard OS file dialog to select any image from your computer.
- **Threshold Slider**: Drag the slider (0.01 - 0.99) to instantly see the mask tighten or expand live on the image, without re-running the model.
- **View Modes**: Instantly toggle the main canvas between:
  - **Original**: The raw input image.
  - **Overlay**: Translates the mask into a thick red seam ([255, 50, 50]) over the image.
  - **Mask**: The raw black-and-white binary heatmap.
  - **Heat Map**: A colour-graded probability map.
  - **Skeleton**: A 1-pixel green line ([0, 255, 0]) representing the actual vector the robot will follow.
- **Save Result**: Instantly export the current visual view to a `.png` file.

