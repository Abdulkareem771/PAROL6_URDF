# YOLOv8 Instance Segmentation Training Guide 
**Path:** `vision_work/seg-yolo/Training Script/YOLO_Segmentation.ipynb`

This document outlines the pipeline used to train the YOLOv8 Nano instance segmentation model (`yolov8n-seg.pt`) for weld seam detection using Google Colab and the Roboflow platform.

---

## 1. Dataset Generation & Export (Roboflow)
The dataset contains images of workpieces with the delicate weld seams annotated. 
- **Annotation Tool:** Roboflow Smart Polygon / SAM3 (Segment Anything Model 3) automatic labeling was used to generate pixel-perfect masks of the seams.
- **Export Format:** The dataset was exported via the Roboflow API using the **`yolov8`** format. 
- **Under the Hood:** YOLOv8 requires polygon coordinates, not raw PNG pixel masks. By exporting an *Instance Segmentation* project into `yolov8` format, Roboflow automatically traces the SAM3 pixel masks and converts them into the normalized coordinate format (`class x1 y1 x2 y2 ...`) that the Ultralytics trainer demands.

## 2. Dataset Cleaning Strategy
During export, Roboflow can occasionally include stray "Bounding Box" annotations alongside the "Polygon" masks. 
- **The Issue:** If YOLOv8 detects a mixed dataset (e.g., `len(segments) = 120, len(boxes) = 121`), it will immediately drop *all* segmentation masks and attempt to train as an Object Detection model, ultimately causing an `IndexError` crash when calculating the segmentation loss.
- **The Solution:** A Python cleaning script was injected into the pipeline *before* training. It scans all `.txt` label files and deletes any line containing exactly 5 values (a standard bounding box), ensuring only polygons (>5 values) remain.

## 3. Training Optimization (Google Colab T4 GPU)
By default, YOLOv8's `AutoBatch` restricts VRAM usage to ~60% to prevent Out-Of-Memory (OOM) crashes. To train the `yolov8n-seg` model at maximum speed using the full 15GB of the Colab Tesla T4 GPU, we applied the following overrides:

1. **Optimizer Switch (`optimizer='SGD'`):** The default `AdamW` optimizer uses twice the memory of `SGD` because it stores momentum and variance variables for every parameter. Switching to `SGD` frees up gigabytes of VRAM.
2. **Static Batch Sizing (`batch=48`):** With the memory overhead cleared, we force a massive batch size of 48 images at `640x640` resolution. This pegs the GPU memory at ~95% capacity without crashing.
3. **Dataloader Parallelization (`workers=8`, `cache=True`):** Images are cached into Colab's RAM to bypass slow disk reads, and 8 CPU threads are dedicated to applying data augmentations so the GPU never waits for the next batch.

## 4. Evaluation and Export
Upon completion of 50 epochs, the script evaluates the `best.pt` weights.
- **Visualizations:** The generated `results.png` (loss curves, mAP, F1-scores) and `confusion_matrix.png` are displayed inline.
- **Inference:** The model is tested on holdout images from the `/test/` directory to verify the tightness of the polygon masks against the real-world weld seams.
- **Archiving:** The entire `runs/segment/Weld_Segmentation/yolo_seam_seg_fast` directory is compressed into a `.zip` file using `shutil` and downloaded directly to the local machine, preserving the environment, metrics, and weights.
