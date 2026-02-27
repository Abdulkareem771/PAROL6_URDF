# YOLO Object Detection and Cropping

## Overview
This script detects objects in images using your trained YOLO11n model, crops each detected object, and saves them to a `ROI_images` folder.

## Files Created
- **`yolo_crop_objects.py`** - Main script with detection and cropping functions
- **`example_crop_usage.py`** - Simple usage examples

## Features
✅ Detects objects using your trained YOLO11n model  
✅ Crops each detected object from the original image  
✅ Saves cropped images with informative filenames  
✅ Supports single image or batch processing  
✅ Configurable confidence threshold, padding, and IoU threshold  
✅ Automatic folder creation  

## Quick Start

### 1. Process a Single Image
```python
from yolo_crop_objects import process_single_image

process_single_image(
    image_path="path/to/your/image.jpg",
    output_folder="ROI_images"
)
```

### 2. Process All Images in a Folder
```python
from yolo_crop_objects import process_image_folder

process_image_folder(
    input_folder="path/to/your/images",
    output_folder="ROI_images"
)
```

## Output Format
Cropped images are saved with the following naming convention:
```
{original_name}_obj{number}_{class_name}_conf{confidence}.jpg
```

**Example:**
```
workspace_image_obj1_piece_conf0.87.jpg
workspace_image_obj2_piece_conf0.92.jpg
```

## Configuration Parameters

### In `yolo_crop_objects.py`:
- **`MODEL_PATH`**: Path to your trained model (`best.pt`)
- **`CONF_THRESHOLD`**: Confidence threshold (default: 0.25)
- **`IOU_THRESHOLD`**: IoU threshold for NMS (default: 0.7)
- **`OUTPUT_FOLDER`**: Output directory (default: "ROI_images")
- **`PADDING`**: Padding around bounding boxes in pixels (default: 10)

### Adjust these based on your needs:
- **Higher confidence** (e.g., 0.5) → Fewer detections, more accurate
- **Lower confidence** (e.g., 0.15) → More detections, may include false positives
- **More padding** (e.g., 20) → Include more context around objects
- **No padding** (0) → Exact bounding box crop

## Usage Examples

### Example 1: Basic Usage
```python
python yolo_crop_objects.py
# (Modify the script to uncomment your desired example)
```

### Example 2: From Another Script
```python
from yolo_crop_objects import process_single_image

# Process a workspace image
process_single_image(
    image_path="/path/to/workspace_capture.jpg",
    output_folder="detected_pieces"
)
```

### Example 3: Custom Settings
```python
from yolo_crop_objects import process_image_folder

# Process multiple images with custom settings
process_image_folder(
    input_folder="/path/to/robot/workspace/images",
    output_folder="ROI_images",
    conf_threshold=0.5,  # Higher confidence
    padding=15           # More padding
)
```

## What the Script Does

### Step 1: Load Model
Loads your trained YOLO11n model from:
```
/home/osama/Desktop/PAROL6_URDF/vision_work/yolo_training/experiment_1/weights/best.pt
```

### Step 2: Detect Objects
Runs detection on the input image(s) with specified confidence threshold.

### Step 3: Crop Objects
For each detected object:
- Extracts bounding box coordinates
- Adds optional padding
- Crops the region from the original image

### Step 4: Save Cropped Images
Saves each cropped object with:
- Original image name
- Object number
- Class name
- Confidence score

## Output Structure
```
ROI_images/
├── image1_obj1_piece_conf0.89.jpg
├── image1_obj2_piece_conf0.76.jpg
├── image2_obj1_piece_conf0.92.jpg
└── ...
```

## Function Reference

### `process_single_image()`
Process a single image and save cropped objects.

**Parameters:**
- `image_path` (str): Path to input image
- `model_path` (str): Path to YOLO model weights
- `output_folder` (str): Output directory for cropped images
- `conf_threshold` (float): Confidence threshold (0-1)
- `iou_threshold` (float): IoU threshold for NMS (0-1)
- `padding` (int): Padding around bounding boxes in pixels

### `process_image_folder()`
Process all images in a folder.

**Parameters:** Same as `process_single_image()` except:
- `input_folder` (str): Path to folder containing images (replaces `image_path`)

**Supported image formats:**
- JPG/JPEG
- PNG
- BMP
- TIFF
- WebP

## Tips

### Finding the Right Confidence Threshold
Check your training results in `BoxF1_curve.png` to find the optimal confidence threshold that maximizes F1-score.

### Padding Recommendations
- **0px**: Exact object boundary
- **5-10px**: Slight context (recommended for tight crops)
- **15-20px**: More context (useful for further processing)
- **30+px**: Include surrounding area

### Performance
- Processing time depends on:
  - Image size
  - Number of objects
  - Hardware (CPU vs GPU)
- Use GPU for faster processing on large batches

## Troubleshooting

**No objects detected?**
- Check if confidence threshold is too high
- Verify the model path is correct
- Ensure input image quality is good

**Too many false positives?**
- Increase confidence threshold
- Fine-tune your model with more data

**Cropped images too tight?**
- Increase the `padding` parameter

## Next Steps
1. Modify the paths in `yolo_crop_objects.py` or use function parameters
2. Run the script on your workspace images
3. Check the `ROI_images` folder for results
4. Adjust confidence threshold based on results

---

**Model Used:** YOLO11n trained on robot workspace pieces  
**Model Performance:** mAP50-95: 0.60-0.65 | mAP50: 0.92-0.97
