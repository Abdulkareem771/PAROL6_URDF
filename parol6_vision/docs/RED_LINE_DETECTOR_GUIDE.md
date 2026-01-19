# Red Line Detector Node - Developer Guide

## 1. Overview
The `red_line_detector` node is the primary perception component of the PAROL6 vision pipeline. It is responsible for stabilizing identifying weld seams (marked by red lines) in RGB camera images and extracting their 2D geometry for downstream 3D reconstruction.

**Node Name:** `red_line_detector`
**Package:** `parol6_vision`
**Source:** `parol6_vision/red_line_detector.py`

## 2. Architecture & Pipeline

The detections follow a sequential processing pipeline designed for robustness against noise and lighting variations.

### Process Flow
1. **Input**: Rectified RGB Image (`sensor_msgs/Image`)
2. **Preprocessing**: 
   - Gaussian Blur (optional)
   - Color Space Conversion (BGR → HSV)
3. **Segmentation**: 
   - Thresholding on HSV Red Spectrum (Wraps around 0/180)
   - Initial Mask Generation
4. **Morphology**:
   - `Erosion`: Removes salt noise and small artifacts.
   - `Dilation`: Connects fragmented line segments.
5. **Geometry Extraction**:
   - `Skeletonization`: reduces the thick mask to a 1-pixel wide centerline.
   - `Contour Extraction`: Identifies connected components in the skeleton.
6. **Filtering**:
   - Length Check: Discards contours shorter than `min_line_length`.
   - Confidence Scoring: Evaluates continuity and signal-to-noise ratio.
7. **Semantic Packaging**:
   - Orders points along the principal direction.
   - Simplifies polyline (Douglas-Peucker).
   - Wraps data into `parol6_msgs/WeldLine`.
8. **Output**: `parol6_msgs/WeldLineArray`

## 3. Algorithm Details

### 3.1 Color Segmentation (Red Spectrum)
Red is unique in the HSV space because it wraps around the hue channel (0-180). The detector uses two separate ranges to capture strong reds:
- **Range 1:** Hue 0–10 (pure red/orange-red)
- **Range 2:** Hue 170–180 (purple-red)

Parameter names: `hsv_lower_1/2`, `hsv_upper_1/2`.

### 3.2 Skeletonization over Contouring
Unlike generic object detectors that find bounding boxes, welding requires the **exact centerline**. 
- We use `skimage.morphology.skeletonize` on the binary mask.
- This creates a homotopic thinning of the shape, preserving the topological properties (connectivity, holes) while reducing width to 1 pixel.

### 3.3 Confidence Metric
Each detected line is assigned a confidence score, calculated as:
```python
Confidence = (Retention Ratio) * (Continuity Score)
```
- **Retention Ratio**: `Mask Area (After Morphology) / Mask Area (Raw)`. High retention means the object is solid and not noisy.
- **Continuity Score**: Measures how "straight" or smooth the line is (currently based on simplification stability).

## 4. ROS API

### Subscribed Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/kinect2/qhd/image_color_rect` | `sensor_msgs/Image` | Rectified RGB input stream |

### Published Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/vision/weld_lines_2d` | `parol6_msgs/WeldLineArray` | Semantic weld line detections |
| `/red_line_detector/debug_image` | `sensor_msgs/Image` | Visualization of detections overlaid on input |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_line_length` | int | 50 | Minimum pixel length to be considered a line |
| `min_confidence` | float | 0.5 | Threshold for publishing a line |
| `morphology_kernel_size` | int | 5 | Size of structuring element for erosion/dilation |

## 5. Development & Troubleshooting

### Simulating/Testing
You can run the detector in isolation using the provided test launch file:
```bash
ros2 launch parol6_vision red_detector_only.launch.py
```

### Common Issues
**"Extracted 0 contours" despite visible red:**
- Check `min_line_length`. If the skeleton is fragmented, individual segments might be too short.
- Verify `hsv` ranges manually using an HSV tuner tool if lighting conditions change significantly.

**"CV Bridge Conversion Failed":**
- Ensure `sensor_msgs` has correct encoding (usually `bgr8` or `rgb8`). The node explicitly requests `bgr8`.

**Empty Output (No lines published):**
- Check `/red_line_detector/debug_image` in RViz.
- If mask is empty: Tuning issue.
- If mask has pixels but no lines: Filtering issue (Confidence or Length).
