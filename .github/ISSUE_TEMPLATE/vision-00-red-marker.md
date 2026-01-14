---
name: "👁️ Vision: Red Marker Detection (Fast Track)"
about: Implement color-based contour detection for immediate pipeline testing
title: "[Vision] Phase 1a - Red Marker Detection"
labels: ["vision", "phase-1", "fast-track"]
assignees: []

---

## 🎯 Objective
Develop a simple OpenCV node to detect red markers/lines. This allows testing the **entire system** (Vision → Robot) tomorrow, without waiting for AI models.

## 🏗️ Modular Architecture
This node replaces the "YOLO Node" in the pipeline.
- **Input:** `/kinect2/qhd/image_color_rect`
- **Logic:** HSV Color Thresholding (Red) + Contour Extractor
- **Output:** `vision_msgs/Detection2DArray` (Standard Interface)

## 📋 Checklist

### Implementation
- [ ] Create `red_marker_node.py`
- [ ] Implement HSV Red Threshold (Range: ~0-10, 170-180 Hue)
- [ ] Extract contours (OpenCV `findContours`)
- [ ] Fit bounding box to largest contour
- [ ] **CRITICAL:** Publish as `vision_msgs/Detection2DArray`
  - So the Depth Matcher doesn't know it's not YOLO!

### Validation
- [ ] Draw red line/shape on paper
- [ ] Verify stable detection
- [ ] Check output topic: `ros2 topic echo /vision/detections_2d`

## 📚 Resources
- [OpenCV Contours Tutorial](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
