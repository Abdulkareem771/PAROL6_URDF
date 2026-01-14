---
name: "👁️ Vision: YOLO Validation"
about: Validate object detection pipeline with generic model
title: "[Vision] Phase 1 - YOLO Validation"
labels: ["vision", "phase-1"]
assignees: []

---

## 🎯 Objective
Setup generic YOLOv8 detection node and validate on live camera feed.

## 📋 Checklist

### Setup
- [ ] Create `parol6_vision` package
- [ ] Install `ultralytics` dependency
- [ ] Verify Kinect v2 connection

### Implementation
- [ ] Implement `yolo_detector_node.py`
- [ ] Subscribe to `/kinect2/sd/image_color_rect`
- [ ] Run inference (generic `yolov8n.pt`)
- [ ] Publish `vision_msgs/Detection2DArray`

### Validation
- [ ] Live detection of common objects (bottle, person)
- [ ] Latency check (< 100ms)
- [ ] Visualization in RViz

## 📚 Resources
- [Vision Integration Guide](../docs/KINECT_INTEGRATION.md)
