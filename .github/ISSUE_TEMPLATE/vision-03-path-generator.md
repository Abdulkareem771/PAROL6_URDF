---
name: "👁️ Vision: Path Generator"
about: Turn 3D detections into robot paths
title: "[Vision] Phase 3 - Path Generator"
labels: ["vision", "phase-3", "path-planning"]
assignees: []

---

## 🎯 Objective
Take a list of 3D points (from Depth Matcher) and generate a smooth robot path.

## 🏗️ Modular Architecture
- **Input:** `vision_msgs/Detection3DArray` (from Depth Matcher)
- **Output:** `nav_msgs/Path` or `geometry_msgs/PoseArray`

## 📋 Checklist

### Logic
- [ ] Subscribe to `/vision/detections_3d`
- [ ] **Ordering:** Sort points to form a logical line (e.g., Nearest Neighbor)
- [ ] **Smoothing:** (Optional) Apply B-Spline or simple averaging
- [ ] **Orientation:** Align Z-axis with surface normal (or fixed down)
- [ ] Publish visualization to RViz

### Validation
- [ ] Visualize path in RViz
- [ ] Verify points are ordered correctly (not jumping around)

## 🔄 Integration
- This node works with **Red Marker** OR **YOLO** input seamlessly!
