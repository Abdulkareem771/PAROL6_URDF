---
name: "👁️ Vision: Depth Matching"
about: Implement 2D-to-3D projection using generic depth
title: "[Vision] Phase 2 - Depth Matching Node"
labels: ["vision", "phase-2", "core-infra"]
assignees: []

---

## 🎯 Objective
Create the core **Model-Agnostic** depth matching node that projects any 2D bounding box to 3D space.

## 📋 Checklist

### Implementation
- [ ] Create `depth_matcher_node.py`
- [ ] Subscribe to `/yolo/detections` and `/kinect2/sd/points`
- [ ] Implement median depth filtering within bbox
- [ ] Publish `vision_msgs/Detection3DArray`

### Testing
- [ ] Place known object at 0.5m, 1.0m, 1.5m
- [ ] Measure error vs ruler
- [ ] Success Criteria: Accuracy ±10mm

## ⚠️ Constraint
MUST work with ANY YOLO output (do not hardcode classes).

## 📚 Resources
- [Parallel Work Guide](../docs/archived/PARALLEL_WORK_GUIDE.md)
