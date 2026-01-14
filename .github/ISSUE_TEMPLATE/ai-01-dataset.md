---
name: "🧠 AI: Dataset Collection"
about: Collect and annotate images for custom model
title: "[AI] Phase 1 - Dataset Generation"
labels: ["ai", "data"]
assignees: []

---

## 🎯 Objective
Create a high-quality dataset of workpieces for seam detection.

## 📋 Checklist

### Collection
- [ ] Capture 50+ images of workpieces
- [ ] Vary lighting conditions
- [ ] Vary distances (0.3m - 1.0m)
- [ ] Vary angles

### Annotation
- [ ] Upload to RoboFlow
- [ ] Label class: `workpiece`
- [ ] Label class: `seam` (polyline or segmentation)
- [ ] Export dataset (YOLOv8 format)

## 📚 Resources
- [RoboFlow Guide](https://roboflow.com)
