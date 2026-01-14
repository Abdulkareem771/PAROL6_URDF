---
name: "🧠 AI: Model Training"
about: Train custom YOLOv8 model
title: "[AI] Phase 2 - Model Training"
labels: ["ai", "training"]
assignees: []

---

## 🎯 Objective
Train a custom model to detect workpieces and seams.

## 📋 Checklist

### Training
- [ ] Setup training environment (GPU)
- [ ] Config `data.yaml`
- [ ] Run training (100 epochs)
- [ ] Evaluate mAP (Target: > 0.9)

### Deployment
- [ ] Export `best.pt`
- [ ] Test with `parol6_vision/yolo_detector_node`
- [ ] Verify realtime performance

## 📚 Resources
- [Ultralytics Docs](https://docs.ultralytics.com)
