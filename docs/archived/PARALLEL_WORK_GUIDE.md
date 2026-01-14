# Parallel Development Strategy - Vision Team

## 🎯 The Problem

**Team Structure:**
- **Teammate A**: Training custom YOLO model for seam detection (2-4 weeks)
- **Teammate B**: Available to code NOW (your colleague from plan)
- **You**: System integration & coordination

**Issue**: Don't want Teammate B waiting idle while model trains

---

## ✅ Solution: Modular Pipeline with Generic YOLO

### Critical Review of Your Suggestion

**Your Idea**: Use generic YOLO + depth matching as reusable foundation  
**Assessment**: ⭐⭐⭐⭐⭐ **EXCELLENT**

**Why it works:**
1. ✅ **Model-agnostic**: Works with ANY YOLO (generic → custom)
2. ✅ **Testable NOW**: Validate pipeline before custom model ready
3. ✅ **Reusable**: Infrastructure stays, just swap detector
4. ✅ **Parallel work**: No dependencies between teammates

**Refinement Needed:**
- Break into 3 separate nodes (not monolithic)
- Define clear interfaces (ROS messages)
- Add validation/visualization tools

---

## 🏗️ Modular Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Node 1: YOLO Detector                                      │
│  - Uses generic yolov8n.pt (pre-trained)                    │
│  - Detects ANY object (person, bottle, etc.)                │
│  - Publishes: vision_msgs/Detection2DArray                  │
│  - SWAPPABLE: Replace with custom model later               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 2: Depth Matcher ← TEAMMATE B BUILDS THIS            │
│  - Input: Detection2DArray + PointCloud2                    │
│  - Samples depth within each bbox                           │
│  - Computes 3D center position                              │
│  - Publishes: Detection3DArray                              │
│  - REUSABLE: Model-agnostic                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 3: Seam Extractor ← WAIT FOR CUSTOM MODEL            │
│  - Input: Detection3D (workpiece) + RGB + Depth             │
│  - Extracts seam from detected workpiece                    │
│  - Publishes: nav_msgs/Path (seam waypoints)                │
│  - MODEL-SPECIFIC: Uses custom YOLO output                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Parallel Work Plan

### Week 1-2: Setup & Node 1

**Teammate B:**
- Setup vision venv
- Create `yolo_detector.py` node
- Test with generic YOLOv8

**Teammate A:**
- Collect workpiece images
- Start annotation (RoboFlow/labelImg)

**You:**
- Repository setup
- ROS package structure
- Integration testing plan

---

### Week 3-4: Node 2 (Core Infrastructure) ← **MAIN FOCUS**

**Teammate B: Build Depth Matcher**

**Deliverable**: `depth_matcher_node.py`

```python
#!/usr/bin/env python3
"""
Generic Depth Matcher - Works with ANY YOLO model

Input:
  - /yolo/detections (vision_msgs/Detection2DArray)
  - /kinect2/sd/points (sensor_msgs/PointCloud2)

Output:
  - /vision/detections_3d (vision_msgs/Detection3DArray)
  - /vision/markers (visualization_msgs/MarkerArray) # For RViz
"""

import rclpy
from vision_msgs.msg import Detection2DArray, Detection3DArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class DepthMatcherNode:
    def __init__(self):
        # Subscribe to detections and point cloud
        self.det_sub = self.create_subscription(
            Detection2DArray, '/yolo/detections',
            self.detection_callback, 10
        )
        self.pc_sub = self.create_subscription(
            PointCloud2, '/kinect2/sd/points',
            self.pointcloud_callback, 10
        )
        
        # Publish 3D detections
        self.det3d_pub = self.create_publisher(
            Detection3DArray, '/vision/detections_3d', 10
        )
        
        self.latest_pointcloud = None
    
    def pointcloud_callback(self, msg):
        """Store latest point cloud"""
        self.latest_pointcloud = msg
    
    def detection_callback(self, msg):
        """Match 2D detections to 3D"""
        if self.latest_pointcloud is None:
            return
        
        detections_3d = Detection3DArray()
        detections_3d.header = msg.header
        
        for det_2d in msg.detections:
            # Extract bbox
            bbox = det_2d.bbox
            center_x = bbox.center.x
            center_y = bbox.center.y
            size_x = bbox.size_x
            size_y = bbox.size_y
            
            # Sample points in bbox
            points_in_bbox = self.sample_bbox_points(
                center_x, center_y, size_x, size_y
            )
            
            # Compute 3D center
            if len(points_in_bbox) > 0:
                center_3d = self.compute_3d_center(points_in_bbox)
                
                # Create Detection3D
                det_3d = Detection3D()
                det_3d.header = det_2d.header
                det_3d.results = det_2d.results  # Copy class/confidence
                det_3d.bbox.center.position = center_3d
                
                detections_3d.detections.append(det_3d)
        
        # Publish
        self.det3d_pub.publish(detections_3d)
    
    def sample_bbox_points(self, cx, cy, sx, sy):
        """Extract point cloud points within bbox"""
        # TODO: Implement
        # 1. Iterate through point cloud
        # 2. Check if (x,y) in image falls in bbox
        # 3. Return list of 3D points
        pass
    
    def compute_3d_center(self, points):
        """Compute median/mean of 3D points"""
        # TODO: Use median for robustness
        pass
```

**Test Cases:**
1. Detect bottle → Get 3D position → Validate with ruler
2. Detect multiple objects → Verify each gets correct depth
3. Edge cases: Partial occlusion, invalid depth

**Deliverable Timeline**: 2 weeks

---

**Teammate A (Parallel):**
- Continue data collection (target: 100 images)
- Annotation complete
- Start YOLO training

**You:**
- Create visualization tools (RViz config)
- Setup testing framework
- Document interfaces

---

### Week 5-6: Integration & Testing

**Teammate B:**
- Refine depth matcher based on testing
- Add edge case handling
- Create unit tests

**Teammate A:**
- Finish YOLO training
- Export model
- Test accuracy

**Integration Test** (Both):
```bash
# Test with generic YOLO
ros2 launch parol6_vision test_pipeline.launch.py model:=yolov8n.pt

# Swap to custom model (no code changes!)
ros2 launch parol6_vision test_pipeline.launch.py model:=custom_seam.pt
```

---

### Week 7+: Seam Extraction (Node 3)

**Now that custom model is ready:**
- Node 2 (depth matcher) stays as-is
- Add Node 3 for seam-specific processing
- Full pipeline ready

---

## 🔑 Key Interfaces (ROS Messages)

### Interface 1: YOLO → Depth Matcher
```yaml
# vision_msgs/Detection2DArray
header:
  stamp: <time>
  frame_id: "camera_rgb_optical_frame"
detections:
  - results:
      - id: "person"  # or "workpiece" with custom model
        score: 0.92
    bbox:
      center: {x: 320, y: 240}
      size_x: 120
      size_y: 200
```

### Interface 2: Depth Matcher → Seam Extractor
```yaml
# vision_msgs/Detection3DArray
header:
  stamp: <time>
  frame_id: "camera_rgb_optical_frame"
detections:
  - results:
      - id: "workpiece"
        score: 0.95
    bbox:
      center:
        position: {x: 0.5, y: 0.0, z: 1.2}  # 3D in camera frame
      size: {x: 0.3, y: 0.4, z: 0.05}
```

### Interface 3: Seam Extractor → Path Planner
```yaml
# nav_msgs/Path
header:
  frame_id: "base_link"
poses:
  - pose:
      position: {x: 0.4, y: 0.1, z: 0.3}
  - pose:
      position: {x: 0.5, y: 0.1, z: 0.3}
  # ... 100 points along seam
```

---

## 🧪 Testing Strategy (Parallel Work)

### Phase 1: Test Depth Matcher with Objects
```bash
# Place bottle on table
# Run generic YOLO + depth matcher
# Measure actual distance with ruler
# Compare with detected 3D position
# Target: ±10mm accuracy
```

### Phase 2: Validate with Known Positions
```python
# test_depth_accuracy.py
known_positions = [
    (0.5, 0.0, 1.0),  # Object 1
    (0.6, 0.2, 0.8),  # Object 2
]

for i, known_pos in enumerate(known_positions):
    detected = detections_3d[i].bbox.center.position
    error = distance(known_pos, detected)
    assert error < 0.01  # 1cm threshold
```

### Phase 3: Swap Models Seamlessly
```bash
# No code changes needed!
# Just change model file:
ros2 param set /yolo_detector model_path /path/to/custom_model.pt
```

---

## 📊 Work Assignments

| Task | Owner | Duration | Dependency |
|------|-------|----------|------------|
| Setup venv | Teammate B | Day 1 | None |
| YOLO Detector Node | Teammate B | Week 1 | venv |
| Data Collection | Teammate A | Week 1-3 | None |
| Annotation | Teammate A | Week 2-3 | Data |
| **Depth Matcher** | **Teammate B** | **Week 3-4** | **YOLO Node** |
| Model Training | Teammate A | Week 4-5 | Annotation |
| Integration Test | Both | Week 5 | Depth Matcher + Model |
| Seam Extractor | Teammate B | Week 6-7 | Custom Model |
| Full Pipeline | You | Week 7-8 | All components |

**Critical Path**: Depth Matcher (Teammate B, Week 3-4)

---

## ✅ Advantages of This Approach

1. **No Waiting**:
   - Teammate B codes NOW with generic YOLO
   - Teammate A trains model in parallel
   - Merge at Week 5

2. **Early Validation**:
   - Test depth matching before custom model ready
   - Find issues early
   - Iterate quickly

3. **Reusability**:
   - Depth matcher works with ANY detector
   - Can test with different YOLO models
   - Future-proof for model updates

4. **Risk Mitigation**:
   - If custom model fails, generic still works
   - Have fallback option
   - Infrastructure already validated

5. **Learning**:
   - Teammate B learns full stack
   - Can contribute to model evaluation
   - Understanding of complete pipeline

---

## 🚀 Quick Start for Teammate B

### Day 1: Setup
```bash
cd PAROL6_URDF
./setup_vision_env.sh
source venv_vision/bin/activate
python -c "from ultralytics import YOLO; print('YOLO ready!')"
```

### Day 2-3: YOLO Detector Node
```bash
# Create node
ros2 pkg create parol6_vision --build-type ament_python

# Implement yolo_detector.py
# Test: Detect objects in test images
```

### Week 2: Test with Kinect
```bash
# Verify Kinect works
ros2 topic echo /kinect2/sd/image_color_rect

# Run detector on live feed
ros2 run parol6_vision yolo_detector

# Visualize in RViz
```

### Week 3-4: Depth Matcher (Main Task)
```bash
# Implement depth_matcher_node.py
# Test accuracy with known objects
# Refine based on results
```

---

## 📝 Teammate B Checklist

- [ ] Setup vision venv
- [ ] Test YOLO with sample image
- [ ] Create `yolo_detector.py` node
- [ ] Test with Kinect live feed
- [ ] Implement `depth_matcher_node.py`
- [ ] Unit tests for depth matcher
- [ ] Integration test with generic YOLO
- [ ] Accuracy validation (±10mm)
- [ ] Documentation & code review
- [ ] Ready for custom model swap

---

## 🎯 Success Criteria

**Week 4 Demo** (Before custom model ready):
- ✅ Generic YOLO detects objects (bottle, box, etc.)
- ✅ Depth matcher computes 3D positions
- ✅ Accuracy validated: ±10mm
- ✅ Visualized in RViz
- ✅ 5Hz processing rate

**Week 7 Demo** (With custom model):
- ✅ Custom YOLO detects workpieces
- ✅ Same depth matcher (no modifications!)
- ✅ Seam extractor finds weld lines
- ✅ End-to-end: Image → Path

---

## 📞 Coordination Points

**Weekly Sync** (You + Both Teammates):
- Week 1: Setup complete? Any blockers?
- Week 2: YOLO working? Data collection on track?
- Week 3: Depth matcher progress? Model training status?
- Week 4: Ready for integration test?
- Week 5: **Integration milestone** - Test with both models

**Daily Standups** (Optional but recommended):
- What did you complete yesterday?
- What are you working on today?
- Any blockers?

---

## 🔥 Critical Success Factor

**THE KEY**: Depth Matcher must be **model-agnostic**

**Do**:
- ✅ Accept ANY Detection2DArray input
- ✅ Use standard ROS messages
- ✅ Make model path a parameter

**Don't**:
- ❌ Hard-code class names
- ❌ Assume specific detection format
- ❌ Couple to custom model logic

**This enables true parallel work!**

---

**Ready to distribute work? Share this plan with both teammates.**
