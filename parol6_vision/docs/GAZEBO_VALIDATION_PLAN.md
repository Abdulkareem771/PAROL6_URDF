# Gazebo Validation Plan - Vision-Guided Path Execution

## 🎯 Objective
Validate the complete vision → planning → execution pipeline using deterministic sensor data (ROS bag replay) and quantify performance metrics in Gazebo simulation.

---

## 📊 Metrics to Measure

### 1. Path Tracking Accuracy
- **End-effector position error** (mm)
  - Mean, max, RMS deviation from planned path
- **Orientation error** (degrees)
  - Mean, max angular deviation
- **Success rate** (%)
  - Percentage of paths completed without failures

### 2. Path Quality
- **Smoothness** (m/s², rad/s²)
  - Joint velocity/acceleration continuity
  - No sudden jerks or discontinuities
- **Path length** (m)
  - Compare generated path vs ideal straight-line distance

### 3. Timing & Performance
- **Vision processing latency** (ms)
  - Time from image to detected line
- **Path generation time** (ms)
  - Time from 3D points to smooth path
- **Planning success rate** (%)
  - MoveIt planning solve rate
- **Execution time** (s)
  - Total time to complete welding path

### 4. Repeatability
- **Inter-trial variance**
  - Running same bag 10x → should produce near-identical results
  - Quantify standard deviation

---

## 🧪 Test Scenarios

### Scenario 1: Single Straight Line
**Setup:**
- Simple red line (horizontal, 200mm)
- Camera perpendicular
- Well-lit environment

**Expected:**
- Detection confidence > 95%
- Path tracking error < 2mm
- Planning success 100%

**Validation:**
- Measures baseline performance
- Confirms end-to-end pipeline works

---

### Scenario 2: Curved Weld Seam
**Setup:**
- Gentle curve (radius ~300mm)
- 45° angle to camera
- Moderate lighting

**Expected:**
- Detection confidence > 90%
- Path tracking error < 3mm
- Smooth motion (no jerks)

**Validation:**
- Tests B-spline smoothing
- Validates orientation generation

---

### Scenario 3: Multi-Segment Path
**Setup:**
- 2-3 disconnected line segments
- Different orientations
- Variable lighting

**Expected:**
- All segments detected
- Correct segmentation
- Independent path generation for each

**Validation:**
- Tests segmentation logic
- Multi-target planning

---

### Scenario 4: Edge Cases
**Setup:**
- Partial occlusion
- Line near image boundary
- Varying brightness

**Expected:**
- Graceful degradation
- Clear failure modes
- Diagnostic messages

**Validation:**
- Robustness testing
- Error handling

---

## 🔄 Validation Workflow

### Phase 1: Bag Replay Setup (10 min)
```bash
# Terminal 1: Replay sensor data
unset ROS_DOMAIN_ID
ros2 bag play test_data/kinect_snapshot_* --loop

# Terminal 2: Launch Gazebo
ros2 launch parol6_moveit_config demo_gazebo.launch.py
```

**Verify:**
- Gazebo showing robot
- Camera topics publishing from bag
- RViz visualization active

---

### Phase 2: Vision Pipeline Activation (5 min)
```bash
# Terminal 3: Launch vision nodes
ros2 launch parol6_vision vision_pipeline.launch.py auto_execute:=false
```

**Verify:**
- Red line detector active
- Debug image showing detections
- 3D points published
- Path generated

---

### Phase 3: Manual Path Inspection (5 min)
**In RViz:**
- View generated `nav_msgs/Path`
- Check waypoint density
- Verify orientations (arrows)
- Inspect for discontinuities

**Data Collection:**
```bash
ros2 topic echo /vision/welding_path > path_log.yaml
```

---

### Phase 4: MoveIt Planning Test (10 min)
```bash
# Enable auto-execution
ros2 param set /moveit_controller auto_execute true
```

**OR manually plan:**
- Use MoveIt RViz plugin
- Plan to first waypoint
- Execute and observe

**Monitor:**
```bash
ros2 topic echo /follow_joint_trajectory/feedback
```

---

### Phase 5: Metric Collection (Automated)

#### A. Position Tracking Error
```python
# Record ground truth + actual
ros2 bag record \
    /vision/welding_path \
    /joint_states \
    /tf \
    -o validation_run_1
```

**Post-process:**
- Extract end-effector poses
- Compare to planned path
- Calculate errors

#### B. Timing Data
```bash
ros2 topic echo /red_line_detector/elapsed_time
ros2 topic echo /path_generator/elapsed_time
```

#### C. Success/Failure Logging
Monitor:
```
/move_group/result
/moveit_controller/status
```

---

### Phase 6: Repeat & Statistics (30 min)
**Run 10 iterations:**
```bash
for i in {1..10}; do
    ros2 service call /moveit_controller/reset std_srvs/srv/Trigger
    sleep 2
   ros2 service call /moveit_controller/execute std_srvs/srv/Trigger
    sleep 30
    echo "Run $i complete"
done
```

**Collect:**
- Per-run metrics
- Mean, std dev, min, max
- Outlier analysis

---

## 📈 Expected Results (Baseline)

| Metric | Target Value | Acceptable Range |
|--------|--------------|------------------|
| Position error (mean) | < 2mm | < 5mm |
| Position error (max) | < 5mm | < 10mm |
| Orientation error | < 2° | < 5° |
| Planning success rate | 100% | > 95% |
| Vision latency | < 100ms | < 200ms |
| Path gen time | < 50ms | < 100ms |
| Inter-trial std dev | < 0.5mm | < 1mm |

---

## 🛠️ Tools & Scripts

### Metric Extraction Script
**File:** `parol6_vision/scripts/analyze_validation_bag.py`

**Purpose:**
- Parse validation bags
- Extract TF trees
- Compute errors
- Generate plots

**Usage:**
```bash
ros2 run parol6_vision analyze_validation_bag validation_run_1
```

**Outputs:**
- `metrics.json` - Structured data
- `tracking_error.png` - Error over time plot
- `path_visualization.png` - 3D path comparison

---

### Auto-Validation Launch File
**File:** `parol6_vision/launch/auto_validation.launch.py`

**Purpose:**
- Launch all components
- Start bag replay
- Enable metric collection
- Run predefined test sequence

**Usage:**
```bash
ros2 launch parol6_vision auto_validation.launch.py \
    bag_path:=test_data/kinect_snapshot_* \
    scenario:=single_straight_line \
    num_runs:=10
```

---

## 🎓 Thesis Documentation

### Dataset Section
> "Validation experiments were conducted using frozen ROS bag dataset `kinect_snapshot_20260124_024153` (commit: `b17d0aa...`) to ensure deterministic, repeatable testing across all trials."

### Methodology Section
> "The complete vision-guided welding pipeline was validated in Gazebo simulation. A frozen 3-second RGB-D sensor snapshot was replayed in loop mode while the robot executed generated welding paths. End-effector tracking error was measured by comparing planned waypoints (from vision system) against actual robot poses (from Gazebo ground truth via TF tree). Ten independent trials were performed per scenario to quantify repeatability."

### Results Section
Present tables like:

| Scenario | Mean Error (mm) | Max Error (mm) | Success Rate |
|----------|-----------------|----------------|--------------|
| Straight line | 1.2 ± 0.3 | 2.1 | 100% |
| Curved seam | 2.4 ± 0.5 | 4.3 | 100% |
| Multi-segment | 1.8 ± 0.4 | 3.2 | 97% |

---

## 🚀 Next Steps After Validation

1. **If metrics acceptable:**
   - Document in thesis
   - Proceed to real hardware testing
   - Calibrate safety parameters

2. **If metrics poor:**
   - Debug specific failure modes
   - Tune vision parameters
   - Adjust path smoothing
   - Re-test

3. **Future enhancements:**
   - Adaptive velocity based on path curvature
   - Torch angle optimization
   - Multi-pass welding support

---

## ⚠️ Important Notes

### Gazebo Limitations
- Physics simulation ≠ real hardware
- No actual welding physics
- Use as **proof of concept**, not final validation

### Bag Replay Considerations
- Ensure bag starts **before** launching pipeline
- TF static must be available
- ROS_DOMAIN_ID should be unset

### Data Management
- Save all validation bags with:
  - Date/time stamp
  - Scenario name
  - Git commit hash
- Archive for thesis appendix

---

## ✅ Validation Checklist

Before claiming validation complete:
- [ ] All 4 test scenarios executed
- [ ] 10 trials per scenario completed
- [ ] Metrics collected and analyzed
- [ ] Plots generated
- [ ] Results documented
- [ ] Outliers investigated
- [ ] Known limitations documented
- [ ] Data archived with git hash

---

## 🎯 Success Criteria

**Validation passes if:**
- ✅ All scenarios have > 95% success rate
- ✅ Mean tracking error < 3mm
- ✅ Inter-trial std dev < 1mm
- ✅ No unexpected failures
- ✅ Results reproducible across different machines (same bag)

**Result:** Thesis-grade experimental validation demonstrating deterministic, quantifiable performance.
