# GitHub Issues for Vision Pipeline - Task Breakdown

This document provides ready-to-use GitHub issue templates for the remaining vision pipeline tasks.

---

## Issue #0: HSV Inspector Utility Tool (Development Aid)

**Title:** `HSV Inspector Node - Interactive Parameter Tuning Tool`

**Labels:** `utility`, `developer-tool`, `vision-pipeline`, `enhancement`

**Milestone:** Vision Pipeline v1.0 (Supporting Tool)

**Description:**

### 📋 Overview
Create an interactive HSV color space inspector tool that helps developers tune red line detection parameters in real-time by displaying pixel-level RGB and HSV values as the mouse moves over the camera feed.

### 🎯 Objectives
- Subscribe to live Kinect camera feed (`/kinect2/qhd/image_color_rect`)
- Display camera image in OpenCV window
- Show real-time RGB and HSV values at mouse cursor position
- Provide visual feedback (crosshair, text overlay)
- Enable rapid parameter tuning workflow for red line detector

### ✅ Acceptance Criteria
- [ ] ROS2 node implemented: `hsv_inspector_node.py`
- [ ] Subscribes to configurable image topic (default: `/kinect2/qhd/image_color_rect`)
- [ ] OpenCV window titled "HSV Inspector" displays live camera feed
- [ ] Mouse move callback triggered on every cursor movement
- [ ] Text overlay shows:
  - Pixel coordinates `(x, y)`
  - RGB values `(R, G, B)`
  - HSV values `(H, S, V)`
- [ ] Visual crosshair/circle at cursor position
- [ ] Entry point registered in `setup.py`: `hsv_inspector = parol6_vision.hsv_inspector_node:main`
- [ ] Node can be launched via: `ros2 run parol6_vision hsv_inspector`

### 📚 Resources
- Node implementation: `parol6_vision/parol6_vision/hsv_inspector_node.py`
- Related config: `parol6_vision/config/detection_params.yaml`
- Usage guide: To be documented in `RED_LINE_DETECTOR_GUIDE.md` (Tuning section)

### 🧪 Testing Requirements
- [ ] Test with live Kinect camera feed
- [ ] Test with mock camera (`mock_camera_publisher.py`)
- [ ] Verify correct HSV conversion (OpenCV BGR→HSV pipeline)
- [ ] Test with different lighting conditions
- [ ] Verify GUI responsiveness (no lag on mouse movement)

### 📊 Usage Workflow

**Developer workflow for HSV tuning:**
1. Launch Kinect driver:
   ```bash
   ros2 launch kinect2_bridge kinect2_bridge.launch.py
   ```

2. Run HSV Inspector:
   ```bash
   ros2 run parol6_vision hsv_inspector
   ```

3. Hover mouse over red markers in the camera view

4. Record HSV values for red regions

5. Update `detection_params.yaml`:
   ```yaml
   hsv_lower_1: [0, 100, 100]    # Adjust based on readings
   hsv_upper_1: [10, 255, 255]
   hsv_lower_2: [160, 50, 0]
   hsv_upper_2: [180, 255, 255]
   ```

6. Test with red line detector to validate

### 🎨 Implementation Details

**Key Features:**
- **Real-time feedback:** Instant HSV display without recording/playback
- **Lightweight:** Simple OpenCV GUI, no heavy dependencies
- **Configurable topic:** Can inspect any image topic (not just Kinect)
- **Development-only:** Not part of production pipeline

**Technical Notes:**
- Use `cv_bridge` for ROS→OpenCV conversion
- HSV conversion: `cv2.cvtColor(image, cv2.COLOR_BGR2HSV)`
- Mouse callback: `cv2.setMouseCallback("HSV Inspector", callback_function)`
- Text overlay: `cv2.putText()` with contrasting color (red on image)
- Circle marker: `cv2.circle()` to highlight cursor position

### 🔗 Dependencies
- Requires: Kinect v2 camera running (or mock camera)
- Supports: Red Line Detector parameter tuning
- Optional: RViz not required (standalone tool)

### 📝 Documentation Tasks
- [ ] Add section to `RED_LINE_DETECTOR_GUIDE.md`:
  - "HSV Parameter Tuning with Inspector Tool"
  - Screenshot showing HSV values being read
  - Example values for different lighting conditions
- [ ] Add usage example to `TESTING_GUIDE.md`
- [ ] Create quick reference in README

### 🎯 Success Metrics
- Reduces HSV tuning time from 30+ minutes to < 5 minutes
- Teammates can independently tune parameters
- No trial-and-error with hardcoded values

### 💡 Future Enhancements (Optional)
- [ ] Trackbars for live HSV range visualization
- [ ] Save/load HSV presets
- [ ] Display histogram of HSV distribution
- [ ] Region-of-interest (ROI) selection for batch sampling

---

**Priority:** **Medium** (Utility tool - very useful but not blocking)  
**Estimated Effort:** 2-3 hours (already implemented, needs documentation)  
**Status:** ✅ **IMPLEMENTED** - Needs documentation and testing

---

## Issue #1: Kinect Calibration Converter Tool

**Title:** `Kinect v2 Calibration File Converter (OpenCV → ROS)`

**Labels:** `utility`, `calibration`, `setup`, `enhancement`

**Milestone:** Vision Pipeline v1.0 (Supporting Tool)

**Description:**

### 📋 Overview
Create a conversion script that transforms Kinect v2 calibration files from OpenCV format (`calib_color.yaml`, `calib_pose.yaml`) to ROS-compatible `camera_params.yaml` format for use in the vision pipeline.

### 🎯 Objectives
- Read OpenCV calibration files from `parol6_vision/data/`
- Extract camera intrinsic parameters (fx, fy, cx, cy, distortion)
- Convert 3×3 rotation matrix to quaternion representation
- Extract depth-to-color sensor alignment (informational)
- Generate ROS-compatible YAML configuration file
- Provide clear instructions for manual camera-robot calibration

### ✅ Acceptance Criteria
- [ ] Python script: `convert_kinect_calibration.py` (root directory)
- [ ] Reads input from `parol6_vision/data/`:
  - `calib_color.yaml` - Camera intrinsics
  - `calib_pose.yaml` - Depth-RGB alignment
- [ ] Outputs to: `parol6_vision/config/camera_params_calibrated.yaml`
- [ ] Extracts intrinsic parameters:
  - Focal lengths (fx, fy)
  - Principal point (cx, cy)
  - Distortion coefficients [k1, k2, p1, p2, k3]
- [ ] Converts rotation matrix to quaternion using scipy
- [ ] Includes depth→color transform (for reference)
- [ ] Warns user about missing camera→robot calibration
- [ ] Provides default placeholder values for extrinsic calibration

### 📚 Resources
- Conversion script: `convert_kinect_calibration.py`
- Input calibration files: `parol6_vision/data/calib_*.yaml`
- Output file: `parol6_vision/config/camera_params_calibrated.yaml`
- Setup guide: `parol6_vision/docs/CALIBRATION_SETUP_GUIDE.md`

### 🧪 Testing Requirements
- [ ] Test with provided Kinect v2 calibration files
- [ ] Verify intrinsic parameters match OpenCV source
- [ ] Validate quaternion conversion (compare with online calculators)
- [ ] Ensure output YAML is valid ROS format
- [ ] Test script runs in Docker environment

### 📊 Conversion Algorithm

**Step 1: Camera Intrinsics**
```python
# From camera matrix:
# [fx  0  cx]
# [0  fy  cy]
# [0   0   1]

fx = matrix_data[0]  # Element [0,0]
fy = matrix_data[4]  # Element [1,1]
cx = matrix_data[2]  # Element [0,2]
cy = matrix_data[5]  # Element [1,2]
```

**Step 2: Rotation Matrix → Quaternion**
```python
# 3x3 rotation matrix from calib_pose.yaml
R = np.array(rot_data).reshape(3, 3)

# Convert using scipy
r = Rotation.from_matrix(R)
quat = r.as_quat()  # [x, y, z, w]
```

**Step 3: Generate ROS YAML**
```yaml
/**:
  ros__parameters:
    camera_intrinsics:
      fx: <extracted>
      fy: <extracted>
      cx: <extracted>
      cy: <extracted>
      distortion: [k1, k2, p1, p2, k3]
    camera_to_base_transform:
      translation: {x: 0.5, y: 0.0, z: 1.0}  # ⚠️ PLACEHOLDER
      rotation: {x: -0.5, y: 0.5, z: -0.5, w: 0.5}  # ⚠️ PLACEHOLDER
    depth_to_color_transform:
      translation: {x: <extracted>, y: <extracted>, z: <extracted>}
      rotation: {x: <converted>, y: <converted>, z: <converted>, w: <converted>}
```

### 🛠️ Usage Workflow

**For Teammates:**
1. Ensure Kinect calibration files exist in `parol6_vision/data/`
2. Run conversion script:
   ```bash
   docker exec parol6_dev bash -c "cd /workspace && python3 convert_kinect_calibration.py"
   ```
3. Review generated file: `parol6_vision/config/camera_params_calibrated.yaml`
4. **CRITICAL:** Measure camera-to-robot transform (see `CALIBRATION_SETUP_GUIDE.md`)
5. Edit `camera_to_base_transform` section with real values
6. Activate calibration:
   ```bash
   cp parol6_vision/config/camera_params_calibrated.yaml \
      parol6_vision/config/camera_params.yaml
   ```

### 🔗 Dependencies
- **Python packages:** `pyyaml`, `numpy`, `scipy`
- **Input files:** Kinect v2 calibration (from manufacturer or custom calibration)
- **Supports:** Depth Matcher node (Issue #2) - provides intrinsic parameters

### 📝 What's Converted vs. What's Missing

**✅ Automatically Converted (from Kinect files):**
- Camera intrinsics (fx, fy, cx, cy)
- Radial distortion (k1, k2, k3)
- Tangential distortion (p1, p2)
- Depth-RGB sensor offset (~52mm for Kinect v2)
- Internal sensor rotation (typically < 1°)

**⚠️ User Must Provide (Manual Calibration):**
- Camera position relative to robot base (x, y, z in meters)
- Camera orientation relative to robot (quaternion or Euler angles)
- See `CALIBRATION_SETUP_GUIDE.md` for 3 calibration methods:
  1. Manual measurement (±10-20mm)
  2. ArUco marker method (±5mm)
  3. Hand-eye calibration (±2-3mm)

### 📊 Output Example

```bash
============================================================
Kinect v2 Calibration Converter
============================================================

[1/4] Loading calibration files...
[2/4] Extracting camera intrinsics...
   fx: 1059.95 pixels
   fy: 1053.93 pixels
   cx: 954.88 pixels
   cy: 523.74 pixels
   Distortion: ['0.0563', '-0.0742', '0.0014', '-0.0017', '0.0241']

[3/4] Extracting depth→color transform (internal calibration)...
   Translation: {'x': -0.052, 'y': -0.0005, 'z': 0.0009}
   Rotation (quat): {'x': -0.0009, 'y': -0.0054, 'z': 0.0084, 'w': 0.9999}

[4/4] Generating camera_params.yaml...
⚠️  WARNING: Using default camera→robot translation!
   You MUST calibrate this for accurate 3D positioning!

✅ Generated: /workspace/parol6_vision/config/camera_params_calibrated.yaml
```

### 🎓 Educational Value

**Why This Tool Matters:**
- **Thesis Defense:** Shows understanding of camera calibration mathematics
- **Reproducibility:** Teammates can regenerate config from source calibration
- **Modularity:** Separates intrinsic (Kinect-specific) from extrinsic (setup-specific) calibration
- **Documentation:** Script itself is educational (rotation matrix → quaternion math)

**Key Concepts Demonstrated:**
- Pinhole camera model (intrinsic parameters)
- OpenCV calibration format parsing
- Rotation representations (matrix vs. quaternion)
- Coordinate frame transformations
- ROS parameter conventions

### 📖 Documentation Tasks
- [x] Script has inline documentation ✅
- [x] `CALIBRATION_SETUP_GUIDE.md` explains usage ✅
- [ ] Add section to `DEPTH_MATCHER_GUIDE.md` referencing this tool
- [ ] Create visual diagram: Calibration data flow
- [ ] Example calibration values for common setups

### 🎯 Success Metrics
- Teammates can convert calibration files independently
- Intrinsic parameters validated (RMS error < 0.5 pixels)
- Clear separation of "done" vs. "to-do" items
- Reduces setup time from hours to minutes

### 💡 Future Enhancements (Optional)
- [ ] Support other camera formats (RealSense, Zed)
- [ ] Automatic validation against known calibration patterns
- [ ] GUI for entering camera-robot transform visually
- [ ] Integration with hand-eye calibration tools

---

**Priority:** **High** (One-time setup, but critical for pipeline accuracy)  
**Estimated Effort:** 3-4 hours (already implemented, needs testing + docs)  
**Status:** ✅ **IMPLEMENTED** - Ready for teammate testing

---

## Issue #2: Depth Matcher - 3D Point Cloud Projection

**Title:** `Implement Depth Matcher Node (3D Projection)`

**Labels:** `enhancement`, `vision-pipeline`, `node-implementation`

**Milestone:** Vision Pipeline v1.0

**Description:**

### 📋 Overview
Implement the `depth_matcher` node that projects 2D weld line detections into 3D space using synchronized depth data from the Kinect v2 camera.

### 🎯 Objectives
- Subscribe to 2D detections (`WeldLineArray`) and depth images
- Implement pinhole camera back-projection using intrinsic parameters
- Synchronize RGB and depth streams using `message_filters`
- Transform 3D points from camera frame to robot base frame
- Apply statistical outlier filtering to remove noise
- Publish `WeldLine3DArray` messages with quality metrics

### ✅ Acceptance Criteria
- [ ] Node subscribes to `/vision/weld_lines_2d`, `/kinect2/qhd/image_depth_rect`, and `/kinect2/qhd/camera_info`
- [ ] Publishes `/vision/weld_lines_3d` with 3D points in `base_link` frame
- [ ] Implements TF transformation from camera to robot base
- [ ] Includes depth quality metric (% valid depth readings)
- [ ] Handles invalid depth values (0, NaN, out of range)
- [ ] Statistical outlier removal reduces noise
- [ ] RViz markers published for visualization

### 📚 Resources
- Implementation plan: `implementation_plan.md` (Lines 234-297)
- Developer guide: `parol6_vision/docs/DEPTH_MATCHER_GUIDE.md`
- Message definitions: `parol6_msgs/msg/WeldLine3D.msg`
- Camera calibration: `parol6_vision/config/camera_params.yaml`

### 🧪 Testing Requirements
- [ ] Unit test with mock depth data
- [ ] Integration test with mock camera publisher
- [ ] Verify 3D coordinates against known object positions
- [ ] Test synchronization with varying frame rates

### 📊 Performance Targets
- Processing rate: 10-15 Hz
- Latency: < 100ms (including synchronization)
- Depth quality: > 70% valid points for reliable lines

### 🔗 Dependencies
- Requires: Issue #1 (Red Line Detector) ✅ Complete
- Blocks: Issue #3 (Path Generator)

---

## Issue #3: Path Generator - Trajectory Smoothing

**Title:** `Implement Path Generator Node (B-Spline Smoothing)`

**Labels:** `enhancement`, `vision-pipeline`, `path-planning`

**Milestone:** Vision Pipeline v1.0

**Description:**

### 📋 Overview
Implement the `path_generator` node that converts 3D weld line points into smooth, ordered welding trajectories with appropriate end-effector orientations.

### 🎯 Objectives
- Subscribe to 3D weld lines (`WeldLine3DArray`)
- Order points along principal direction using PCA
- Fit B-spline curves for smooth trajectories (C² continuity)
- Resample to uniform waypoint spacing (configurable, default 5mm)
- Generate end-effector orientations based on path tangent + fixed approach angle
- Publish `nav_msgs/Path` messages for MoveIt

### ✅ Acceptance Criteria
- [ ] Node subscribes to `/vision/weld_lines_3d`
- [ ] Publishes `/vision/welding_path` as `nav_msgs/Path`
- [ ] B-spline smoothing with configurable degree (default: 3)
- [ ] Orientation generation for planar surfaces (tangent + approach angle)
- [ ] Quality checks: minimum waypoints, maximum curvature
- [ ] Path statistics service (`~/get_path_statistics`)
- [ ] RViz visualization markers for path preview

### 📚 Resources
- Implementation plan: `implementation_plan.md` (Lines 299-356)
- Configuration: `parol6_vision/config/path_params.yaml`
- Utility module: `parol6_vision/utils/path_utils.py`

### 🧪 Testing Requirements
- [ ] Unit test with synthetic 3D points
- [ ] Verify smooth transitions (no sharp corners)
- [ ] Test waypoint spacing accuracy
- [ ] Validate orientation consistency
- [ ] Integration test with full pipeline

### 📊 Performance Targets
- Processing rate: 1-2 Hz (triggered on new detection)
- Waypoint density: Configurable (default 5mm spacing)
- Path smoothness: C² continuous (smooth velocity & acceleration)

### 🔗 Dependencies
- Requires: Issue #2 (Depth Matcher)
- Blocks: Issue #4 (MoveIt Controller)

### 📝 Notes
**Scope:** Initial implementation assumes planar welding surfaces. Surface normal estimation from depth data is future work (document this in thesis as valid scoping).

---

## Issue #4: MoveIt Controller - Motion Execution

**Title:** `Implement MoveIt Controller Node (Trajectory Execution)`

**Labels:** `enhancement`, `motion-planning`, `moveit`

**Milestone:** Vision Pipeline v1.0

**Description:**

### 📋 Overview
Implement the `moveit_controller` node that executes welding paths using MoveIt2 Cartesian path planning with robust fallback strategies.

### 🎯 Objectives
- Subscribe to welding paths (`nav_msgs/Path`)
- Interface with MoveIt2 `move_group` for Cartesian planning
- Implement multi-resolution fallback strategy (fine → medium → coarse)
- Validate trajectory feasibility and collision safety
- Execute trajectories via `FollowJointTrajectory` action
- Provide execution control services (execute, abort, pause)

### ✅ Acceptance Criteria
- [ ] Node subscribes to `/vision/welding_path`
- [ ] Connects to MoveIt2 `move_group` action server
- [ ] Implements 3-tier fallback strategy (2mm → 5mm → 10mm step sizes)
- [ ] Success rate thresholds: 95%, 95%, 90% for each tier
- [ ] Service interface for manual execution trigger
- [ ] Pre-weld approach and post-weld retract phases
- [ ] Collision checking enabled
- [ ] Execution status feedback (state machine)

### 📚 Resources
- Implementation plan: `implementation_plan.md` (Lines 358-456)
- Configuration: `parol6_vision/config/path_params.yaml`
- MoveIt config: `parol6_moveit_config/`

### 🧪 Testing Requirements
- [ ] Unit test with mock MoveIt interface
- [ ] Simulation test in Gazebo or RViz (visual validation)
- [ ] Test fallback strategy with challenging paths
- [ ] Verify collision avoidance
- [ ] Integration test with full pipeline (mock camera → execution)

### 📊 Performance Targets
- Planning time: 1-5 seconds per path (acceptable for welding)
- Success rate: > 90% for typical weld line geometries
- Velocity scaling: 0.3 (safety factor)

### 🔗 Dependencies
- Requires: Issue #3 (Path Generator)
- Requires: `parol6_moveit_config` package configured

### ⚠️ Safety Considerations
- Implement emergency stop service
- Velocity/acceleration limits enforced
- Collision checking always enabled
- Manual approval required for first-time trajectories

---

## Issue #5: Integration Testing & Validation

**Title:** `End-to-End Integration Testing & Performance Validation`

**Labels:** `testing`, `integration`, `validation`

**Milestone:** Vision Pipeline v1.0

**Description:**

### 📋 Overview
Complete end-to-end integration testing of the vision pipeline, from live camera feed to motion execution, including performance benchmarking and calibration validation.

### 🎯 Objectives
- Test full pipeline with live Kinect v2 camera
- Validate camera-robot calibration accuracy
- Perform MoveIt execution tests in simulation
- Benchmark processing rates and latencies
- Create comprehensive test reports and demos

### ✅ Acceptance Criteria

#### Camera Integration
- [ ] Kinect v2 driver launches successfully
- [ ] RGB and depth streams publishing at target rates (~30 Hz, ~15 Hz)
- [ ] Camera intrinsic parameters loaded correctly
- [ ] Camera-robot TF transform validated with known objects

#### Pipeline Integration
- [ ] All 4 nodes launch via `vision_pipeline.launch.py`
- [ ] Topics connected correctly (verified with `ros2 topic info`)
- [ ] Message synchronization working (no timeout errors)
- [ ] Data flows: Camera → Detector → Depth Matcher → Path Gen → MoveIt

#### Calibration Validation
- [ ] Intrinsic calibration: RMS error < 0.5 pixels
- [ ] Extrinsic calibration: Position accuracy ±10mm (manual) or ±5mm (ArUco)
- [ ] Known object test: Detected position matches ground truth

#### Performance Validation
- [ ] Red line detector: 5-10 Hz achieved
- [ ] Depth matcher: 10-15 Hz achieved
- [ ] Path generator: 1-2 Hz (on trigger)
- [ ] End-to-end latency: < 500ms (detection → path)

#### MoveIt Execution (Simulation)
- [ ] Cartesian paths execute successfully in RViz
- [ ] Fallback strategy tested (all 3 tiers)
- [ ] Collision avoidance verified
- [ ] Pre/post-weld motions execute correctly

### 📚 Testing Procedures
- [ ] Follow `TESTING_GUIDE.md`:
  - Level 1: Unit tests (all nodes)
  - Level 2: Mock camera integration
  - Level 3: Live camera standalone
  - Level 4: Full pipeline with execution
- [ ] Document results in `walkthrough.md`
- [ ] Record demo video showing:
  - Detection working on real red marker
  - 3D points visualized in RViz
  - Smooth path generated
  - Robot executing trajectory (sim)

### 📊 Deliverables
- [ ] Test report with benchmark results
- [ ] Demo video (3-5 minutes)
- [ ] Updated `walkthrough.md` with validation results
- [ ] Parameter tuning recommendations document

### 🔗 Dependencies
- Requires: Issues #1, #2, #3, #4 complete
- Requires: Camera calibration complete (see `CALIBRATION_SETUP_GUIDE.md`)

### 🎬 Demo Requirements
**For Thesis Defense:**
1. Show live red marker detection
2. Visualize 3D reconstruction in RViz
3. Display generated smooth path
4. Execute welding motion in simulation
5. Present performance metrics table

---

## Issue #6: Documentation & Knowledge Transfer

**Title:** `Complete Documentation and Team Knowledge Transfer`

**Labels:** `documentation`, `knowledge-transfer`

**Milestone:** Vision Pipeline v1.0

**Description:**

### 📋 Overview
Finalize all documentation, create video tutorials, and ensure teammates can independently work with the vision pipeline.

### 🎯 Objectives
- Complete all developer guides
- Create video walkthroughs for each major component
- Document parameter tuning procedures
- Prepare thesis documentation excerpts

### ✅ Acceptance Criteria

#### Developer Guides (Complete)
- [x] Red Line Detector Guide ✅
- [x] Depth Matcher Guide ✅
- [x] Testing Guide ✅
- [x] Calibration Setup Guide ✅
- [ ] Path Generator Guide
- [ ] MoveIt Controller Guide
- [ ] Troubleshooting Compendium

#### Video Tutorials
- [ ] Getting Started (setup, build, run)
- [ ] HSV Parameter Tuning Demo
- [ ] Camera Calibration Walkthrough
- [ ] Full Pipeline Demo
- [ ] Common Issues & Solutions

#### Parameter Reference
- [ ] Complete parameter tables for all nodes
- [ ] Tuning recipes for different scenarios
- [ ] Default vs. recommended vs. production values

#### Thesis Preparation
- [ ] System architecture diagram (publish-ready)
- [ ] Performance comparison tables
- [ ] Scientific justification for design decisions
- [ ] Limitations and future work section

### 📚 Documentation Structure
```
parol6_vision/docs/
├── RED_LINE_DETECTOR_GUIDE.md ✅
├── DEPTH_MATCHER_GUIDE.md ✅
├── PATH_GENERATOR_GUIDE.md (TODO)
├── MOVEIT_CONTROLLER_GUIDE.md (TODO)
├── TESTING_GUIDE.md ✅
├── CALIBRATION_SETUP_GUIDE.md ✅
├── TROUBLESHOOTING.md (TODO)
└── PARAMETER_REFERENCE.md (TODO)
```

### 🔗 Dependencies
- Requires: All implementation issues complete
- Requires: Integration testing complete

---

## Task Priority & Timeline

**Recommended Order:**

1. **Week 1:** Issue #2 (Depth Matcher) - Critical for 3D reconstruction
2. **Week 2:** Issue #3 (Path Generator) - Enables trajectory generation
3. **Week 3:** Issue #4 (MoveIt Controller) - Motion execution
4. **Week 4:** Issue #5 (Integration Testing) - End-to-end validation
5. **Ongoing:** Issue #6 (Documentation) - Parallel with implementation

**Critical Path:**
```
Depth Matcher → Path Generator → MoveIt Controller → Integration Testing
     (2)             (3)              (4)                  (5)
```

**Parallel Work:**
- Documentation can progress alongside implementation
- Unit tests should be written with each node
- RViz visualization helps with debugging during development

---

## How to Use These Templates

1. **Copy each issue section** into GitHub's issue creation form
2. **Set appropriate labels** (already suggested in each template)
3. **Assign to teammate(s)** based on expertise:
   - Depth Matcher: Computer vision / 3D geometry
   - Path Generator: Path planning / mathematics
   - MoveIt Controller: ROS2 / motion planning
   - Integration: Testing / system integration
4. **Link issues** using GitHub's dependency features
5. **Track progress** in your GitHub project board

---

**Ready to create in GitHub!** Copy these templates to your issues and adapt milestone dates as needed.
