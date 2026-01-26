# GitHub Issues for Vision Pipeline - Task Breakdown

This document provides ready-to-use GitHub issue templates for the remaining vision pipeline tasks.

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
