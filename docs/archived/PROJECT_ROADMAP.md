# Real Robot Integration Roadmap

## Phase 1: Architecture & Documentation [Done]
- [x] Create `PROJECT_ARCHITECTURE_REPORT.md` (System Manual)
- [x] Create `REAL_ROBOT_INTEGRATION.md` (Implementation Guide)
    - [x] Define MoveIt -> Python -> ESP32 Data Flow
    - [x] Define "Digital Twin" vs "Standard" Architectures
    - [x] Define "Visual Seam Tracking" Pipeline (YOLO + Splines)
    - [x] Add Full Source Code Appendices (A-F)
- [x] Generate PDF/HTML versions of reports

## Phase 2: The Driver Layer [Done]
- [x] Create `parol6_driver` ROS 2 Package (Python)
- [x] Implement `real_robot_driver.py` (Action Server Node)
- [x] Configure `setup.py` and `package.xml` dependencies
- [x] Fix Build Permissions
- [x] Verify node startup

## Phase 3: The Firmware Layer [Hardware Failed -> Virtual]
- [x] (Attempted) Create `firmware.ino`
- [x] Implement `virtual_esp32.py` (Software Simulation)
- [x] Setup `socat` Virtual Serial Port (Ports 8/9 active)
- [ ] **[NEXT]** Verify ROS -> Virtual Serial -> Virtual ESP32 Loop
    - [x] Create `start_real_robot.sh` (The Docker Launcher)
    - [x] Launch Setup and Verify Motion

## Phase 4: Dependencies & Environment [ToDo]
- [x] Merge `xbox-camera` branch (Contains Kinect Drivers)
- [ ] Update `Dockerfile`
    - [ ] Add `libfreenect2` (Kinect)
    - [ ] Add `ultralytics` (YOLOv8)
    - [ ] Add `scipy` (B-Splines)
    - [ ] Add `socat` (For simulation)
- [ ] Rebuild Docker Image

## Phase 5: The Vision Layer [ToDo]
- [ ] Create `parol6_vision` ROS 2 Package
- [ ] Implement `vision_processor.py`
    - [ ] YOLOv8 Integration
    - [ ] Depth/Pointcloud Deprojection
- [ ] Implement `seam_path_planner.py`
    - [ ] B-Spline Path Smoothing
    - [ ] MoveIt Cartesian Path Call

## Phase 6: System Integration (The "Bringup") [ToDo]
- [ ] Create `real_robot_bringup.launch.py` (For Deployment)
    - [ ] Orchestrate Driver + Vision + MoveIt + RViz
- [ ] Final "Dry Run" Test (No Power)
- [ ] Final "Live" Test (With Power)
