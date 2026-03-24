# PAROL6 Vision-Guided Trajectory System: User Guide

Welcome to the PAROL6 Vision-Guided Pipeline! This guide provides everything you need to know to initialize the Kinect v2 depth stream, visualize the point clouds, detect weld paths, and dynamically command the robotic arm natively in the Ubuntu/Docker environment.

## 1. Quick Start

### Entering the Container
All operations **must** be executed sequentially from inside the dedicated Docker container. The repository provides a unified script for this:
```bash
./start_container.sh
docker exec -it parol6_dev bash
```
*(Your workspace is automatically sourced upon opening the container's shell!)*

### Launching the Graphical Interface
We've integrated a sleek PyQT6 Controller GUI specifically designed for vision-execution workflows. To launch the command center:
```bash
python3 /workspace/parol6_vision/scripts/vision_gui.py
```
From the GUI, you can instantly spin up the entire launch orchestrator (`vision_moveit.launch.py`) with a single click. Make sure you select the **Live Camera Stream** mode rather than Bag Replay.

---

## 2. Launch Profile Architecture

If you prefer to command the interface via standard terminal execution, the system relies on one mega-orchestrator node:
```bash
ros2 launch parol6_vision vision_moveit.launch.py use_bag:=false
```

When triggered, this file activates the following architecture stack:
1. **Depth Extractor / CV Backend**: Maps the `SD` streams from `kinect2_bridge` natively into `/vision/` topics.
2. **MoveIt2 Server**: Configures `ompl` pathfinding constraints and spawns the virtual simulation nodes (`rviz2`).
3. **Hardware Integrator**: Instantiates the `.xacro` URDF configuration for the arm and loads the position-based controllers.
4. **Trajectory Executor**: Initializes our custom `vision_trajectory_executor` action-client designed to convert the 3D vision waypoints directly into interpolated Joint Commands.

---

## 3. The `SD` Depth Resolution Strategy

Important: Following recent calibration updates, the vision hardware relies **strictly** on the `SD` (Standard Definition: 512x424) infrared streams from the Kinect v2, replacing the higher latency `QHD` arrays.

All internal configurations (Camera Parameters, Extrinsic TFs, RViz views, YOLO Mask bindings) have been optimized specifically around the `sd` topic cluster:
* `/kinect2/sd/image_color_rect`
* `/kinect2/sd/image_depth_rect`
* `/kinect2/sd/camera_info`

If you attempt to remap or modify standard nodes into `qhd` resolutions, the intersection boundaries will instantly lose depth coordination and fail to triangulate.

---

## 4. Path Execution Process

Once the RViz window materializes, follow the standard workflow to initiate a precision robotic maneuver:
1. **Verify Target Visualization**: You should see a highly synchronized 3D XYZRGB PointCloud overlaying your physical calibration board.
2. **Identify Computed Path**: The vision nodes will constantly listen for weld lines on the workspace surface and render them mathematically into cyan 3D markers overlaid on the image.
3. **Execute Mode**: Utilize the "Plan & Execute" button in the RViz MotionPlanning widget, or trigger it via the GUI.
4. **Controller Action**: The trajectory interpolator reads the nearest cyan node coordinates and plans a dynamic collision-free Cartesian pathway. The hardware plugin intercepts this via the `parol6_arm_controller` (streaming velocity / position goal limits) and articulates the physical axes.

## 5. Troubleshooting / Common Errors

* **Error: `CMake Error: Could not find CMAKE_ROOT !!!`**
   * **Cause**: You mistakenly triggered a `colcon build` on the host machine.
   * **Fix**: Clean your `build` directory inside the Docker container (`rm -rf build install log`) and run the build again fully containerized.
* **Error: `missing state interfaces: ' joint_L1/effort '`**
   * **Cause**: Residual force feedback tags injected into the URDF parameters. 
   * **Fix**: Pull the original position-oriented `parol6.ros2_control.xacro` or run `git checkout parol6_hardware/` to purge the stray effort vectors.
* **Nodes crash silently or display zero data**
   * **Cause**: Ensure you're not passing `use_bag:=true` when meaning to capture real camera data. Verify the Kinect driver is generating traffic via `ros2 topic hz /kinect2/sd/image_depth_rect`.
