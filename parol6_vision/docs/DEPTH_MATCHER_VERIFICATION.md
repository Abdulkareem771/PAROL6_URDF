# Depth Matcher Verification Guide

This guide helps you verify that the `depth_matcher` node is compatible with the `parol6_project` and is functioning correctly.

## Prerequisites

Ensure you are inside the Docker container:
```bash
docker exec -it parol6_dev bash
```

## Step 1: Build the Workspace

We need to build the custom messages and the vision package.

```bash
cd /workspace
colcon build --packages-select parol6_msgs parol6_vision
source install/setup.bash
```

## Step 3: Run the Verification Script (Automated)

We have created an automated script that builds the workspace, starts the necessary background nodes (TF publisher, depth matcher), and runs the verification test.

Simply run:

```bash
./run_verification.sh
```

Alternatively, if you want to run steps manually:

1. **Build**: `colcon build --packages-select parol6_msgs parol6_vision && source install/setup.bash`
2. **Start TF**: `ros2 run tf2_ros static_transform_publisher ...` (see script for args)
3. **Start Node**: `ros2 run parol6_vision depth_matcher`
4. **Run Test**: `python3 parol6_vision/scripts/verify_depth_matcher.py`

## Expected Output

If successful, you should see:

```text
[INFO] [depth_matcher_verifier]: Published synchronized mock data batch.
[INFO] [depth_matcher_verifier]: ✅ SUCCESS! Received 3D Weld Lines.
...
TEST PASSED: Depth Matcher is compatible and working!
```

## Troubleshooting

If the test fails:

1. **TF Transform Missing**: Ensure the `static_transform_publisher` from Step 2 is running. The node will not publish if it cannot transform points to `base_link`.
2. **Topic Mismatch**: Check if `ros2 topic list` shows:
   - `/vision/weld_lines_2d`
   - `/vision/weld_lines_3d`
   - `/kinect2/qhd/image_depth_rect`
   - `/kinect2/qhd/camera_info`
3. **Synchronization**: The test script automatically timestamps messages. If latency is high, `message_filters` might drop messages.

## Cleanup

To stop the background processes:

```bash
pkill -f static_transform_publisher
pkill -f depth_matcher
```

## Method 2: Visual Verification with ROS Bag

If you are testing with recorded data (ROS Bag) using `test_depth_matcher_bag.launch.py`:

### 1. Visual Check in RViz
When the launch file runs, RViz should open. Look for:
- **Robot Model**: The PAROL6 robot should be visible.
- **Weld Points (Blue Dots)**: 3D points representing the weld seam. They should appear in front of the robot.
- **Weld Connectivity (Cyan Lines)**: A line connecting the blue dots.
- **Alignment**: The points should roughly match where the red line was in the real world relative to the robot base (e.g., ~0.5m in front X, ~0.6m up Z, depending on calibration).

### 2. Data Check (Terminal)
Open a new terminal inside the container (`docker exec -it parol6_dev bash`) and run:

```bash
ros2 topic echo /vision/weld_lines_3d
```

**What to look for:**
- **`lines` array is NOT empty**. If it's empty `[]`, no 3D lines are being generated.
- **`depth_quality`**: Should be > 0.6 (60%).
- **`points`**: The X, Y, Z coordinates should look reasonable (e.g., Z is approx 0.0 if on table, or matching the stand height).

### 3. Debug Image
Check the 2D detector's debug view to ensure it sees the red line in the first place:
- topic: `/red_line_detector/debug_image`
- You can view this in RViz by adding an **Image** display.

