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
