# Eye-to-Hand Calibrator Node

## Overview

`eye_to_hand_calibrator` is a ROS 2 (Humble) node that computes the **static spatial relationship between a fixed (eye-to-hand) camera and an ArUco marker attached to the robot's end-effector or workspace**. It achieves this by listening to the TF tree over a configurable number of samples, then averaging the collected transforms to produce a stable, noise-reduced result.

The output is a ready-to-paste `ros2 run tf2_ros static_transform_publisher` command that permanently registers the calibrated transform in the TF tree.

---

## File Location

```
parol6_vision/
└── parol6_vision/
    └── eye_to_hand_calibrator.py
```

**Entry point (registered in `setup.py`):**
```
eye_to_hand_calibrator = parol6_vision.eye_to_hand_calibrator:main
```

---

## Node Details

| Property        | Value                        |
|-----------------|------------------------------|
| Node name       | `eye_to_hand_calibrator`     |
| Language        | Python 3 / rclpy             |
| ROS version     | ROS 2 Humble                 |
| Package         | `parol6_vision`              |

---

## How It Works

### Step 1 — TF Listening

The node creates a `tf2_ros.Buffer` + `TransformListener` pair to subscribe to the live TF tree. A 10 Hz timer (`0.1 s`) repeatedly calls `lookup_transform()` to query the transform from the **camera optical frame** to the **ArUco marker frame**.

```
source_frame  →  kinect_rgb_optical_frame   (camera)
target_frame  →  cube_marker               (ArUco marker)
```

### Step 2 — Sample Collection

Each successful lookup appends:
- **Translation** `[x, y, z]` (meters) to a list.
- **Quaternion** `[x, y, z, w]` to a list.

Progress is logged every 10 samples. If the transform is unavailable (e.g., the marker is occluded or TF not yet published), a `WARN` is emitted and the timer retries on the next tick.

### Step 3 — Statistical Averaging

Once `samples_to_collect` (default: **100**) samples are gathered, the timer is cancelled and `compute_final_transform()` runs:

| Quantity    | Method                                                                 |
|-------------|------------------------------------------------------------------------|
| Translation | Simple arithmetic mean (`np.mean`) across all samples                  |
| Rotation    | Geometric mean of rotations via `scipy.spatial.transform.Rotation.mean()` — handles the non-linear quaternion space correctly |

> **Why not average quaternions directly?**  
> Quaternions live on a unit sphere (SO(3) manifold). A simple arithmetic mean does not stay on the manifold and can produce an invalid rotation. `Rotation.mean()` computes the Fréchet mean, which is the correct geodesic average.

### Step 4 — Output

Results are printed to both the ROS logger and stdout:

```
--- CALIBRATION RESULTS (Camera to Marker) ---
Translation (meters): X=0.1234, Y=-0.0567, Z=0.8901
Quaternion: x=0.0012, y=-0.0034, z=0.7071, w=0.7071
Euler (degrees): Roll=0.10, Pitch=-0.20, Yaw=90.05

--- SUGGESTED STATIC TRANSFORM COMMAND ---
ros2 run tf2_ros static_transform_publisher 0.1234 -0.0567 0.8901 \
    0.0012 -0.0034 0.7071 0.7071 kinect_rgb_optical_frame cube_marker
```

Paste the suggested command into your launch file or run it in a terminal to permanently broadcast the calibrated transform.

---

## Configuration

All configuration is at the top of `__init__()` under the `# --- CONFIGURATION ---` block:

| Parameter            | Default                        | Description                                          |
|----------------------|--------------------------------|------------------------------------------------------|
| `source_frame`       | `kinect_rgb_optical_frame`     | TF frame of the camera's optical axis                |
| `target_frame`       | `cube_marker`                  | TF frame of the ArUco marker (published by your ArUco node) |
| `samples_to_collect` | `100`                          | Number of transform samples to average               |

To change them, edit the values directly in the file — no ROS parameters are required.

---

## Prerequisites

Before running this node, the following must be active:

1. **Camera driver** — publishing images and its TF frame (e.g., `kinect_rgb_optical_frame`).
2. **ArUco detection node** — listening to the camera feed and broadcasting the `cube_marker` TF frame. Any standard ArUco ROS 2 package (e.g., `ros2_aruco`, `aruco_opencv`) works.
3. **Static TF for camera mounting** (if the camera is rigidly mounted, its base frame must already be in the TF tree).

---

## Running the Node

### Inside the Docker container

```bash
# Open a shell in the container
docker exec -it parol6_dev bash

# Source the workspaces
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

# Run the node
ros2 run parol6_vision eye_to_hand_calibrator
```

### Expected terminal output (normal operation)

```
[INFO] Starting calibration. Looking for transform from kinect_rgb_optical_frame to cube_marker...
[INFO] Collected 10/100 samples...
[INFO] Collected 20/100 samples...
...
[INFO] Collected 100/100 samples...
[INFO] Collection complete. Calculating average...
[INFO] --- CALIBRATION RESULTS (Camera to Marker) ---
```

### Expected terminal output (marker not visible)

```
[WARN] Could not find transform: "kinect_rgb_optical_frame" passed to lookupTransform argument target_frame does not exist.
```
This is normal — the node retries automatically. Make sure the ArUco marker is in the camera's field of view and the detection node is running.

---

## Using the Calibration Result

After the node prints the suggested command, add the static transform to your launch file:

```python
# In your launch file (e.g., parol6_vision.launch.py)
from launch_ros.actions import Node

Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    arguments=[
        '0.1234', '-0.0567', '0.8901',        # x y z (translation)
        '0.0012', '-0.0034', '0.7071', '0.7071', # qx qy qz qw (rotation)
        'kinect_rgb_optical_frame', 'cube_marker'
    ]
)
```

Or run it directly in a terminal alongside your other nodes:

```bash
ros2 run tf2_ros static_transform_publisher \
    0.1234 -0.0567 0.8901 0.0012 -0.0034 0.7071 0.7071 \
    kinect_rgb_optical_frame cube_marker
```

---

## Dependencies

| Library                            | Purpose                                      |
|------------------------------------|----------------------------------------------|
| `rclpy`                            | ROS 2 Python client library                  |
| `tf2_ros` (`Buffer`, `TransformListener`) | TF tree subscription and lookup        |
| `numpy`                            | Translation averaging                        |
| `scipy.spatial.transform.Rotation`| Geometrically correct rotation averaging     |

These are all available in the `parol6_dev` Docker image.

---

## Role in the PAROL6 Vision Pipeline

```
Camera (Kinect) ──► ArUco Detection Node ──► TF: cube_marker
                                                      │
                                          eye_to_hand_calibrator
                                                      │
                                         static_transform_publisher
                                                      │
                                    Depth Matcher / Path Optimizer ──► MoveIt2
```

The calibrated transform is a prerequisite for the **depth matcher** node, which projects 2D weld path points from the camera frame into 3D robot base coordinates. Without an accurate camera-to-robot transform, all downstream path coordinates will be offset.

---

## Improving Accuracy

- **More samples**: Increase `samples_to_collect` to 200–500 for a more robust average, especially in noisy or low-frame-rate scenarios.
- **Stable marker**: Ensure the ArUco marker is rigidly fixed and lit evenly during calibration.
- **Multiple poses**: For the highest accuracy, consider running calibration at several different robot poses and averaging across runs (hand-eye calibration via OpenCV's `calibrateHandEye`).
- **Verify with TF tools**: After applying the static transform, validate with:
  ```bash
  ros2 run tf2_tools view_frames
  ros2 run tf2_ros tf2_echo kinect_rgb_optical_frame cube_marker
  ```
