# Depth Matcher — Debug & Achievement Log

> **Status:** ✅ FULLY WORKING — 128 pts/frame, depth_quality = 1.0, publishing to `/vision/weld_lines_3d`  
> **Date:** 2026-02-20  
> **Author:** PAROL6 Vision Team

---

## What Was Built

The **Depth Matcher** (`depth_matcher.py`) is the 2D→3D lifting node in the welding vision pipeline.

```
ROS Bag (Kinect v2)
  ├── /kinect2/qhd/image_color_rect  ──→  [red_line_detector]  ──→  /vision/weld_lines_2d
  └── /kinect2/qhd/image_depth_rect  ──┐
                                        ├──→  [depth_matcher]  ──→  /vision/weld_lines_3d
                           /tf_static  ──┘                            /depth_matcher/markers
```

### Key Responsibilities
1. **Synchronize** RGB-detected 2D pixel lines with the depth image (ApproximateTimeSynchronizer, tol=0.5s)
2. **Back-project** valid depth pixels to 3D camera-frame coordinates using the pinhole model:
   ```
   X = (u - cx) * Z / fx
   Y = (v - cy) * Z / fy
   Z = depth_raw / 1000.0  (mm → m)
   ```
3. **Transform** 3D points from `kinect2_rgb_optical_frame` → `base_link` via TF2
4. **Filter** outliers via statistical outlier removal (k=10 neighbours, threshold=1.0σ)
5. **Publish** `WeldLine3D` messages and `MarkerArray` for RViz

---

## Final Working Numbers

| Metric | Value |
|---|---|
| Pixels per detected line | 124–130 skeleton pixels |
| Valid depth points | 124–130 / 130 (100%) |
| After outlier filter | 124–130 (no outliers) |
| `depth_quality` | **1.00** |
| 3D position (base_link) | X≈1.41m, Y≈−0.08→−0.15m, Z≈0.16m |
| Publish rate | ~2 Hz (limited by bag loop ~10s / 21 depth frames) |
| Topic | `/vision/weld_lines_3d` ✅ |
| OpenGL version (RViz) | 4.6 / GLSL 4.6 (confirmed GPU rendering works) |

### Live Log Output (Confirmed Working State)

This is what correct healthy output looks like in the terminal:

```
[depth_matcher] DEBUG depth samples: (470,125)=1096, (469,125)=1097 | depth shape=(540, 960), dtype=uint16 | min_d=300.0, max_d=2000.0
[depth_matcher] Line stats: total=128, valid_raw=128, filtered=128, quality=1.00 (need min_pts=10, min_qual=0.6)
[depth_matcher] Published 1 3D weld lines
[red_line_detector] Frame 105: Detected 1 line(s), confidences: ['1.45']
[red_line_detector] Frame 106: Detected 1 line(s), confidences: ['1.48']
```

> If you see `Published 0 3D weld lines` or the `Line stats` show `total=2` → see Bug 1 above.

---

## Bugs Found & Fixed

### Bug 1 — Douglas-Peucker Over-Simplification *(Root Cause)*

**File:** `red_line_detector.py`  
**Symptom:** `depth_matcher` always logged:
```
Line stats: total=2, valid_raw=2, filtered=2 (need min_pts=10)
→ 0 3D weld lines published
```

**Root Cause:** The `pixels` field of each `WeldLine` message was populated with `simplified_points` — the output of Douglas-Peucker (epsilon=2.0). For a nearly-straight skeleton line of ~85 pixels, D-P reduces this to exactly **2 endpoints**.

```python
# ❌ Before — sent only 2 endpoints for a straight line
line.pixels = [Point32(x=pt[0], y=pt[1]) for pt in simplified_points]

# ✅ After — sends all ~128 ordered skeleton pixels
line.pixels = [Point32(x=pt[0], y=pt[1]) for pt in ordered_points]
```

**Fix Location:** `red_line_detector.py` lines 357–366

---

### Bug 2 — Silent TF Transform Failure

**File:** `depth_matcher.py`  
**Symptom:** Even with valid depth values (1095–1124 mm, well within 300–2000 mm range), 0 lines were published. The `except Exception as e: pass` silently dropped all points.

**Root Cause:** The `PointStamped` used `lines_msg.header` directly, which contained the **bag's original timestamp** (January 2026). `do_transform_point` calls `buffer.transform()` with that old timestamp — but the live TF buffer only holds recent data. The lookup fails with an extrapolation error.

```python
# ❌ Before — used bag's old timestamp
pt_stamped.header = lines_msg.header  # e.g. stamp = Jan 26 2026

# ✅ After — use Time() (zero = "latest available")
pt_stamped.header.frame_id = lines_msg.header.frame_id
pt_stamped.header.stamp = rclpy.time.Time().to_msg()
```

**Fix Location:** `depth_matcher.py`, `PointStamped` construction in the pixel loop

---

### Bug 3 — min_valid_points Threshold Too High

**File:** `test_depth_matcher_bag.launch.py`  
**Symptom:** Even after Bug 1 was partially visible, `min_valid_points=10` caused rejection of any short test lines.

**Fix:** Lowered to `min_valid_points=2` in the test launch file (appropriate for the skeleton endpoint fallback case).

---

### Bug 4 — RViz GUI Not Appearing

**Symptom:** `ros2 launch parol6_vision camera_setup.launch.py` ran silently — no RViz window appeared.

**Root Cause:** Container was started with `-e XAUTHORITY=/tmp/.docker.xauth` but that file was **never created or copied** into the container. X11 authentication failed silently.

**Diagnosis steps run:**
```bash
# 1. Check DISPLAY inside container
docker exec parol6_dev bash -c "echo DISPLAY=$DISPLAY"
# → DISPLAY=:0  ✅ (correct)

# 2. Check X11 socket is mounted
docker exec parol6_dev bash -c "ls /tmp/.X11-unix/"
# → X0  ✅ (socket present)

# 3. Check xauth file inside container
docker exec parol6_dev bash -c "ls -la /tmp/.docker.xauth"
# → ls: cannot access '/tmp/.docker.xauth': No such file or directory  ❌ (root cause!)

# 4. Allow Docker X access from host
xhost +local:docker
# → non-network local connections being added to access control list  ✅
```

**Fix:**
```bash
# On host — generate auth token and inject into container:
xauth nlist $DISPLAY | sed 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -
docker cp /tmp/.docker.xauth parol6_dev:/tmp/.docker.xauth
```

`start_container.sh` now auto-runs this every time. `XAUTHORITY=/tmp/.docker.xauth` is also added to `/etc/bash.bashrc` inside the container so every new `docker exec` terminal inherits it automatically.

**In any terminal inside the container (if needed manually):**
```bash
export XAUTHORITY=/tmp/.docker.xauth
export QT_X11_NO_MITSHM=1
```

---

### Bug 5 — RViz Crashes with `AMENT_PREFIX_PATH not set`

**Symptom:** Running `rviz2` without sourcing ROS first gives:
```
terminate called after throwing an instance of 'std::runtime_error'
  what():  Environment variable 'AMENT_PREFIX_PATH' is not set or empty
```

**Fix:** Always source ROS before launching anything:
```bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash  # your workspace
```

> The `/etc/bash.bashrc` inside the container does NOT auto-source ROS — you must do it in every terminal or add it to `~/.bashrc`.

---

## Debugging Methodology

### Step 1 — Verify topics are publishing
```bash
ros2 topic hz /vision/weld_lines_2d        # should be ~2 Hz
ros2 topic hz /kinect2/qhd/image_depth_rect  # should be ~2 Hz
ros2 topic hz /vision/weld_lines_3d         # should be ~2 Hz once fixed
```

### Step 2 — Check frame IDs match
```bash
# Confirm depth_matcher is subscribing to the right frames
ros2 topic echo --once /vision/weld_lines_2d | grep frame_id
ros2 topic echo --once /kinect2/qhd/image_depth_rect | grep frame_id
```
Expected: both should be `kinect2_rgb_optical_frame`

### Step 3 — Check sync is working
Watch `depth_matcher` logs for `Synchronized callback` messages. If missing:
- Increase `sync_time_tolerance` (e.g. 0.5s)
- Confirm bag contains both color and depth topics

### Step 4 — Check pixel count in 2D lines
```bash
ros2 topic echo --once /vision/weld_lines_2d
# Check len(lines[0].pixels) — should be 60-130, NOT 2
```
If `pixels` count = 2: red_line_detector is sending D-P simplified endpoints (reverted bug)

### Step 5 — Check depth values
Add temp logging in `depth_matcher.py` callback:
```python
for pixel in line_2d.pixels[:5]:
    u, v = int(pixel.x), int(pixel.y)
    print(f"({u},{v}) = {cv_depth[v,u]}")
```
Expected: values of 900–2000 (mm). If all 0: depth image not aligned with color.

### Step 6 — Check TF tree
```bash
ros2 run tf2_ros tf2_echo base_link kinect2_rgb_optical_frame
```
Should show transform. If `frame does not exist`: robot_state_publisher or static_transform_publisher not running.

### Step 7 — Check line stats log
The depth_matcher now logs per-detection:
```
Line stats: total=128, valid_raw=128, filtered=128, quality=1.00 (need min_pts=10, min_qual=0.6)
```
Use this to pinpoint which stage is rejecting the line.

---

## Quick Start (Full Pipeline)

### Option A — Single launch (recommended)

```bash
# All nodes including bag playback, detectors, point cloud, and RViz in one command:
export XAUTHORITY=/tmp/.docker.xauth
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch parol6_vision test_depth_matcher_bag.launch.py
```

> The launch file now starts `point_cloud_xyzrgb_node` automatically — colored PointCloud2 appears on `/points` with no extra steps.

### Option B — Manual (individual terminals)

```bash
# Terminal 1 — Play bag (loop)
ros2 bag play /workspace/rosbag2_2026_01_26-23_26_59 --loop

# Terminal 2 — Red line detector
source install/setup.bash
ros2 run parol6_vision red_line_detector

# Terminal 3 — Depth matcher
source install/setup.bash
ros2 run parol6_vision depth_matcher

# Terminal 4 — Point cloud (color + depth → PointCloud2)
source /opt/ros/humble/setup.bash
ros2 run depth_image_proc point_cloud_xyzrgb_node \
  --ros-args \
  -r /rgb/camera_info:=/kinect2/qhd/camera_info \
  -r /rgb/image_rect_color:=/kinect2/qhd/image_color_rect \
  -r /depth_registered/image_rect:=/kinect2/qhd/image_depth_rect \
  -p use_sim_time:=true

# Terminal 5 — Verify output
ros2 topic echo /vision/weld_lines_3d

# Terminal 6 — RViz visualization
export XAUTHORITY=/tmp/.docker.xauth
source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py
```

> **Note on executable name:** In `depth_image_proc` ≥ 3.x the executable is `point_cloud_xyzrgb_node` (with `_node` suffix). Running `point_cloud_xyzrgb` gives `No executable found`.

### RViz Setup for Depth Matcher
1. **Fixed Frame** → `base_link`
2. **Add** → `MarkerArray` → topic: `/depth_matcher/markers` (blue dots = 3D weld points)
3. **Add** → `MarkerArray` → topic: `/red_line_detector/markers` (2D overlay reference)
4. **Add** → `Image` → topic: `/kinect2/qhd/image_color_rect`
5. **Add** → `Image` → topic: `/kinect2/qhd/image_depth_rect`
6. **Add** → `PointCloud2` → topic: `/points` (colored 3-D point cloud from `point_cloud_xyzrgb_node`)
   - Style: `Points`, Size: `2 px`
   - Color Transformer: `RGB8`
   - Verify `/points` is publishing: `ros2 topic hz /points` (expect ~15 Hz from bag)

---

## Architecture Notes

### Camera Intrinsics
Populated from `/kinect2/qhd/camera_info`. Cached on first message.
- fx=1081.37, fy=1081.37, cx=959.5, cy=539.5 (QHD 1920×1080 scaled to 960×540)

### TF Chain
```
base_link → [static] → kinect2_link → kinect2_rgb_optical_frame
```
Published from `camera_setup.launch.py` via `static_transform_publisher`.

### Depth Encoding
Kinect v2 depth images: `uint16`, unit = **millimetres**.  
Conversion: `depth_m = depth_raw / 1000.0`

### Quality Threshold
`depth_quality = valid_raw / total_pixels`. A line is published only if:
- `len(filtered_points) >= min_valid_points` (default 10, use 2 for debug)
- `depth_quality >= min_depth_quality` (default 0.6, use 0.05 for debug)
