# Unified Pipeline — Implementation Walkthrough

## What Was Done

### New Nodes

**[inject_path_node.py](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/parol6_vision/inject_path_node.py)**  
Subscribes to `/vision/inject_path` (VOLATILE, persistent ROS subscriber).  
On receipt: re-publishes to `/vision/welding_path/injected` as **TRANSIENT_LOCAL**.  
Replaces all shell injectors.

**[path_holder.py](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/parol6_vision/path_holder.py)**  
The **only** node that publishes `/vision/welding_path` (TRANSIENT_LOCAL).  
Caches `latest_generated` and `latest_injected` separately.  
On source switch: immediately republishes from cache (or fails cleanly if cache is empty).  
Services: `~/set_source`, `~/get_status`.

### Modified Files

| File | Change |
|---|---|
| [path_generator.py](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/parol6_vision/path_generator.py) | Output topic → `/vision/welding_path/generated` (TRANSIENT_LOCAL). Removed `auto_generate` param. Now fires reactively on every `WeldLine3DArray`. Kept `~/trigger_path_generation` service. |
| [moveit_controller.py](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/parol6_vision/moveit_controller.py) | Subscriber QoS `VOLATILE` → `TRANSIENT_LOCAL`. Safe: [path_holder](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/scripts/vision_pipeline_gui.py#3199-3232) is now the sole TRANSIENT_LOCAL publisher. |
| [setup.py](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/setup.py) | Added `inject_path` and [path_holder](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/scripts/vision_pipeline_gui.py#3199-3232) entry points. |
| [live_pipeline.launch.py](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/launch/live_pipeline.launch.py) | Added `inject_path_node` and `path_holder_node` with `active_source: generated`. |
| [vision_pipeline_gui.py](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/scripts/vision_pipeline_gui.py) | **Inject button**: now publishes via persistent `self._ros_node` publisher to `/vision/inject_path`, then calls `path_holder/set_source injected`. Zero subprocess, zero DDS race. **Stage 4 button**: [_send_path_to_moveit()](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/scripts/vision_pipeline_gui.py#3484-3559) stripped to a single execute service call. **Removed**: `_welding_path_sub`, `_latest_welding_path`, `_welding_path_pub`, `_publish_cached_welding_path()`, `_pathgen_trigger_client`. **Added**: [_switch_path_holder_source()](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/scripts/vision_pipeline_gui.py#3199-3232) helper. |

---

## Final Pipeline Flows

**Flow A — Live Vision:**
```
capture_images → path_optimizer → depth_matcher → path_generator
      → /vision/welding_path/generated (TRANSIENT_LOCAL)
      → path_holder (active=generated) → /vision/welding_path (TRANSIENT_LOCAL)
      → [GUI: Send to MoveIt] → execute service → moveit_controller → robot
```

**Flow B — Inject Test:**
```
GUI inject button → /vision/inject_path → inject_path_node
      → /vision/welding_path/injected (TRANSIENT_LOCAL)
      → path_holder switches to active=injected → /vision/welding_path (TRANSIENT_LOCAL)
      → [GUI: Send to MoveIt] → execute service → moveit_controller → robot
```

---

## 1-Click Execution Dashboard
A new "Simple Workflow" tab was added to the vision GUI, replacing the step-by-step NodeButtons list with a direct 3-step macro interface: **Start Pipeline**, **Capture & Generate Path**, and **Execute Weld**. It wraps around the existing node subprocesses cleanly.

![Final Simple Workflow Dashboard](/home/kareem/.gemini/antigravity/brain/0c0cf47b-ac4d-428b-821f-da1e7ee2c5e6/final_gui_simple_workflow_1774798514494.png)

## Robustification & Quality of Life
**1) Duplicate Process Extermination:** The GUI's [NodeWorker](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/scripts/vision_pipeline_gui.py#217-288) now vigorously runs `pkill -f` before spawning processes to avoid multiple overlapping ROS capture/CV nodes crashing the data stream.
**2) OMPL Trajectory Capping:** [path_generator.py](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/parol6_vision/path_generator.py) bounds output waypoints (configurable up to N, default 80). If drawn strokes exceed this length, they are down-sampled dynamically, entirely preventing the persistent "goal tree sampling failure" error in OMPL.

### 2. MoveIt Planning & OMPL Robustness
- **Coordinate-Aware Logging**: Enhanced `moveit_controller` to log 4-decimal (X,Y,Z) coordinates during planning, making it easy to spot unreachable waypoints.
- **Reachability Safety Toggle**: Added an "Enforce Reachable Workspace" checkbox in the GUI. When enabled, it automatically clamps path waypoints to the robot's physical reach before planning, preventing OMPL "Goal Tree" sampler failures.
- **XYZ Path Offsets**: Added ±50mm sliders for live welding adjustments (X, Y, and Z axes) applied directly in the controller's path callback.

### 3. Real-Time Feedback System
- **Truthful Status**: Replaced timers with real-world OS process polling and ROS topic confirmations (`[SCAN]`, `[PATH]`, `[OK]`).
- **GUI Robustness**: Fixed layout overlaps in Step 2 and replaced Unicode emojis with ASCII-safe tags to prevent rendering issues ('tofu' boxes).

**4) Dynamic Parameter UI:** Parameters like `waypoints_spacing` and `max_waypoints` are controlled via the ROS2 `SetParameters` endpoint straight from the new GUI "Settings" tab, requiring no node resets.
**5) Stroke Serialization:** Hand-drawn red lines can now be exported natively as reusable JSON representations along with brush properties on disk via the *Save Profile / Load Profile* buttons in the Manual tools.
**6) Deduplication & Filtering:** [path_holder.py](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/parol6_vision/path_holder.py) actively guards against sending zero-changed paths to MoveIt using geometric hashing, and [path_optimizer.py](file:///home/kareem/Desktop/PAROL6_URDF/parol6_vision/parol6_vision/path_optimizer.py) silently discards sub-5-pixel structural artifacts extracted from CV skeletons.  

---

## Validation Commands

```bash
# 1. Build in container
docker exec parol6_dev bash -lc "cd /workspace && colcon build --packages-select parol6_vision"

# 2. Confirm single publisher
ros2 topic info /vision/welding_path --verbose
# Expected: 1 publisher (path_holder), TRANSIENT_LOCAL

# 3. Check path_holder status
ros2 service call /path_holder/get_status std_srvs/srv/Trigger '{}'

# 4. Switch source manually
ros2 param set /path_holder active_source injected
ros2 service call /path_holder/set_source std_srvs/srv/Trigger '{}'
```
