# Capture Images Node Guide

## Overview

The `capture_images_node` (`capture_images_node.py`) is the entry point (Stage 1) for the PAROL6 Vision Pipeline. This ROS 2 node is strictly responsible for ingesting colour and depth imagery from a Kinect v2 sensor, perfectly synchronizing them, and selectively publishing matched pairs to downstream processing topics (e.g., cropping, depth matching, red line detection).

By selectively capturing frames instead of continuously streaming them, the pipeline saves significant computational overhead.

## Node Architecture

### Subscribed Topics

- `/kinect2/sd/image_color_rect` (`sensor_msgs/Image`): The rectified, standard-definition colour image from the Kinect v2 sensor.
- `/kinect2/sd/image_depth_rect` (`sensor_msgs/Image`): The aligned depth image perfectly registered to the color stream.
- `/kinect2/sd/camera_info` (`sensor_msgs/CameraInfo`): The camera intrinsics necessary for 3D point cloud generation down the line.
- `/vision/capture_trigger` (`std_msgs/Empty`): An external trigger topic. When a message is received here, the node immediately publishes the latest valid synchronized frame pair.

### Published Topics

- `/vision/captured_image_raw` (`sensor_msgs/Image`): The captured colour frame. The topic name can be overridden using the `output_topic` parameter.
- `/vision/captured_image_depth` (`sensor_msgs/Image`): The correspondingly synced depth frame.
- `/vision/captured_camera_info` (`sensor_msgs/CameraInfo`): Relayed `CameraInfo` stream for downstream mathematical projections.

### Parameters

- `capture_mode` (string, default: `'keyboard'`): Mechanism used to trigger a capture. Supported options are `'keyboard'` or `'timed'`.
- `frame_time` (float, default: `10.0`): Automatic interval timer (in seconds) between captures, used strictly when `capture_mode` is set to `'timed'`.
- `output_topic` (string, default: `'/vision/captured_image_raw'`): Target ROS topic for publishing the captured colour image.

---

## Modes of Operation

### Keyboard Mode (Default)
In Keyboard mode, a daemonized background thread listens to standard input (stdin) in the terminal running the node.
- Entering **`s` + `Enter`** flags the node that a capture is requested.
- As soon as the next valid, synchronized color + depth pair arrives from the Kinect v2, the node will pair them and publish to the `/vision/` topics.
- Any other keys pressed are ignored.

### Timed Mode
In Timed mode, the node circumvents the need for manual interaction. A ROS timer fires uninterruptedly every `frame_time` seconds. When the timer pops, the node grabs the most recently cached, perfectly synchronized image pair and immediately publishes it.

### Topic Trigger (Interactive GUI Mode)
Regardless of the active `capture_mode`, the node perpetually listens to the `/vision/capture_trigger` topic. This is essential for operations driven by the `vision_pipeline_gui`.
- Because the Kinect v2 bridge allows framerate throttling (e.g., `fps_limit=1.0`), waiting for a *new* synchronized pair upon receiving a capture request introduces unnatural UI latency.
- To counter this, sending an `Empty` message to`/vision/capture_trigger` bypasses the asynchronous wait and instantly pairs and publishes the most recently cached images cached by the subscriber hook. 
- If no images have ever been synced yet, it gracefully falls back, flagging a pending request so the very first pair processed is automatically published.

---

## Detailed Implementation Workflow

1. **Initialization:** Instantiates ROS publishers, subscribers, handles parameter validation, and sets up synchronization policies.
2. **Synchronization Engine (`_sync_callback`):** Utilizes `message_filters.ApproximateTimeSynchronizer` (with a queue depth of 10 and a slop of 0.1s) to enforce temporally matched frames between colour and depth. Every time a new matched pair is recognized, it's safely cached behind a `threading.Lock()` as `_latest_color` and `_latest_depth`.
3. **Triggers Evaluated:** 
    - The `_keyboard_listener` evaluates stdin inputs, setting the `_save_requested` thread event on an `'s'` keystroke.
    - The `_timed_trigger` executes predictably via the ROS timer layer checking the cache lock.
    - The `_trigger_callback` fires instantly upon a ROS topic arrival.
4. **Broadcast (`_do_publish`):** Both `Image` topics are propagated forward natively, while the independent `_camera_info_callback` mirrors the intrinsic `CameraInfo` separately to unblock downstream point mapping logic.
