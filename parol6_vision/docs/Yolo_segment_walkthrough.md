# Yolo_segment ROS 2 Node – Walkthrough

## What Was Done

Converted [phase_2_first_mode.py](file:///home/osama/Desktop/PAROL6_URDF/venvs/vision_venvs/ultralytics_cpu_env/YOLO_resources/phase_2_first_mode.py) into a proper ROS 2 node and added it to the [parol6_vision](file:///home/osama/Desktop/PAROL6_URDF/parol6_vision) package.

---

## Files Changed

| Action | File |
|--------|------|
| **NEW** | [yolo_segment.py](file:///home/osama/Desktop/PAROL6_URDF/parol6_vision/parol6_vision/yolo_segment.py) |
| **MODIFIED** | [setup.py](file:///home/osama/Desktop/PAROL6_URDF/parol6_vision/setup.py) |

```diff:setup.py
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'parol6_vision'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        # Include config files (.yaml and .rviz)
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='PAROL6 Team',
    maintainer_email='your.email@example.com',
    description='Vision-guided welding path detection for PAROL6 robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'red_line_detector = parol6_vision.red_line_detector:main',
            'depth_matcher = parol6_vision.depth_matcher:main',
            'path_generator = parol6_vision.path_generator:main',
            'moveit_controller = parol6_vision.moveit_controller:main',
            'dummy_joint_publisher = parol6_vision.dummy_joint_publisher:main',
            'hsv_inspector = parol6_vision.hsv_inspector_node:main',
            'capture_images = parol6_vision.capture_images_node:main',
            'read_image = parol6_vision.read_image_node:main',
        ],
    },
    scripts=['test/mock_camera_publisher.py', 'test/check_path.py'],
)
===
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'parol6_vision'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        # Include config files (.yaml and .rviz)
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='PAROL6 Team',
    maintainer_email='your.email@example.com',
    description='Vision-guided welding path detection for PAROL6 robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'red_line_detector = parol6_vision.red_line_detector:main',
            'depth_matcher = parol6_vision.depth_matcher:main',
            'path_generator = parol6_vision.path_generator:main',
            'moveit_controller = parol6_vision.moveit_controller:main',
            'dummy_joint_publisher = parol6_vision.dummy_joint_publisher:main',
            'hsv_inspector = parol6_vision.hsv_inspector_node:main',
            'capture_images = parol6_vision.capture_images_node:main',
            'read_image = parol6_vision.read_image_node:main',
            'yolo_segment = parol6_vision.yolo_segment:main',
        ],
    },
    scripts=['test/mock_camera_publisher.py', 'test/check_path.py'],
)
```

---

## Node Summary

| Property | Value |
|---|---|
| **Node name** | `Yolo_segment` |
| **Class** | [YoloSegmentNode](file:///home/osama/Desktop/PAROL6_URDF/parol6_vision/parol6_vision/yolo_segment.py#90-366) |
| **Entry point** | `ros2 run parol6_vision yolo_segment` |

### Subscribed Topics
| Topic | Type |
|---|---|
| `/vision/captured_image_color` | `sensor_msgs/Image` |

### Published Topics
| Topic | Type | Description |
|---|---|---|
| `/yolo_segment/annotated_image` | `sensor_msgs/Image` (bgr8) | Raw frame + filled red intersection contour only |
| `/yolo_segment/debug_image` | `sensor_msgs/Image` (bgr8) | All debug layers (blue=original, green=expanded, red=intersection) |
| `/yolo_segment/seam_centroid` | `geometry_msgs/PointStamped` | Pixel-space centroid of the seam intersection |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| `model_path` | `/workspace/venvs/vision_venvs/ultralytics_cpu_env/yolo_segmentation_models_results/experiment_2/weights/best.pt` | Path to YOLO weights |
| `image_topic` | `/vision/captured_image_color` | Input camera topic |
| `expand_px` | `8` | Dilation radius for mask expansion |
| `publish_debug` | `true` | Enable full debug overlay topic |

---

## Build Verification

```
colcon build --packages-select parol6_vision --symlink-install
# → Finished <<< parol6_vision [1.37s]  ✅
```

```
ros2 pkg executables parol6_vision
# → parol6_vision yolo_segment  ✅  (and all other nodes still present)
```

---

## How to Run

```bash
# Inside the Docker container, after sourcing ROS:
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

ros2 run parol6_vision yolo_segment
```

Override the model path or any other parameter at launch:
```bash
ros2 run parol6_vision yolo_segment --ros-args \
  -p model_path:=/path/to/your/best.pt \
  -p expand_px:=12 \
  -p publish_debug:=false
```
