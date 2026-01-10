# Vision-Guided Welding System - Development Plan

**Ultimate Goal**: Automated welding/gluing system that:
1. **Captures** workspace image with Kinect
2. **Detects** workpiece using YOLO
3. **Extracts** ROI (welding seam/edge) from detected object
4. **Generates** smooth welding path (B-spline)
5. **Executes** path with mkservo42c closed-loop control

**Your Role**: Implement the vision pipeline that goes from camera image → welding path coordinates

---

## 🎯 The Complete Workflow

```
Camera → YOLO Detection → ROI Extraction → 3D Projection → Path Generation → Robot Execution
  ↓           ↓                ↓                ↓                ↓                ↓
Kinect    Workpiece      Find Seam        Convert to      B-Spline        MoveIt +
RGB+Depth  Bounding       Within Bbox      World (X,Y,Z)   Smoothing       mkservo42c
          Box                              Coordinates
```

**Focus**: Steps 2-5 (Detection → Path Generation)


---

## 📋 Development Phases

### Phase 1: YOLO Workpiece Detection (Week 1-2)
**Goal**: Detect metal plates/workpieces in camera view

**Deliverables**:
1. `yolo_detector.py` - ROS node running YOLOv8
2. Test with pre-trained model (detect generic objects first)
3. Collect dataset of workpieces for custom training

**Output**: `vision_msgs/Detection2DArray` with bounding boxes

---

### Phase 2: ROI Seam Extraction (Week 3-4) ← **YOUR MAIN TASK**
**Goal**: From detected workpiece bbox, find the welding seam/edge

**What You'll Build**:
```python
# Input: Bounding box of detected workpiece
# Output: List of 2D points representing the seam

def extract_seam(rgb_crop, depth_crop):
    # 1. Edge detection (Canny)
    # 2. Line/contour extraction
    # 3. Filter for welding seam (longest edge, specific orientation)
    # 4. Return ordered points along seam
    pass
```

**Techniques**:
- Canny edge detection
- Hough line transform (for straight welds)
- Contour following (for curved welds)
- Optional: Use depth discontinuities to find seam

**Deliverable**: `seam_extractor.py` node

---

### Phase 3: 3D Projection (Week 4-5)
**Goal**: Convert 2D seam points → 3D world coordinates

**What You'll Build**:
```python
# For each 2D point on the seam:
# 1. Get depth at that pixel
# 2. Project (u,v,depth) → (x,y,z) camera frame
# 3. Transform camera frame → robot base frame
# 4. Return 3D path waypoints
```

**Deliverable**: `camera_utils.py` (projection functions)

---

### Phase 4: Path Smoothing (Week 6)
**Goal**: Convert raw 3D points → smooth B-spline welding path

**What You'll Build**:
```python
from scipy.interpolate import splprep, splev

def generate_welding_path(rough_points_3d, num_waypoints=100):
    # Fit B-spline through points
    # Resample at uniform intervals
    # Add approach/retract moves
    # Return smooth path
    pass
```

**Deliverable**: `path_generator.py` node

---

## 📝 Detailed Task Breakdown

### Task 1: Seam Extraction (Priority 1) ← **START HERE**

**File**: `parol6_vision/seam_extractor.py`

#### Algorithm

```python
def extract_seam_from_bbox(rgb_image, bbox_2d, depth_image):
    """
    Extract welding seam from detected workpiece
    
    Args:
        rgb_image: Full RGB image
        bbox_2d: (x_min, y_min, x_max, y_max) of workpiece
        depth_image: Aligned depth image
    
    Returns:
        seam_points_2d: List of (u,v) pixel coordinates along seam
    """
    x_min, y_min, x_max, y_max = bbox_2d
    
    # Step 1: Crop to workpiece
    roi_rgb = rgb_image[y_min:y_max, x_min:x_max]
    roi_depth = depth_image[y_min:y_max, x_min:x_max]
    
    # Step 2: Edge detection
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    
    # Step 3: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 4: Select seam contour
    # Strategy 1: Longest contour
    seam_contour = max(contours, key=cv2.contourArea)
    
    # OR Strategy 2: Use depth discontinuity
    # depth_edges = find_depth_discontinuities(roi_depth)
    # seam_contour = combine_rgb_depth_edges(edges, depth_edges)
    
    # Step 5: Convert to ordered points
    seam_points = contour_to_ordered_points(seam_contour)
    
    # Step 6: Convert from ROI coords to full image coords
    seam_points_global = [(x + x_min, y + y_min) for x, y in seam_points]
    
    return seam_points_global
```

#### Test Cases

1. **Straight weld**: Metal plate with straight edge
2. **L-shaped weld**: Corner joint
3. **Curved weld**: Pipe or circular component

---

### Task 2: 3D Projection

**File**: `parol6_vision/utils/camera_utils.py`

```python
import numpy as np
from scipy.spatial.transform import Rotation

class WeldingCameraProjector:
    def __init__(self, fx, fy, cx, cy, camera_to_base_tf):
        """
        Camera parameters for welding application
        
        Args:
            fx, fy, cx, cy: Camera intrinsics
            camera_to_base_tf: 4x4 transform matrix (camera → robot base)
        """
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        self.T_cam_to_base = camera_to_base_tf
    
    def seam_to_world_path(self, seam_points_2d, depth_image):
        """
        Convert 2D seam points to 3D welding path in robot frame
        
        Returns:
            waypoints_3d: Nx3 array of (x,y,z) in robot base frame
        """
        waypoints_cam = []
        
        for (u, v) in seam_points_2d:
            # Get depth at this point
            depth = depth_image[int(v), int(u)]
            
            if depth > 0:  # Valid depth
                # Backproject to camera frame
                point_cam = self.pixel_to_camera_frame(u, v, depth)
                waypoints_cam.append(point_cam)
        
        # Transform to robot base frame
        waypoints_cam = np.array(waypoints_cam)
        waypoints_base = self.transform_points(waypoints_cam, 
                                                self.T_cam_to_base)
        
        return waypoints_base
    
    def pixel_to_camera_frame(self, u, v, depth):
        """Pinhole camera projection"""
        x = (u - self.K[0, 2]) * depth / self.K[0, 0]
        y = (v - self.K[1, 2]) * depth / self.K[1, 1]
        z = depth
        return np.array([x, y, z])
```

---

### Task 3: Path Generation

**File**: `parol6_vision/path_generator.py`

```python
from scipy.interpolate import splprep, splev
import numpy as np

class WeldingPathGenerator:
    def __init__(self, approach_height=0.05, retract_height=0.05):
        self.approach_height = approach_height  # 5cm above seam
        self.retract_height = retract_height
    
    def generate_smooth_path(self, rough_waypoints_3d, density=100):
        """
        Generate smooth B-spline welding path
        
        Args:
            rough_waypoints_3d: Nx3 array from seam extraction
            density: Number of waypoints in final path
        
        Returns:
            smooth_path: (density)x3 array
        """
        # Fit B-spline (cubic for smooth acceleration)
        tck, u = splprep([rough_waypoints_3d[:,0], 
                          rough_waypoints_3d[:,1],
                          rough_waypoints_3d[:,2]], 
                         s=0, k=3)  # k=3 for cubic
        
        # Resample uniformly
        u_new = np.linspace(0, 1, density)
        smooth_x, smooth_y, smooth_z = splev(u_new, tck)
        
        smooth_path = np.column_stack([smooth_x, smooth_y, smooth_z])
        
        # Add approach and retract
        full_path = self.add_approach_retract(smooth_path)
        
        return full_path
    
    def add_approach_retract(self, weld_path):
        """Add safe approach and retract moves"""
        # Approach: Start 5cm above first point
        approach = weld_path[0].copy()
        approach[2] += self.approach_height
        
        # Retract: End 5cm above last point
        retract = weld_path[-1].copy()
        retract[2] += self.retract_height
        
        return np.vstack([approach, weld_path, retract])
```

---

## 🤖 Revised AI Prompts

### Prompt 1: Seam Detection from Workpiece
```
I'm building a vision-guided welding robot. I need to detect welding seams on metal workpieces.

Context:
- Input: RGB image crop of detected metal plate (from YOLO bbox)
- Input: Aligned depth image crop
- Need: Extract the welding seam (edge/line to weld along)
- Challenges: Lighting variations, reflective surfaces

Task:
Implement a robust seam extraction algorithm that:
1. Uses Canny edge detection on RGB
2. Optionally uses depth discontinuities
3. Filters for the main seam (longest continuous edge)
4. Returns ordered 2D points along the seam

Provide Python implementation with OpenCV.
Include strategies for handling:
- Multiple edges (how to select the weld seam)
- Noisy edge detection
- Curved vs straight seams
```

### Prompt 2: Welding Path Smoothing
```
I have a list of 3D points representing a welding seam extracted from vision.
The points are noisy and unevenly spaced.

Requirements:
- Input: Nx3 array of (x,y,z) coordinates in meters
- Output: Smooth B-spline path with 100 evenly-spaced waypoints
- Constraint: Path must pass close to original points (welding accuracy)
- Add: Approach move (5cm above start) and retract move (5cm above end)

Task:
Implement using scipy.interpolate.splprep/splev.
Show how to:
1. Fit cubic B-spline through points
2. Control smoothness (s parameter)
3. Resample uniformly
4. Add approach/retract moves

Provide complete Python code with visualization.
```

### Prompt 3: Camera Calibration & TF
```
I need to transform detected seam points from camera frame to robot base frame.

Setup:
- Kinect V2 mounted on robot or workbench
- Camera intrinsics: fx=365.4, fy=365.4, cx=254.9, cy=205.3
- Need: 4x4 transformation matrix from camera to robot base

Tasks:
1. Explain hand-eye calibration process
2. Show how to use cv2.solvePnP for calibration
3. Implement point transformation function
4. Validate accuracy (how to measure error)

Provide:
- Calibration procedure (using checkerboard + robot)
- Python code for transformation
- Test cases with known positions
```

### Prompt 4: ROS Integration
```
I have separate functions for:
1. Seam extraction (2D points)
2. 3D projection (camera frame)
3. Path smoothing (B-spline)

Need to integrate into ROS 2 pipeline:

Architecture:
- Subscribe: /yolo/detections, /kinect2/sd/image_color_rect, /kinect2/sd/image_depth_rect
- Process: Detection → Seam → 3D → Smooth path
- Publish: nav_msgs/Path (for MoveIt)

Requirements:
- Synchronize RGB + Depth + Detections
- Process at 5Hz (welding doesn't need real-time)
- Error handling and logging
- Visualize intermediate steps in RViz

Provide complete ROS 2 node implementation.
```

---

## ✅ Success Criteria (Revised for Welding)

**Milestone 1: Seam Detection**
- ✅ Detects straight welds on flat plates
- ✅ Handles L-shaped joints (corners)
- ✅ Works under workshop lighting

**Milestone 2: 3D Accuracy**
- ✅ Seam position accurate to ±5mm
- ✅ Validated with ruler/caliper measurement
- ✅ Consistent across different object distances

**Milestone 3: Path Quality**
- ✅ Smooth B-spline (no sharp corners)
- ✅ Uniform point spacing (~5mm)
- ✅ Approach/retract moves included

**Milestone 4: End-to-End Demo**
- ✅ Place workpiece → Detect → Generate path → Execute weld motion
- ✅ Cycle time <30 seconds
- ✅ Repeatable results

---

## 📊 Final Deliverables

1. **Code**:
   - `seam_extractor.py` ✓
   - `camera_utils.py` ✓
   - `path_generator.py` ✓
   - Integration launch file

2. **Documentation**:
   - Seam detection algorithm explanation
   - Camera calibration procedure
   - Accuracy validation report

3. **Demo Video**:
   - Show workpiece
   - Show detected seam overlay (2D)
   - Show 3D path in RViz
   - Show robot executing motion

4. **Dataset**:
   - 50+ images of workpieces
   - Annotated for custom YOLO training

---

## 🎯 The Endgame: Full Pipeline

```python
# main_welding_workflow.py

def automated_welding_workflow():
    # 1. Capture
    rgb, depth = kinect.get_rgbd()
    
    # 2. Detect
    detections = yolo_detector.detect(rgb)
    workpiece_bbox = detections[0]  # Assuming one workpiece
    
    # 3. Extract Seam
    seam_2d = seam_extractor.extract(rgb, workpiece_bbox, depth)
    
    # 4. Project to 3D
    seam_3d = camera_projector.seam_to_world_path(seam_2d, depth)
    
    # 5. Generate Smooth Path
    welding_path = path_generator.generate_smooth_path(seam_3d)
    
    # 6. Execute
    moveit_interface.execute_cartesian_path(welding_path)
    
    print("Welding complete!")
```

**This is your thesis contribution: End-to-end automated welding from vision to execution.** 🚀

A ROS 2 node that bridges vision (YOLO) and manipulation (MoveIt):

```
YOLO → 2D Bounding Boxes → YOUR NODE → 3D Poses → Path Generator → Robot
```

**Key Feature**: Works with ANY YOLO model (generic → custom)

---

## 📋 Prerequisites

### 1. Setup Vision Environment
```bash
cd /path/to/PAROL6_URDF

# Create Python venv
python3 -m venv venv_vision
source venv_vision/bin/activate

# Install dependencies
pip install ultralytics opencv-python scipy torch open3d
pip freeze > requirements_vision.txt

deactivate
```

### 2. Verify Kinect Works
```bash
# In Docker container
ros2 topic list | grep kinect

# Should see:
# /kinect2/sd/image_color_rect
# /kinect2/sd/image_depth_rect
# /kinect2/sd/points
```

---

## 🏗️ Package Structure

Create `parol6_vision` package:

```bash
cd /workspace
ros2 pkg create parol6_vision --build-type ament_python --dependencies rclpy sensor_msgs geometry_msgs vision_msgs
```

**File Structure:**
```
parol6_vision/
├── parol6_vision/
│   ├── yolo_detector.py       # Provided starter code
│   ├── bbox_matcher.py        # ← YOUR MAIN TASK
│   └── utils/
│       └── camera_utils.py    # ← YOUR HELPER FUNCTIONS
├── launch/
│   └── vision_pipeline.launch.py
├── config/
│   └── kinect_intrinsics.yaml
└── requirements.txt
```

---

## 📝 Task 1: Camera Utilities

**File**: `parol6_vision/utils/camera_utils.py`

**What it does**: Project 2D pixels + depth → 3D world coordinates

### Implementation Template

```python
import numpy as np

class CameraProjector:
    def __init__(self, fx, fy, cx, cy):
        """
        Camera intrinsic parameters
        fx, fy: focal lengths (pixels)
        cx, cy: principal point (pixels)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def pixel_to_3d(self, u, v, depth):
        """
        Convert pixel (u,v) with depth to 3D point (x,y,z)
        
        Args:
            u: pixel column
            v: pixel row
            depth: depth value (meters)
        
        Returns:
            (x, y, z) in camera frame
        """
        # TODO: Implement pinhole camera projection
        # x = (u - cx) * depth / fx
        # y = (v - cy) * depth / fy
        # z = depth
        
        pass
    
    def bbox_to_3d(self, bbox_2d, depth_image):
        """
        Convert 2D bounding box to 3D pose
        
        Args:
            bbox_2d: (x_min, y_min, x_max, y_max)
            depth_image: numpy array of depth values
        
        Returns:
            center_3d: (x, y, z) of bbox center
            size_3d: (width, height, depth) of bbox
        """
        x_min, y_min, x_max, y_max = bbox_2d
        
        # TODO:
        # 1. Extract depth values within bbox
        # 2. Filter invalid depths (0 or NaN)
        # 3. Sample points (e.g., every 5th pixel)
        # 4. Project to 3D
        # 5. Compute center and size
        
        pass
```

### Testing

```python
# test_camera_utils.py
projector = CameraProjector(fx=365.4, fy=365.4, cx=254.9, cy=205.3)

# Test single pixel
depth = 1.5  # 1.5 meters
x, y, z = projector.pixel_to_3d(320, 240, depth)
print(f"3D Point: ({x:.2f}, {y:.2f}, {z:.2f})")

# Test bbox
bbox = (100, 100, 200, 200)
depth_img = np.random.rand(480, 640) * 2.0  # Fake depth
center, size = projector.bbox_to_3d(bbox, depth_img)
```

---

## 📝 Task 2: 2D→3D Matching Node

**File**: `parol6_vision/bbox_matcher.py`

**What it does**: Subscribe to YOLO detections + point cloud → Publish 3D poses

### Implementation Template

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseArray, Pose
import cv2
from cv_bridge import CvBridge
import numpy as np

# Add venv to path
import sys
sys.path.insert(0, '/workspace/venv_vision/lib/python3.10/site-packages')

from parol6_vision.utils.camera_utils import CameraProjector

class BBoxMatcher(Node):
    def __init__(self):
        super().__init__('bbox_matcher')
        
        # Camera intrinsics (from calibration)
        self.projector = CameraProjector(
            fx=365.4, fy=365.4,  # TODO: Load from config
            cx=254.9, cy=205.3
        )
        
        # ROS interfaces
        self.bridge = CvBridge()
        
        # Subscribers
        self.det_sub = self.create_subscription(
            Detection2DArray, '/yolo/detections',
            self.detection_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/kinect2/sd/image_depth_rect',
            self.depth_callback, 10
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(
            PoseArray, '/vision/bbox_3d', 10
        )
        
        # Storage
        self.latest_depth = None
        
        self.get_logger().info("BBox Matcher initialized")
    
    def depth_callback(self, msg):
        """Store latest depth image"""
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    def detection_callback(self, msg):
        """Process YOLO detections"""
        if self.latest_depth is None:
            self.get_logger().warn("No depth data yet")
            return
        
        poses = PoseArray()
        poses.header = msg.header
        
        for detection in msg.detections:
            # TODO:
            # 1. Extract 2D bbox from detection
            # 2. Convert bbox to 3D using camera_utils
            # 3. Create Pose message
            # 4. Append to poses
            
            pass
        
        # Publish 3D poses
        self.pose_pub.publish(poses)
        self.get_logger().info(f"Published {len(poses.poses)} 3D bboxes")

def main():
    rclpy.init()
    node = BBoxMatcher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 🧪 Testing Workflow

### Step 1: Test with Sample Data
```bash
# Terminal 1: Start Kinect
ros2 launch kinect2_ros2 driver.launch.py

# Terminal 2: Run YOLO detector (provided code)
ros2 run parol6_vision yolo_detector

# Terminal 3: Run your matcher
source venv_vision/bin/activate
ros2 run parol6_vision bbox_matcher
```

### Step 2: Visualize in RViz
```bash
# Add displays:
# - Camera (Image) → /kinect2/sd/image_color_rect
# - PoseArray → /vision/bbox_3d
# - TF frames
```

### Step 3: Validate Accuracy
```python
# validation_node.py
# Place known object at measured position
# Run detection → Compare detected 3D pose vs ground truth
# Calculate error (should be <5cm)
```

---

## 🤖 AI Assistant Prompts

### Prompt 1: Implement Camera Projection
```
I'm working on a ROS 2 vision system for a welding robot. I need to implement pinhole camera projection.

Context:
- Input: 2D pixel coordinates (u,v) and depth value
- Camera intrinsics: fx=365.4, fy=365.4, cx=254.9, cy=205.3
- Output: 3D point (x,y,z) in camera frame

Task:
Implement the `pixel_to_3d()` function in Python that converts pixel + depth to 3D coordinates using the pinhole camera model.

Provide:
1. Implementation
2. Test cases
3. Edge case handling (invalid depth, out of bounds)
```

### Prompt 2: Bbox Depth Sampling
```
I need to extract 3D information from a 2D bounding box using a depth image.

Context:
- 2D bbox: (x_min, y_min, x_max, y_max) in pixels
- Depth image: numpy array (480x640) with depth in meters
- Problem: Some pixels have invalid depth (0 or NaN)

Task:
Implement a function that:
1. Extracts depth values within the bbox
2. Filters invalid depths
3. Computes median depth (robust to outliers)
4. Returns center_3d and size_3d of the detected object

Provide robust implementation with error handling.
```

### Prompt 3: ROS 2 Message Conversion
```
I'm building a ROS 2 node that converts vision_msgs/Detection2D to geometry_msgs/Pose.

Context:
- Input: Detection2D with bbox.center and bbox.size_x, size_y
- Need to: Extract (x_min, y_min, x_max, y_max) from Detection2D format
- Output: Pose with position (x,y,z) and orientation (quaternion)

Task:
Show me how to:
1. Parse Detection2D message structure
2. Extract 2D bbox coordinates
3. Convert 3D center to Pose message
4. Set default orientation (facing forward)

Provide code snippets for ROS 2 Humble.
```

### Prompt 4: Node Integration
```
I have two separate functions working:
1. pixel_to_3d(u, v, depth) → (x, y, z)
2. Detection2D parsing

I need to integrate them into a ROS 2 node that:
- Subscribes to /yolo/detections and /kinect2/sd/image_depth_rect
- For each detection, samples depth within bbox
- Projects bbox corners to 3D
- Publishes PoseArray to /vision/bbox_3d

Provide complete node implementation with:
1. Proper synchronization (latest depth + detection)
2. Error handling
3. Logging
4. Performance considerations
```

---

## ✅ Success Criteria

**Your node is working when:**
1. ✅ Subscribes to both topics successfully
2. ✅ Processes detections without errors
3. ✅ Publishes 3D poses at ~10Hz
4. ✅ RViz shows bbox poses aligned with objects
5. ✅ Manual measurement matches detected 3D position (±5cm)

---

## 📊 Deliverables

1. **Code**:
   - `camera_utils.py` (tested)
   - `bbox_matcher.py` (working node)
   
2. **Documentation**:
   - Unit test results
   - Accuracy validation report
   - Usage instructions

3. **Demo**:
   - Video showing:
     - Object in camera view
     - 2D detection overlay
     - 3D pose in RViz
     - Measured vs detected position

---

## 🆘 Getting Help

**Common Issues:**

1. **"No depth data"**
   - Check: `ros2 topic hz /kinect2/sd/image_depth_rect`
   - Verify Kinect is running

2. **"Import Error: ultralytics"**
   - Activate venv: `source venv_vision/bin/activate`
   - Check PYTHONPATH in launch file

3. **"Pose is way off"**
   - Verify camera intrinsics
   - Check depth units (mm vs m)
   - Validate with known object position

**Ask Kareem** if:
- Camera calibration file location
- Coordinate frame conventions (camera → world)
- Integration with MoveIt planning

**Contact**: [Your contact info]

---

## 📚 Additional Resources

- ROS 2 Python Tutorial: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html
- OpenCV Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- vision_msgs: https://github.com/ros-perception/vision_msgs

---

**Good luck! This is a critical component of the vision-guided welding system. Take your time to get it right, and don't hesitate to test incrementally.** 🚀
