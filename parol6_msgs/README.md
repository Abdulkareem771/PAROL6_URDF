# parol6_msgs - Custom ROS2 Messages for Welding Vision

**Package:** `parol6_msgs`  
**Type:** ROS2 Message Package  
**Purpose:** Define semantic message types for welding seam detection and 3D reconstruction

---

## Overview

This package provides custom ROS2 message definitions specifically designed for vision-guided robotic welding. The messages encode welding seam geometry rather than using generic object detection messages, providing semantic clarity and thesis-grade documentation.

**Scientific Contribution:**
> *"We defined custom semantic messages that encode welding seam geometry, rather than overloading object-detection message types."*

This demonstrates research maturity and clean system architecture.

---

## Message Definitions

### 1. WeldLine.msg (2D Detection)

Represents a detected welding seam in 2D image space.

**Fields:**
- `string id` - Unique identifier for this weld line
- `float32 confidence` - Detection confidence score (0.0 to 1.0)
- `geometry_msgs/Point32[] pixels` - Ordered points along line centerline in image coordinates
- `geometry_msgs/Point bbox_min` - Bounding box minimum corner
- `geometry_msgs/Point bbox_max` - Bounding box maximum corner  
- `std_msgs/Header header` - Image frame reference

**Confidence Formula:**
```
confidence = (N_valid / N_total) × continuity_score
```

Where:
- `N_valid` = Valid pixels after morphological filtering
- `N_total` = Total pixels initially detected
- `continuity_score` = Line smoothness metric [0,1]

**Usage Example:**
```python
from parol6_msgs.msg import WeldLine
import geometry_msgs.msg import Point32

line = WeldLine()
line.id = "red_line_0"
line.confidence = 0.95
line.pixels = [Point32(x=320.0, y=240.0, z=0.0), ...]
```

---

### 2. WeldLineArray.msg (2D Detection Array)

Array container for multiple 2D weld line detections.

**Fields:**
- `std_msgs/Header header` - Frame reference
- `WeldLine[] lines` - Array of detected weld lines

**Usage Example:**
```python
from parol6_msgs.msg import WeldLineArray

array = WeldLineArray()
array.header.frame_id = "camera_rgb_optical_frame"
array.lines = [line1, line2, ...]
```

---

### 3. WeldLine3D.msg (3D Representation)

Represents a welding seam in 3D space (robot coordinate frame).

**Fields:**
- `string id` - Unique identifier (matches 2D detection)
- `float32 confidence` - Detection confidence (0.0 to 1.0)
- `geometry_msgs/Point[] points` - Ordered 3D points along seam
- `float32 line_width` - Average line width in meters
- `float32 depth_quality` - Percentage of valid depth points (0.0 to 1.0)
- `int32 num_points` - Number of valid 3D points
- `std_msgs/Header header` - Coordinate frame reference (typically "base_link")

**Quality Metrics:**
- `depth_quality > 0.9`: Excellent depth data coverage
- `depth_quality 0.7-0.9`: Good coverage with minor gaps
- `depth_quality < 0.7`: Poor coverage, may need re-detection

**Usage Example:**
```python
from parol6_msgs.msg import WeldLine3D
from geometry_msgs.msg import Point

line_3d = WeldLine3D()
line_3d.id = "red_line_0"
line_3d.confidence = 0.95
line_3d.points = [Point(x=0.4, y=0.1, z=0.3), ...]
line_3d.depth_quality = 0.92
line_3d.num_points = 180
```

---

### 4. WeldLine3DArray.msg (3D Array)

Array container for multiple 3D weld lines.

**Fields:**
- `std_msgs/Header header` - Frame reference
- `WeldLine3D[] lines` - Array of 3D weld lines

---

## Building the Package

### Prerequisites

- ROS2 Humble installed
- Inside PAROL6 Docker container or proper ROS2 workspace

### Build Steps

```bash
# 1. Navigate to workspace root
cd /workspace  # or /home/osama/Desktop/PAROL6_URDF

# 2. Build the package
colcon build --packages-select parol6_msgs

# 3. Source the workspace
source install/setup.bash

# 4. Verify message generation
ros2 interface list | grep parol6_msgs
```

**Expected Output:**
```
parol6_msgs/msg/WeldLine
parol6_msgs/msg/WeldLineArray
parol6_msgs/msg/WeldLine3D
parol6_msgs/msg/WeldLine3DArray
```

### Verify Message Structure

```bash
# View WeldLine message definition
ros2 interface show parol6_msgs/msg/WeldLine

# View WeldLine3D message definition
ros2 interface show parol6_msgs/msg/WeldLine3D
```

---

## Using Messages in Python

### Publisher Example

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from parol6_msgs.msg import WeldLineArray, WeldLine
from geometry_msgs.msg import Point32
from std_msgs.msg import Header

class WeldLinePublisher(Node):
    def __init__(self):
        super().__init__('weld_line_publisher')
        self.publisher = self.create_publisher(
            WeldLineArray,
            '/vision/weld_lines_2d',
            10
        )
        
    def publish_detection(self, line_id, pixels, confidence):
        """Publish a weld line detection"""
        msg = WeldLineArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_rgb_optical_frame"
        
        line = WeldLine()
        line.id = line_id
        line.confidence = confidence
        line.pixels = pixels  # List of Point32
        line.header = msg.header
        
        msg.lines = [line]
        self.publisher.publish(msg)
        self.get_logger().info(f'Published weld line: {line_id}')

# Usage
rclpy.init()
node = WeldLinePublisher()
pixels = [Point32(x=float(i), y=float(i*2), z=0.0) for i in range(100)]
node.publish_detection("test_line", pixels, 0.95)
```

### Subscriber Example

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from parol6_msgs.msg import WeldLineArray

class WeldLineSubscriber(Node):
    def __init__(self):
        super().__init__('weld_line_subscriber')
        self.subscription = self.create_subscription(
            WeldLineArray,
            '/vision/weld_lines_2d',
            self.callback,
            10
        )
        
    def callback(self, msg):
        """Process received weld line detections"""
        self.get_logger().info(f'Received {len(msg.lines)} weld lines')
        
        for line in msg.lines:
            self.get_logger().info(
                f'  Line {line.id}: '
                f'{len(line.pixels)} pixels, '
                f'confidence={line.confidence:.2f}'
            )

# Usage
rclpy.init()
node = WeldLineSubscriber()
rclpy.spin(node)
```

---

## Using Messages in C++

### Publisher Example

```cpp
#include <rclcpp/rclcpp.hpp>
#include "parol6_msgs/msg/weld_line_array.hpp"

class WeldLinePublisher : public rclcpp::Node {
public:
    WeldLinePublisher() : Node("weld_line_publisher") {
        publisher_ = this->create_publisher<parol6_msgs::msg::WeldLineArray>(
            "/vision/weld_lines_2d", 10);
    }
    
    void publishDetection(const std::string& line_id, float confidence) {
        auto msg = parol6_msgs::msg::WeldLineArray();
        msg.header.stamp = this->now();
        msg.header.frame_id = "camera_rgb_optical_frame";
        
        parol6_msgs::msg::WeldLine line;
        line.id = line_id;
        line.confidence = confidence;
        // Add pixels...
        
        msg.lines.push_back(line);
        publisher_->publish(msg);
    }
    
private:
    rclcpp::Publisher<parol6_msgs::msg::WeldLineArray>::SharedPtr publisher_;
};
```

---

## Topic Naming Conventions

When publishing/subscribing to weld line messages:

### Semantic Outputs (Pipeline Communication)
- `/vision/weld_lines_2d` - WeldLineArray (2D detections)
- `/vision/weld_lines_3d` - WeldLine3DArray (3D projections)

### Debug Topics (Development Only)
- `/red_line_detector/debug_image` - sensor_msgs/Image
- `/red_line_detector/markers` - visualization_msgs/MarkerArray

---

## Testing Messages

### Command Line Testing

```bash
# Publish a test message
ros2 topic pub --once /vision/weld_lines_2d parol6_msgs/msg/WeldLineArray \
  "{header: {frame_id: 'camera_rgb_optical_frame'}, lines: []}"

# Echo messages
ros2 topic echo /vision/weld_lines_2d

# Check message rate
ros2 topic hz /vision/weld_lines_2d

# View message info
ros2 topic info /vision/weld_lines_2d
```

### Unit Test Example

```python
import unittest
from parol6_msgs.msg import WeldLine, WeldLine3D
from geometry_msgs.msg import Point32, Point

class TestWeldLineMessages(unittest.TestCase):
    
    def test_weld_line_creation(self):
        """Test creating a 2D weld line message"""
        line = WeldLine()
        line.id = "test"
        line.confidence = 0.95
        line.pixels = [Point32(x=1.0, y=2.0, z=0.0)]
        
        self.assertEqual(line.id, "test")
        self.assertAlmostEqual(line.confidence, 0.95)
        self.assertEqual(len(line.pixels), 1)
    
    def test_weld_line_3d_quality(self):
        """Test 3D weld line quality metrics"""
        line_3d = WeldLine3D()
        line_3d.depth_quality = 0.92
        line_3d.num_points = 180
        
        self.assertGreater(line_3d.depth_quality, 0.9)
        self.assertGreater(line_3d.num_points, 0)

if __name__ == '__main__':
    unittest.main()
```

---

## Troubleshooting

### Issue: Messages not found after building

**Solution:**
```bash
# Re-source the workspace
source /workspace/install/setup.bash

# Or add to ~/.bashrc for persistence
echo "source /workspace/install/setup.bash" >> ~/.bashrc
```

### Issue: Import error in Python

**Problem:**
```
ModuleNotFoundError: No module named 'parol6_msgs'
```

**Solution:**
```bash
# Ensure package is built
colcon build --packages-select parol6_msgs

# Source workspace
source install/setup.bash

# Verify import path
python3 -c "from parol6_msgs.msg import WeldLine; print('Success')"
```

### Issue: CMake errors during build

**Common Causes:**
1. Missing dependencies in `package.xml`
2. Incorrect `CMakeLists.txt` configuration
3. ROS2 environment not sourced

**Solution:**
```bash
# Clean build
rm -rf build/ install/ log/

# Rebuild
colcon build --packages-select parol6_msgs --cmake-clean-cache
```

---

## Integration with Vision Pipeline

This package is used by:

1. **red_line_detector** node → Publishes `WeldLineArray`
2. **depth_matcher** node → Subscribes `WeldLineArray`, publishes `WeldLine3DArray`
3. **path_generator** node → Subscribes `WeldLine3DArray`

See the main [implementation_plan.md](file:///home/osama/.gemini/antigravity/brain/d2ceb7f4-79cd-48c2-8f8b-a2b6c9627e7c/implementation_plan.md) for complete system architecture.

---

## Thesis Documentation

When documenting in your thesis:

**Message Design Section:**
> "Custom ROS2 messages were designed to encode welding seam geometry semantically. The `WeldLine` and `WeldLine3D` message types include confidence metrics and quality indicators, enabling downstream filtering and validation."

**Confidence Metric Section:**
> "Detection confidence is computed as the product of spatial coverage (ratio of valid to total points) and geometric quality (line continuity score), providing a single unified metric for assessing detection reliability."

---

## Related Documentation

- [Implementation Plan](file:///home/osama/.gemini/antigravity/brain/d2ceb7f4-79cd-48c2-8f8b-a2b6c9627e7c/implementation_plan.md)
- [Camera Calibration Guide](file:///home/osama/Desktop/PAROL6_URDF/docs/CAMERA_CALIBRATION_GUIDE.md)
- [ROS System Architecture](file:///home/osama/Desktop/PAROL6_URDF/docs/ROS_SYSTEM_ARCHITECTURE.md)

---

**Version:** 1.0.0  
**Author:** PAROL6 Team  
**License:** MIT
