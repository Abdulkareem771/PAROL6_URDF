# PAROL6 Developer Guide

This guide explains how to extend the PAROL6 project, add new nodes, and integrate sensors.

## 1. Project Structure

- **`PAROL6/`**: Contains the robot description (URDF), meshes, and basic launch files.
- **`parol6_moveit_config/`**: Contains MoveIt configuration (SRDF), controllers, and MoveIt launch files.
- **`docs/`**: Documentation.

## 2. Managing Dependencies

If you add new packages or dependencies, you must update the `package.xml` file in the relevant package.

To install dependencies:
```bash
./setup_dependencies.sh
```
This runs `rosdep install` to automatically fetch missing packages.

## 3. Creating a New ROS 2 Node

To add custom logic (e.g., a camera processor or a custom controller), create a new package or add a script to an existing one.

### Option A: Python Script (Quickest)

1. Create a file `my_script.py` in `PAROL6/scripts/` (create the folder if needed).
2. Make it executable: `chmod +x my_script.py`.
3. Example code:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        self.get_logger().info(f'Joints: {msg.name}')

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Option B: New Package (Recommended for larger features)

```bash
cd /workspace
ros2 pkg create --build-type ament_python my_new_package --dependencies rclpy
```

## 4. Connecting Sensors (e.g., Kinect, Realsense)

To add a camera like a Kinect or Realsense:

1.  **Update URDF**: Add the camera link and joint to `PAROL6/urdf/PAROL6.urdf`.
    ```xml
    <link name="camera_link">
      <visual> ... </visual>
    </link>
    <joint name="camera_joint" type="fixed">
      <parent link="world"/>
      <child link="camera_link"/>
      <origin xyz="1.0 0 0.5" rpy="0 0 3.14"/>
    </joint>
    ```

2.  **Add Gazebo Plugin**: Add the camera sensor plugin to the URDF so it works in simulation.
    ```xml
    <gazebo reference="camera_link">
      <sensor type="camera" name="camera1">
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          ...
        </plugin>
      </sensor>
    </gazebo>
    ```

3.  **Launch**: The camera topics (e.g., `/camera/image_raw`) will appear automatically when you run `./start_ignition.sh`.

## 5. Visualizing the System

To see how nodes are connected:

```bash
# Inside the container
rqt_graph
```

This opens a GUI showing all nodes and topics. It's great for debugging connections.

## 6. Using `uv` or `pip`

The container comes with system Python. If you need specific Python libraries:

```bash
# Inside container
pip install package_name
```

**Note**: These changes are lost if you delete the container. To make them permanent, add them to the `Dockerfile` (if you have access) or create a `requirements.txt` and run `pip install -r requirements.txt` in your startup script.

## 7. Best Practices

- **Always source setup files**: `source /workspace/install/setup.bash`
- **Use `colcon build`**: Run this in `/workspace` after changing C++ code or package configuration.
- **Keep it modular**: Don't modify `PAROL6` core files if you can avoid it. Create a separate package for your application logic.
