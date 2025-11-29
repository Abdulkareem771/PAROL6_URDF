import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

class XboxControllerNode(Node):
    def __init__(self):
        super().__init__('xbox_controller_node')
        
        # Try different control methods - Gazebo typically uses JointTrajectory
        self.joint_traj_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.joint_commands_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.joint_group_pub = self.create_publisher(Float64MultiArray, '/joint_group_velocity_controller/commands', 10)
        
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        
        # Current joint states
        self.current_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.deadzone = 0.1
        self.sensitivity = 0.5
        self.last_publish_time = time.time()
        
        self.get_logger().info('🎮 Xbox Controller Node Started - Testing Gazebo Control...')

    def apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def joy_callback(self, msg):
        try:
            # Xbox 360 controller mapping
            left_x = self.apply_deadzone(msg.axes[0])   # Left stick X
            left_y = self.apply_deadzone(msg.axes[1])   # Left stick Y
            right_x = self.apply_deadzone(msg.axes[3])  # Right stick X
            right_y = self.apply_deadzone(msg.axes[4])  # Right stick Y
            
            # Triggers for wrist roll
            left_trigger = (msg.axes[2] + 1) / 2   # Convert from -1:1 to 0:1
            right_trigger = (msg.axes[5] + 1) / 2  # Convert from -1:1 to 0:1
            wrist_roll = right_trigger - left_trigger
            
            # Calculate joint increments
            joint_increments = [
                left_x * self.sensitivity * 0.1,      # Joint_L1 - Base
                -left_y * self.sensitivity * 0.1,     # Joint_L2 - Shoulder  
                right_x * self.sensitivity * 0.1,     # Joint_L3 - Elbow
                -right_y * self.sensitivity * 0.1,    # Joint_L4 - Wrist pitch
                wrist_roll * self.sensitivity * 0.05, # Joint_L5 - Wrist roll
                0.0                                   # Joint_L6
            ]
            
            # Update current joints
            for i in range(6):
                self.current_joints[i] += joint_increments[i]
                # Simple joint limits
                self.current_joints[i] = max(-3.14, min(3.14, self.current_joints[i]))
            
            # Publish to multiple control methods to see what works
            self.publish_joint_commands()
            self.publish_joint_trajectory()
            
            # Handle buttons
            if msg.buttons[0]:  # A button - Home
                self.send_home_command()
            if msg.buttons[1]:  # B button - Stop
                self.emergency_stop()
                
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def publish_joint_commands(self):
        # Method 1: Direct joint commands
        joint_msg = Float64MultiArray()
        joint_msg.data = self.current_joints
        self.joint_commands_pub.publish(joint_msg)

    def publish_joint_trajectory(self):
        # Method 2: Joint trajectory (common for Gazebo)
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']
        
        point = JointTrajectoryPoint()
        point.positions = self.current_joints
        point.time_from_start.sec = 1
        
        traj_msg.points.append(point)
        self.joint_traj_pub.publish(traj_msg)

    def send_home_command(self):
        home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.current_joints = home_position
        
        # Publish home to all methods
        joint_msg = Float64MultiArray()
        joint_msg.data = home_position
        self.joint_commands_pub.publish(joint_msg)
        
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']
        point = JointTrajectoryPoint()
        point.positions = home_position
        point.time_from_start.sec = 1
        traj_msg.points.append(point)
        self.joint_traj_pub.publish(traj_msg)
        
        self.get_logger().info('🏠 Home position sent')

    def emergency_stop(self):
        self.get_logger().warn('🛑 STOP')

def main():
    rclpy.init()
    node = XboxControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\\n🛑 Controller stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
