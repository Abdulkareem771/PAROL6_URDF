import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

class XboxTrajectoryController(Node):
    def __init__(self):
        super().__init__('xbox_trajectory_controller')
        
        # Use the correct trajectory topic for Gazebo
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, 
            '/parol6_arm_controller/joint_trajectory', 
            10
        )
        
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        
        # Current joint positions
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.deadzone = 0.15
        self.sensitivity = 0.8
        
        # Joint names from your PAROL6 robot
        self.joint_names = [
            'joint_L1',  # Base rotation
            'joint_L2',  # Shoulder
            'joint_L3',  # Elbow
            'joint_L4',  # Wrist pitch
            'joint_L5',  # Wrist roll
            'joint_L6'   # Gripper (fixed for now)
        ]
        
        self.get_logger().info('🎮 Xbox Trajectory Controller Started!')
        self.get_logger().info('Using: /parol6_arm_controller/joint_trajectory')

    def apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def joy_callback(self, msg):
        try:
            # Xbox 360 controller mapping
            left_x = self.apply_deadzone(msg.axes[0])   # Left stick X -> Base
            left_y = self.apply_deadzone(msg.axes[1])   # Left stick Y -> Shoulder
            right_x = self.apply_deadzone(msg.axes[3])  # Right stick X -> Elbow
            right_y = self.apply_deadzone(msg.axes[4])  # Right stick Y -> Wrist pitch
            
            # Triggers for wrist roll
            left_trigger = msg.axes[2]  # LT: -1 to 1
            right_trigger = msg.axes[5] # RT: -1 to 1
            wrist_roll = (right_trigger - left_trigger) * 0.5
            
            # Calculate position changes
            position_changes = [
                left_x * 0.05,      # Base rotation
                -left_y * 0.05,     # Shoulder lift (inverted)
                right_x * 0.05,     # Elbow bend
                -right_y * 0.05,    # Wrist pitch (inverted)
                wrist_roll * 0.02,  # Wrist roll
                0.0                 # Gripper
            ]
            
            # Update joint positions
            for i in range(6):
                self.joint_positions[i] += position_changes[i]
                # Apply joint limits
                self.joint_positions[i] = max(-3.14, min(3.14, self.joint_positions[i]))
            
            # Publish trajectory command
            self.publish_trajectory()
            
            # Handle buttons
            if msg.buttons[0]:  # A button - Reset to zero
                self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.publish_trajectory()
                self.get_logger().info('🔄 Reset to zero position')
                
            if msg.buttons[1]:  # B button - Home position
                self.joint_positions = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]  # Nice home pose
                self.publish_trajectory()
                self.get_logger().info('🏠 Home position')
                
        except Exception as e:
            self.get_logger().error(f'Controller error: {e}')

    def publish_trajectory(self):
        # Create trajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = self.joint_positions.copy()
        point.velocities = [0.0] * 6
        point.accelerations = [0.0] * 6
        point.effort = [0.0] * 6
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 500000000  # 0.5 seconds
        
        traj_msg.points.append(point)
        
        # Publish the trajectory
        self.trajectory_pub.publish(traj_msg)
        
        # Log current positions occasionally
        if time.time() % 5 < 0.1:  # Log every ~5 seconds
            self.get_logger().info(f'Positions: {[f"{p:.2f}" for p in self.joint_positions]}')

def main():
    rclpy.init()
    node = XboxTrajectoryController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Controller stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
