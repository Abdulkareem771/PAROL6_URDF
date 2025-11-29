import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Joy, JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import time

class XboxActionController(Node):
    def __init__(self):
        super().__init__('xbox_action_controller',
                        parameter_overrides=[
                            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
                        ])
        
        # Use ACTION client instead of topic publisher
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/parol6_arm_controller/follow_joint_trajectory'
        )
        
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.state_sub = self.create_subscription(JointState, '/joint_states', self.state_callback, 10)
        
        # Current joint positions (initialized to None until first state received)
        self.joint_positions = None
        self.deadzone = 0.15
        self.sensitivity = 0.05  # Reduced for smoother control
        
        # Joint names from your PAROL6 robot
        self.joint_names = [
            'joint_L1',  # Base rotation
            'joint_L2',  # Shoulder
            'joint_L3',  # Elbow
            'joint_L4',  # Wrist pitch
            'joint_L5',  # Wrist roll
            'joint_L6'   # Gripper (fixed for now)
        ]
        
        self.last_send_time = time.time()
        self.send_interval = 0.1  # Send commands every 100ms max
        
        self.get_logger().info('🎮 Xbox ACTION Controller Started!')
        self.get_logger().info('Waiting for action server and joint states...')
        
        # Wait for action server
        self._action_client.wait_for_server()
        self.get_logger().info('✅ Action server connected!')

    def state_callback(self, msg):
        # Initialize positions from robot state if not yet set
        if self.joint_positions is None:
            self.joint_positions = [0.0] * 6
            # Map incoming joint states to our order
            for i, name in enumerate(self.joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    self.joint_positions[i] = msg.position[idx]
            self.get_logger().info(f'✅ Initialized positions: {[f"{p:.2f}" for p in self.joint_positions]}')

    def apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def joy_callback(self, msg):
        if self.joint_positions is None:
            return

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
                left_x * self.sensitivity,      # Base rotation
                -left_y * self.sensitivity,     # Shoulder lift (inverted)
                right_x * self.sensitivity,     # Elbow bend
                -right_y * self.sensitivity,    # Wrist pitch (inverted)
                wrist_roll * 0.02,              # Wrist roll
                0.0                             # Gripper
            ]
            
            # Update joint positions
            for i in range(6):
                self.joint_positions[i] += position_changes[i]
                # Apply joint limits (approximate)
                self.joint_positions[i] = max(-3.14, min(3.14, self.joint_positions[i]))
            
            # Rate limit sending
            current_time = time.time()
            if current_time - self.last_send_time >= self.send_interval:
                self.send_goal()
                self.last_send_time = current_time
            
            # Handle buttons
            if msg.buttons[0]:  # A button - Reset to zero
                self.joint_positions = [0.0] * 6
                self.send_goal()
                self.get_logger().info('🔄 Reset to zero position')
                
            if msg.buttons[1]:  # B button - Home position
                self.joint_positions = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
                self.send_goal()
                self.get_logger().info('🏠 Home position')
                
        except Exception as e:
            self.get_logger().error(f'Controller error: {e}')

    def send_goal(self):
        # Create action goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = self.joint_positions.copy()
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 200000000  # 0.2 seconds
        
        goal_msg.trajectory.points = [point]
        
        # Send goal asynchronously (don't wait for result)
        self._action_client.send_goal_async(goal_msg)

def main():
    rclpy.init()
    node = XboxActionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Controller stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
