import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Joy, JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

class XboxActionController(Node):
    def __init__(self):
        super().__init__('xbox_action_controller',
                        parameter_overrides=[
                            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
                        ])
        
        # Action client with better configuration
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/parol6_arm_controller/follow_joint_trajectory'
        )
        
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.state_sub = self.create_subscription(JointState, '/joint_states', self.state_callback, 10)
        
        # Current joint positions
        self.joint_positions = None
        self.target_positions = None
        
        # Control parameters - OPTIMIZED FOR RESPONSIVENESS
        self.deadzone = 0.15
        self.sensitivity = 0.08  # Increased for faster response
        self.max_speed = 0.5  # Max radians per command
        
        # Joint names
        self.joint_names = [
            'joint_L1',  # Base rotation
            'joint_L2',  # Shoulder
            'joint_L3',  # Elbow
            'joint_L4',  # Wrist pitch
            'joint_L5',  # Wrist roll
            'joint_L6'   # Gripper
        ]
        
        # Joint limits (from URDF)
        self.joint_limits = {
            'joint_L1': (-3.05, 3.05),    # Base
            'joint_L2': (-1.91, 1.91),    # Shoulder
            'joint_L3': (-2.53, 2.53),    # Elbow
            'joint_L4': (-2.70, 2.70),    # Wrist pitch
            'joint_L5': (-6.28, 6.28),    # Wrist roll (continuous)
            'joint_L6': (0.0, 0.0)        # Gripper (fixed for now)
        }
        
        # Timing - FASTER for real-time control
        self.command_duration = 0.05  # 50ms execution time (was 200ms)
        
        # Goal tracking
        self._goal_handle = None
        self._last_goal_time = self.get_clock().now()
        
        # Button state tracking
        self.last_button_state = [0] * 12
        
        self.get_logger().info('🎮 Xbox Action Controller Started!')
        self.get_logger().info('Waiting for action server...')
        
        # Wait for action server
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('❌ Action server not available!')
        else:
            self.get_logger().info('✅ Action server connected!')

    def state_callback(self, msg):
        # Update current positions from robot
        if self.joint_positions is None:
            self.joint_positions = [0.0] * 6
            self.target_positions = [0.0] * 6
            
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.joint_positions[i] = msg.position[idx]
                # Initialize target to current if not set
                if self.target_positions[i] == 0.0:
                    self.target_positions[i] = msg.position[idx]

    def apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        # Apply scaling after deadzone
        sign = 1.0 if value > 0 else -1.0
        return sign * ((abs(value) - self.deadzone) / (1.0 - self.deadzone))

    def clamp_to_limits(self, joint_idx, value):
        """Clamp joint value to its limits"""
        joint_name = self.joint_names[joint_idx]
        min_val, max_val = self.joint_limits[joint_name]
        return max(min_val, min(max_val, value))

    def joy_callback(self, msg):
        if self.joint_positions is None or self.target_positions is None:
            return

        try:
            # Xbox 360 controller mapping with deadzone
            left_x = self.apply_deadzone(msg.axes[0])   # Left stick X
            left_y = self.apply_deadzone(msg.axes[1])   # Left stick Y
            right_x = self.apply_deadzone(msg.axes[3])  # Right stick X
            right_y = self.apply_deadzone(msg.axes[4])  # Right stick Y
            
            # Triggers: -1 (released) to 1 (pressed)
            left_trigger = (1.0 - msg.axes[2]) / 2.0   # 0 to 1
            right_trigger = (1.0 - msg.axes[5]) / 2.0  # 0 to 1
            wrist_input = right_trigger - left_trigger  # -1 to 1
            
            # Calculate velocity commands (radians per command)
            velocity_commands = [
                left_x * self.sensitivity,          # Base (L stick X)
                -left_y * self.sensitivity,         # Shoulder (L stick Y, inverted)
                right_x * self.sensitivity,         # Elbow (R stick X)
                -right_y * self.sensitivity,        # Wrist pitch (R stick Y, inverted)
                wrist_input * self.sensitivity,     # Wrist roll (triggers)
                0.0                                 # Gripper (fixed)
            ]
            
            # Apply speed limits
            for i in range(6):
                velocity_commands[i] = max(-self.max_speed, min(self.max_speed, velocity_commands[i]))
            
            # Update target positions
            movement_detected = False
            for i in range(6):
                if abs(velocity_commands[i]) > 0.001:
                    self.target_positions[i] += velocity_commands[i]
                    # Clamp to joint limits
                    self.target_positions[i] = self.clamp_to_limits(i, self.target_positions[i])
                    movement_detected = True
            
            # Only send goal if there's movement
            if movement_detected:
                self.send_goal()
            
            # Handle buttons (on press, not hold)
            for i in range(len(msg.buttons)):
                if msg.buttons[i] == 1 and self.last_button_state[i] == 0:
                    self.handle_button_press(i)
                self.last_button_state[i] = msg.buttons[i]
                
        except Exception as e:
            self.get_logger().error(f'Controller error: {e}')

    def handle_button_press(self, button_idx):
        """Handle button press events"""
        if button_idx == 0:  # A button - Reset to zero
            self.target_positions = [0.0] * 6
            self.send_goal()
            self.get_logger().info('🔄 Reset to zero position')
            
        elif button_idx == 1:  # B button - Home position
            self.target_positions = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
            self.send_goal()
            self.get_logger().info('🏠 Moving to home position')
            
        elif button_idx == 2:  # X button - Print current position
            pos_str = ", ".join([f"{p:.2f}" for p in self.joint_positions])
            self.get_logger().info(f'📍 Current: [{pos_str}]')

    def send_goal(self):
        """Send goal to action server - NON-BLOCKING"""
        # Create goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        # Create single trajectory point with SHORT duration for responsiveness
        point = JointTrajectoryPoint()
        point.positions = self.target_positions.copy()
        point.velocities = [0.0] * 6
        point.accelerations = [0.0] * 6
        
        # FAST execution time for real-time control
        point.time_from_start = Duration(sec=0, nanosec=int(self.command_duration * 1e9))
        
        goal_msg.trajectory.points = [point]
        
        # Send goal asynchronously - don't wait for result
        # This allows rapid fire commands
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal acceptance/rejection"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('⚠️ Goal rejected')
            return
        
        self._goal_handle = goal_handle

    def feedback_callback(self, feedback_msg):
        """Handle action feedback (optional)"""
        pass  # We don't need to do anything with feedback for now

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
