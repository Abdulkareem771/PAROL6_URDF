import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from sensor_msgs.msg import Joy, JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import time

class XboxIndustrialController(Node):
    def __init__(self):
        super().__init__('xbox_industrial_controller',
                        parameter_overrides=[
                            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
                        ])
        
        # ---------------------------------------------------------
        # 1. PARAMETERS (Industrial / Customizable)
        # ---------------------------------------------------------
        self.declare_parameter('sensitivity', 0.05)
        self.declare_parameter('deadzone', 0.15)
        self.declare_parameter('max_speed', 0.5)
        self.declare_parameter('control_rate', 10.0)  # Hz
        self.declare_parameter('trajectory_duration', 0.2) # Seconds
        
        # Load initial values
        self.sensitivity = self.get_parameter('sensitivity').value
        self.deadzone = self.get_parameter('deadzone').value
        self.max_speed = self.get_parameter('max_speed').value
        self.control_rate = self.get_parameter('control_rate').value
        self.traj_duration = self.get_parameter('trajectory_duration').value

        # ---------------------------------------------------------
        # 2. INTERFACES
        # ---------------------------------------------------------
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/parol6_arm_controller/follow_joint_trajectory'
        )
        
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.state_sub = self.create_subscription(JointState, '/joint_states', self.state_callback, 10)
        
        # ---------------------------------------------------------
        # 3. STATE MANAGEMENT
        # ---------------------------------------------------------
        self.joint_names = [
            'joint_L1', 'joint_L2', 'joint_L3', 
            'joint_L4', 'joint_L5', 'joint_L6'
        ]
        
        # Strict limits from URDF
        self.joint_limits = {
            'joint_L1': (-3.05, 3.05),
            'joint_L2': (-1.91, 1.91),
            'joint_L3': (-2.53, 2.53),
            'joint_L4': (-2.70, 2.70),
            'joint_L5': (-6.28, 6.28),
            'joint_L6': (0.0, 0.0)
        }
        
        self.current_joints = [0.0] * 6
        self.target_joints = [0.0] * 6
        self.initial_sync_done = False
        
        # Joystick state
        self.joy_cmd = [0.0] * 6
        self.buttons = [0] * 12
        self.last_buttons = [0] * 12
        
        # Control loop timer
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        
        # Safety / Status
        self.last_valid_joy_time = 0
        self.active_goal_handle = None
        
        self.get_logger().info('🏭 Industrial Xbox Controller Started')
        self.get_logger().info(f'   Rate: {self.control_rate}Hz, Sensitivity: {self.sensitivity}')

    def state_callback(self, msg):
        """Read robot state. ONLY sync target if we haven't started yet."""
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_joints[i] = msg.position[idx]
        
        if not self.initial_sync_done:
            self.target_joints = list(self.current_joints)
            self.initial_sync_done = True
            self.get_logger().info('✅ Initial state synced')

    def joy_callback(self, msg):
        """Read joystick input and map to velocity commands"""
        self.last_valid_joy_time = time.time()
        
        # Axes mapping
        # 0: Left X (Base), 1: Left Y (Shoulder)
        # 3: Right X (Elbow), 4: Right Y (Wrist Pitch)
        # 2: LT, 5: RT (Wrist Roll)
        
        raw_cmds = [
            msg.axes[0],                # L1 Base
            -msg.axes[1],               # L2 Shoulder (Inverted)
            msg.axes[3],                # L3 Elbow
            -msg.axes[4],               # L4 Wrist Pitch (Inverted)
            (msg.axes[5] - msg.axes[2]) / 2.0, # L5 Wrist Roll (Triggers)
            0.0                         # L6 Gripper
        ]
        
        # Apply deadzone and sensitivity
        for i in range(6):
            if abs(raw_cmds[i]) < self.deadzone:
                self.joy_cmd[i] = 0.0
            else:
                # Normalize (0 to 1) after deadzone
                val = raw_cmds[i]
                sign = 1 if val > 0 else -1
                normalized = (abs(val) - self.deadzone) / (1.0 - self.deadzone)
                self.joy_cmd[i] = sign * normalized * self.sensitivity

        self.buttons = msg.buttons

    def control_loop(self):
        """Main control loop running at fixed rate"""
        if not self.initial_sync_done:
            return

        # 1. Update Target Positions
        movement_detected = False
        for i in range(6):
            if self.joy_cmd[i] != 0.0:
                self.target_joints[i] += self.joy_cmd[i]
                movement_detected = True
            
            # ALWAYS clamp to limits
            min_l, max_l = self.joint_limits[self.joint_names[i]]
            self.target_joints[i] = max(min_l, min(max_l, self.target_joints[i]))

        # 2. Handle Buttons (One-shot)
        self.handle_buttons()

        # 3. Send Command if needed
        # We send commands continuously if moving, or once to hold position
        # To prevent "falling", we must ensure the controller always has a valid goal
        
        if movement_detected:
            self.send_trajectory()
        
        self.last_buttons = list(self.buttons)

    def handle_buttons(self):
        # A Button (0): Reset to Zero
        if self.buttons[0] and not self.last_buttons[0]:
            self.get_logger().info('🔄 RESET to Zero')
            self.target_joints = [0.0] * 6
            self.send_trajectory(duration=2.0) # Slower move for safety
            
        # B Button (1): Home
        if self.buttons[1] and not self.last_buttons[1]:
            self.get_logger().info('🏠 HOME Position')
            self.target_joints = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
            self.send_trajectory(duration=2.0)

    def send_trajectory(self, duration=None):
        if not self._action_client.server_is_ready():
            return

        if duration is None:
            duration = self.traj_duration

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = list(self.target_joints)
        point.velocities = [0.0] * 6 # Stop at the target point
        
        # CRITICAL: Time from start must be enough to reach target without violating velocity limits
        # But for teleop, we want it fast. The controller will interpolate.
        point.time_from_start = Duration(sec=0, nanosec=int(duration * 1e9))
        
        goal.trajectory.points = [point]
        
        # Use simple send_goal_async
        # We don't wait for result to avoid blocking the loop
        self._action_client.send_goal_async(goal)

def main():
    rclpy.init()
    node = XboxIndustrialController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
