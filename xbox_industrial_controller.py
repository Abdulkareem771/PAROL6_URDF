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
        # 1. PARAMETERS
        # ---------------------------------------------------------
        self.declare_parameter('sensitivity', 0.05)
        self.declare_parameter('deadzone', 0.15)
        self.declare_parameter('max_speed', 0.8) # Rad/s
        self.declare_parameter('control_rate', 20.0)  # Hz (Smoother)
        self.declare_parameter('trajectory_lookahead', 0.1) # Seconds
        
        # Load initial values
        self.sensitivity = self.get_parameter('sensitivity').value
        self.deadzone = self.get_parameter('deadzone').value
        self.max_speed = self.get_parameter('max_speed').value
        self.control_rate = self.get_parameter('control_rate').value
        self.traj_lookahead = self.get_parameter('trajectory_lookahead').value

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
        
        self.joint_limits = {
            'joint_L1': (-3.05, 3.05),
            'joint_L2': (-1.91, 1.91),
            'joint_L3': (-2.53, 2.53),
            'joint_L4': (-2.70, 2.70),
            'joint_L5': (-6.28, 6.28),
            'joint_L6': (0.0, 0.0)
        }
        
        self.current_joints = [0.0] * 6
        self.commanded_joints = [0.0] * 6 # The setpoint we are moving towards
        self.trajectory_joints = [0.0] * 6 # The actual point we are sending (ramped)
        self.initial_sync_done = False
        
        # Joystick state
        self.joy_cmd = [0.0] * 6
        self.buttons = [0] * 12
        self.last_buttons = [0] * 12
        
        # Control loop timer
        self.dt = 1.0 / self.control_rate
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info('🏭 Industrial Xbox Controller Started')

    def state_callback(self, msg):
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_joints[i] = msg.position[idx]
        
        if not self.initial_sync_done:
            self.commanded_joints = list(self.current_joints)
            self.trajectory_joints = list(self.current_joints)
            self.initial_sync_done = True
            self.get_logger().info('✅ Initial state synced')

    def joy_callback(self, msg):
        # Map axes to velocity commands
        raw_cmds = [
            msg.axes[0],                # L1 Base
            -msg.axes[1],               # L2 Shoulder
            msg.axes[3],                # L3 Elbow
            -msg.axes[4],               # L4 Wrist Pitch
            (msg.axes[5] - msg.axes[2]) / 2.0, # L5 Wrist Roll
            0.0
        ]
        
        for i in range(6):
            if abs(raw_cmds[i]) < self.deadzone:
                self.joy_cmd[i] = 0.0
            else:
                val = raw_cmds[i]
                sign = 1 if val > 0 else -1
                normalized = (abs(val) - self.deadzone) / (1.0 - self.deadzone)
                self.joy_cmd[i] = sign * normalized * self.sensitivity

        self.buttons = msg.buttons

    def control_loop(self):
        if not self.initial_sync_done:
            return

        # 1. Update Commanded Goal (User Input)
        # We integrate velocity into the "Commanded" setpoint
        for i in range(6):
            if self.joy_cmd[i] != 0.0:
                self.commanded_joints[i] += self.joy_cmd[i]
                
            # Clamp Commanded to limits
            min_l, max_l = self.joint_limits[self.joint_names[i]]
            self.commanded_joints[i] = max(min_l, min(max_l, self.commanded_joints[i]))

        # 2. Handle Buttons (Set Commanded Goal instantly)
        self.handle_buttons()

        # 3. Slew Rate Limiter (Trajectory Generation)
        # Move trajectory_joints towards commanded_joints at max_speed
        moving = False
        max_step = self.max_speed * self.dt
        
        for i in range(6):
            diff = self.commanded_joints[i] - self.trajectory_joints[i]
            if abs(diff) > 0.0001:
                moving = True
                if abs(diff) > max_step:
                    step = max_step if diff > 0 else -max_step
                    self.trajectory_joints[i] += step
                else:
                    self.trajectory_joints[i] = self.commanded_joints[i]

        # 4. Send Trajectory Point
        # We send it even if not moving, to hold position firmly
        self.send_trajectory()
        
        self.last_buttons = list(self.buttons)

    def handle_buttons(self):
        # A Button (0): Reset to Zero
        if self.buttons[0] and not self.last_buttons[0]:
            self.get_logger().info('🔄 Commanded: RESET to Zero')
            self.commanded_joints = [0.0] * 6
            
        # B Button (1): Home
        if self.buttons[1] and not self.last_buttons[1]:
            self.get_logger().info('🏠 Commanded: HOME Position')
            self.commanded_joints = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]

    def send_trajectory(self):
        if not self._action_client.server_is_ready():
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = list(self.trajectory_joints)
        point.velocities = [0.0] * 6
        
        # Lookahead time tells the controller "be here in X seconds"
        # Since we are generating the ramp ourselves, we can keep this small and constant
        point.time_from_start = Duration(sec=0, nanosec=int(self.traj_lookahead * 1e9))
        
        goal.trajectory.points = [point]
        
        # Send async and forget
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
