#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Joy, JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

class XboxDirectControl(Node):
    def __init__(self):
        super().__init__('xbox_direct_control')
        
        # Action client for the arm controller
        self._action_client = ActionClient(self, FollowJointTrajectory, '/parol6_arm_controller/follow_joint_trajectory')
        
        # Subscribe to Xbox controller
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        
        # Subscribe to current joint states to know where we are
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        
        self.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']
        self.current_positions = [0.0] * 6
        self.target_positions = [0.0] * 6
        self.joints_received = False
        
        # Speed scaling
        self.speed_scale = 0.05
        
        self.get_logger().info('Xbox Direct Control READY. Waiting for joint states...')

    def joint_state_callback(self, msg):
        # Update current positions from robot feedback
        # We need to map the joint names correctly because msg.name might be in different order
        try:
            for i, name in enumerate(self.joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    # Only update if we haven't received joints yet, or if we are not actively moving
                    # This is a simplification; a real controller would integrate velocity
                    if not self.joints_received:
                        self.current_positions[i] = msg.position[idx]
                        self.target_positions[i] = msg.position[idx]
            
            if not self.joints_received:
                self.joints_received = True
                self.get_logger().info('Joint states received! Ready to move.')
                
        except ValueError:
            pass

    def joy_callback(self, msg):
        if not self.joints_received:
            return

        # Map Joystick to Joint Velocities (Position Integration)
        # Left Stick X (Axis 0) -> Base (L1)
        # Left Stick Y (Axis 1) -> Shoulder (L2)
        # Right Stick Y (Axis 4) -> Elbow (L3)
        # Right Stick X (Axis 3) -> Wrist Pitch (L4)
        # D-Pad Y (Axis 7) -> Wrist Yaw (L5)
        # Triggers (Axis 2, 5) -> Wrist Roll (L6)

        # Update targets based on joystick input
        self.target_positions[0] -= msg.axes[0] * self.speed_scale      # L1
        self.target_positions[1] += msg.axes[1] * self.speed_scale      # L2
        self.target_positions[2] += msg.axes[4] * self.speed_scale      # L3
        self.target_positions[3] -= msg.axes[3] * self.speed_scale      # L4
        self.target_positions[4] += msg.axes[7] * self.speed_scale      # L5
        
        # Triggers for L6
        trigger_val = (msg.axes[5] - msg.axes[2]) / 2.0
        self.target_positions[5] += trigger_val * self.speed_scale      # L6

        # Software Joint Limits (Simple clamping)
        # These are approximate limits to prevent self-collision
        self.target_positions[0] = max(-3.14, min(3.14, self.target_positions[0]))
        self.target_positions[1] = max(-1.57, min(1.57, self.target_positions[1])) # Shoulder limit
        self.target_positions[2] = max(-2.5, min(2.5, self.target_positions[2]))
        
        # Send command
        self.send_goal()

    def send_goal(self):
        goal_msg = FollowJointTrajectory.Goal()
        
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = self.target_positions
        point.time_from_start = Duration(sec=0, nanosec=200000000) # 200ms execution time
        
        traj.points.append(point)
        goal_msg.trajectory = traj
        
        # Send async to not block
        self._action_client.send_goal_async(goal_msg)

def main():
    rclpy.init()
    node = XboxDirectControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
