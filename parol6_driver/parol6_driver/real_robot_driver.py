#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
import serial
import time
import math
import threading

class RealRobotDriver(Node):
    def __init__(self):
        super().__init__('real_robot_driver')
        
        # 1. Serial Connection - Auto-detect
        self.ser = None
        ports_to_try = ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyUSB1']
        
        for port in ports_to_try:
            try:
                self.ser = serial.Serial(port, 115200, timeout=0.1)
                time.sleep(2) # Wait for Arduino reset
                self.get_logger().info(f"Connected to Microcontroller at {port}")
                break
            except Exception:
                pass
        
        if self.ser is None:
             self.get_logger().warn("Could not connect to any Serial Port! Mode: SIMULATION")

        # 2. Homing Wait
        # self.wait_for_homing()

        # 3. Action Server
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/parol6_arm_controller/follow_joint_trajectory',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

        # 4. Joint State Publisher (Feedback)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.05, self.publish_fake_feedback) # 20Hz
        
        self.current_joints = [0.0] * 6
        self.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']

    def wait_for_homing(self):
        if not self.ser: return
        self.get_logger().info("Waiting for Robot Homing...")
        while True:
            if self.ser.in_waiting:
                line = self.ser.readline().decode().strip()
                if "READY" in line:
                    self.get_logger().info("Robot Homing Complete!")
                    break
            time.sleep(0.1)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received Goal Request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received Cancel Request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing Trajectory...')
        
        # Get the trajectory
        traj = goal_handle.request.trajectory
        points = traj.points
        
        # Simple Execution Loop
        for point in points:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal Canceled')
                return FollowJointTrajectory.Result()

            # 1. Update Internal State
            self.current_joints = list(point.positions)
            
            # 2. Format & Send
            cmd_str = f"<{','.join([f'{p:.4f}' for p in self.current_joints])}>\n"
            
            if self.ser:
                self.ser.write(cmd_str.encode())
                
                if self.ser.in_waiting:
                    resp = self.ser.readline().decode()
                    if "ERROR" in resp:
                        self.get_logger().fatal("ROBOT STALL DETECTED!")
                        goal_handle.abort()
                        return FollowJointTrajectory.Result()

            # 3. Timing (Approximation)
            time.sleep(0.05) 

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        return result

    def publish_fake_feedback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_joints
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RealRobotDriver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
