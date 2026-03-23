#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
import serial
import time
import math
import threading
import csv
from datetime import datetime
import os

class RealRobotDriver(Node):
    def __init__(self):
        super().__init__('real_robot_driver')
        
        # Declare logging parameter
        self.declare_parameter('enable_logging', True)
        self.declare_parameter('log_dir', '/workspace/logs')
        
        self.enable_logging = self.get_parameter('enable_logging').value
        self.log_dir = self.get_parameter('log_dir').value
        
        # Callback Groups to isolate action server execution from timers
        self.action_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        
        # Setup logging
        if self.enable_logging:
            os.makedirs(self.log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(self.log_dir, f'driver_commands_{timestamp}.csv')
            self.log_file = open(log_file, 'w', newline='')
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow([
                'seq', 'timestamp_pc_us', 'timestamp_pc_iso',
                'j1_pos', 'j2_pos', 'j3_pos', 'j4_pos', 'j5_pos', 'j6_pos',
                'j1_vel', 'j2_vel', 'j3_vel', 'j4_vel', 'j5_vel', 'j6_vel',
                'j1_acc', 'j2_acc', 'j3_acc', 'j4_acc', 'j5_acc', 'j6_acc',
                'command_sent'
            ])
            self.seq_counter = 0
            self.get_logger().info(f'Logging enabled: {log_file}')
        else:
            self.get_logger().info('Logging disabled')
        
        self.current_joints = [0.0] * 6
        self.latest_commanded_joints = [0.0] * 6  # Thread-safe storage for active trajectory
        self.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']
        self.data_lock = threading.Lock()
        
        # 1. Serial Connection - Auto-detect
        self.ser = None
        self.running = True
        ports_to_try = ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/pts/8']
        
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
        else:
            # Start background serial read thread
            self.read_thread = threading.Thread(target=self.serial_read_loop, daemon=True)
            self.read_thread.start()

        # 3. Action Server (Isolated Callback Group)
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/parol6_arm_controller/follow_joint_trajectory',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.action_cb_group)

        # 4. Joint State Publisher (Feedback)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.05, self.publish_actual_feedback, callback_group=self.timer_cb_group) # 20Hz

    def serial_read_loop(self):
        """Continuously reads incoming feedback from the microcontroller without blocking ROS."""
        while self.running and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('<ACK') and line.endswith('>'):
                        # Parse <ACK,seq,p0,v0,p1,v1,...>
                        content = line[1:-1]
                        parts = content.split(',')
                        if len(parts) >= 14:  # ACK + seq + (6 * 2 values)
                            with self.data_lock:
                                # Update current_joints directly from Teensy feedback
                                self.current_joints = [
                                    float(parts[2]), float(parts[4]), float(parts[6]),
                                    float(parts[8]), float(parts[10]), float(parts[12])
                                ]
            except Exception as e:
                self.get_logger().error(f"Serial Read Error: {e}")
            time.sleep(0.005)  # Yield thread (5ms) to prevent maxing CPU

    def wait_for_homing(self):
        # We handle homing manually or ignore for now
        pass

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
        
        # Non-blocking Execution Loop via timing sync
        start_time = time.time()
        
        for point in points:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal Canceled')
                return FollowJointTrajectory.Result()

            # Wait exactly 50ms per trajectory point (20Hz) cleanly without freezing the whole node
            # (MultiThreadedExecutor ensures this only blocks the Action thread)
            point_time_from_start = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            
            # 1. Update Commanded State
            with self.data_lock:
                self.latest_commanded_joints = list(point.positions)
            
            # 2. Format & Send (with sequence number for Teensy)
            if self.enable_logging:
                seq = self.seq_counter
                cmd_str = f"<{seq},{','.join([f'{p:.4f}' for p in self.latest_commanded_joints])}>\n"
            else:
                cmd_str = f"<0,{','.join([f'{p:.4f}' for p in self.latest_commanded_joints])}>\n"
            
            if self.ser:
                try:
                    self.ser.write(cmd_str.encode())
                except Exception as e:
                    self.get_logger().error(f"Failed to write to serial: {e}")
            
            # 3. Log command
            if self.enable_logging:
                timestamp_us = int(time.time() * 1_000_000)
                timestamp_iso = datetime.now().isoformat()
                
                # Get velocities and accelerations (may be empty)
                velocities = list(point.velocities) if len(point.velocities) == 6 else [0.0] * 6
                accelerations = list(point.accelerations) if len(point.accelerations) == 6 else [0.0] * 6
                
                self.log_writer.writerow([
                    seq,
                    timestamp_us,
                    timestamp_iso,
                    *self.latest_commanded_joints,
                    *velocities,
                    *accelerations,
                    cmd_str.strip()
                ])
                self.log_file.flush()
                self.seq_counter += 1

            # Sleep precisely enough to match the trajectory timing
            elapsed = time.time() - start_time
            sleep_time = point_time_from_start - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        return result

    def publish_actual_feedback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        
        with self.data_lock:
            # We now publish the ACTUAL encoder positions safely read by the background thread.
            msg.position = list(self.current_joints)
            
        self.joint_pub.publish(msg)
    
    def __del__(self):
        """Cleanup on shutdown"""
        self.running = False
        if hasattr(self, 'read_thread') and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)
            
        if hasattr(self, 'ser') and self.ser:
            self.ser.close()
            
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            self.get_logger().info('Log file closed')

def main(args=None):
    rclpy.init(args=args)
    node = RealRobotDriver()
    
    # Use MultiThreadedExecutor to allow Actions, Timers, and Subscriptions to run in parallel
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
