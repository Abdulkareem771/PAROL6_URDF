#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import threading
import time
import math
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Load the joystick HTML interface
with open('/workspace/joystick_interface.html', 'r') as f:
    JOYSTICK_HTML = f.read()

class RobotBridge(Node):
    def __init__(self):
        super().__init__('mobile_robot_bridge')
        
        # Action client for joint control
        self._action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/parol6_arm_controller/follow_joint_trajectory'
        )
        
        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Current state
        self.current_joint_positions = [0.0] * 6
        self.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']
        
        # For joystick control
        self.current_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x, y, z, roll, pitch, yaw
        self.home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.get_logger().info('Robot Bridge with Joystick Control initialized')

    def joint_state_callback(self, msg):
        try:
            for i, name in enumerate(self.joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    self.current_joint_positions[i] = msg.position[idx]
        except Exception as e:
            self.get_logger().error(f'Joint state error: {e}')

    def send_joint_positions(self, positions, duration=2.0):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available!')
            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = JointTrajectory()
        goal_msg.trajectory.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        goal_msg.trajectory.points.append(point)
        
        self.get_logger().info(f'Sending goal: {positions}')
        self._action_client.send_goal_async(goal_msg)
        return True

    def joystick_to_joint_positions(self, joystick_x, joystick_y, joystick_z=0):
        """
        Convert joystick coordinates to joint positions using simple inverse kinematics
        This is a simplified version - you might want to use MoveIt's IK service for complex motions
        """
        # Get current joint positions as base
        current_positions = self.current_joint_positions.copy()
        
        # Simple mapping: joystick controls first 3 joints for demonstration
        # X controls joint1 (base rotation)
        # Y controls joint2 (shoulder)
        # Z could control joint3 (elbow) but we'll use it for demonstration
        
        # Scale factors for sensitivity
        scale_x = 0.5  # Base rotation sensitivity
        scale_y = 0.3  # Shoulder sensitivity
        scale_z = 0.3  # Elbow sensitivity
        
        # Calculate new positions (simple additive control)
        new_positions = [
            current_positions[0] + joystick_x * scale_x,  # joint1 - base rotation
            current_positions[1] + joystick_y * scale_y,  # joint2 - shoulder
            current_positions[2] + joystick_z * scale_z,  # joint3 - elbow
            current_positions[3],  # Keep other joints at current position
            current_positions[4],
            current_positions[5]
        ]
        
        # Apply joint limits (simplified)
        joint_limits = [
            (-3.14, 3.14),  # joint1
            (-2.0, 2.0),    # joint2  
            (-2.0, 2.0),    # joint3
            (-3.14, 3.14),  # joint4
            (-3.14, 3.14),  # joint5
            (-3.14, 3.14)   # joint6
        ]
        
        for i in range(6):
            new_positions[i] = max(joint_limits[i][0], min(joint_limits[i][1], new_positions[i]))
        
        return new_positions

    def get_joint_positions(self):
        return self.current_joint_positions

# Flask application
app = Flask(__name__)
CORS(app)

# Global robot bridge instance
robot_bridge = None

@app.route('/')
def index():
    return JOYSTICK_HTML

@app.route('/api/status')
def get_status():
    if robot_bridge:
        positions = robot_bridge.get_joint_positions()
        return jsonify({
            'joints': positions,
            'joint_names': robot_bridge.joint_names,
            'connected': True
        })
    return jsonify({'connected': False, 'joints': [], 'joint_names': []})

@app.route('/api/move', methods=['POST'])
def move_robot():
    if not robot_bridge:
        return jsonify({'success': False, 'error': 'Robot bridge not initialized'})
    
    try:
        positions = [
            float(request.json.get('joint1', 0)),
            float(request.json.get('joint2', 0)), 
            float(request.json.get('joint3', 0)),
            float(request.json.get('joint4', 0)),
            float(request.json.get('joint5', 0)),
            float(request.json.get('joint6', 0))
        ]
        duration = float(request.json.get('duration', 2.0))
        
        success = robot_bridge.send_joint_positions(positions, duration)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/joystick', methods=['POST'])
def joystick_control():
    if not robot_bridge:
        return jsonify({'success': False, 'error': 'Robot bridge not initialized'})
    
    try:
        joystick_x = float(request.json.get('x', 0))
        joystick_y = float(request.json.get('y', 0))
        joystick_z = float(request.json.get('z', 0))
        
        # Convert joystick coordinates to joint positions
        target_positions = robot_bridge.joystick_to_joint_positions(joystick_x, joystick_y, joystick_z)
        
        # Send to robot with shorter duration for responsive control
        success = robot_bridge.send_joint_positions(target_positions, duration=0.5)
        
        return jsonify({
            'success': success,
            'target_positions': target_positions,
            'joystick_input': [joystick_x, joystick_y, joystick_z]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/home', methods=['POST'])
def home_robot():
    if not robot_bridge:
        return jsonify({'success': False, 'error': 'Robot bridge not initialized'})
    
    try:
        success = robot_bridge.send_joint_positions([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop', methods=['POST'])
def emergency_stop():
    # Send current positions (stop movement)
    if robot_bridge:
        current = robot_bridge.get_joint_positions()
        robot_bridge.send_joint_positions(current, 0.1)
    return jsonify({'success': True, 'message': 'Emergency stop activated'})

def main():
    global robot_bridge
    
    # Initialize ROS 2
    rclpy.init()
    robot_bridge = RobotBridge()
    
    # Start ROS spinning in a separate thread
    spin_thread = threading.Thread(target=lambda: rclpy.spin(robot_bridge))
    spin_thread.daemon = True
    spin_thread.start()
    
    print("🚀 Mobile Control Bridge with Joystick starting on http://0.0.0.0:5000")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
