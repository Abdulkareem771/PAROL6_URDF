#!/usr/bin/env python3
"""
Mobile Control Bridge for PAROL6 Robot
Provides REST API for controlling the robot via HTTP
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for web access

class RobotBridge(Node):
    def __init__(self):
        super().__init__('mobile_robot_bridge')
        
        # Action client for trajectory control
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/parol6_arm_controller/follow_joint_trajectory'
        )
        
        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.current_joint_state = None
        self.get_logger().info('Robot Bridge initialized')
    
    def joint_state_callback(self, msg):
        """Update current joint state"""
        self.current_joint_state = msg
    
    def get_joint_positions(self):
        """Get current joint positions"""
        if self.current_joint_state:
            return {
                'joints': list(self.current_joint_state.name),
                'positions': list(self.current_joint_state.position),
                'velocities': list(self.current_joint_state.velocity) if self.current_joint_state.velocity else [],
            }
        return None
    
    def send_joint_positions(self, positions, duration=2.0):
        """Send joint trajectory goal"""
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return False
        
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        
        goal_msg.trajectory.points = [point]
        
        self.get_logger().info(f'Sending goal: {positions}')
        future = self._action_client.send_goal_async(goal_msg)
        return True

# Global robot bridge instance
robot_bridge = None

# Flask API Routes
@app.route('/api/status', methods=['GET'])
def get_status():
    """Get robot status"""
    if robot_bridge:
        joint_state = robot_bridge.get_joint_positions()
        return jsonify({
            'status': 'online',
            'joint_state': joint_state
        })
    return jsonify({'status': 'offline'}), 503

@app.route('/api/move', methods=['POST'])
def move_robot():
    """Move robot to target position"""
    data = request.json
    
    if 'positions' not in data:
        return jsonify({'error': 'Missing positions'}), 400
    
    positions = data['positions']
    duration = data.get('duration', 2.0)
    
    if len(positions) != 6:
        return jsonify({'error': 'Need exactly 6 joint positions'}), 400
    
    if robot_bridge:
        success = robot_bridge.send_joint_positions(positions, duration)
        if success:
            return jsonify({'status': 'moving', 'target': positions})
        else:
            return jsonify({'error': 'Failed to send command'}), 500
    
    return jsonify({'error': 'Robot bridge not initialized'}), 503

@app.route('/api/home', methods=['POST'])
def move_home():
    """Move robot to home position"""
    home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    if robot_bridge:
        success = robot_bridge.send_joint_positions(home_position, 3.0)
        if success:
            return jsonify({'status': 'moving_home'})
        else:
            return jsonify({'error': 'Failed to send command'}), 500
    
    return jsonify({'error': 'Robot bridge not initialized'}), 503

@app.route('/')
def index():
    """Serve main page"""
    return '''
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PAROL6 Robot Control</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .control-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .joystick-area { width: 200px; height: 200px; background: #e0e0e0; border-radius: 50%; margin: 10px auto; position: relative; }
        .joystick-handle { width: 60px; height: 60px; background: #007bff; border-radius: 50%; position: absolute; top: 70px; left: 70px; }
        button { padding: 10px 15px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .stop-btn { background: #dc3545; }
        .status { background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 PAROL6 Robot Control</h1>
        
        <div class="control-section">
            <h3>🎮 Joystick Control</h3>
            <p>Drag the blue circle to control the robot:</p>
            <div class="joystick-area" id="joystick">
                <div class="joystick-handle" id="handle"></div>
            </div>
            <p>X: <span id="coordX">0</span> | Y: <span id="coordY">0</span></p>
        </div>
        
        <div class="control-section">
            <h3>🔧 Joint Control</h3>
            <div id="sliders">
            </div>
        </div>
        
        <div>
            <button onclick="sendHome()">🏠 Home Position</button>
            <button class="stop-btn" onclick="emergencyStop()">🛑 Emergency Stop</button>
        </div>
        
        <div class="status">
            <h3>Status</h3>
            <div id="robotStatus">Loading...</div>
        </div>
    </div>

    <script>
        // Joystick functionality
        const joystick = document.getElementById('joystick');
        const handle = document.getElementById('handle');
        const coordX = document.getElementById('coordX');
        const coordY = document.getElementById('coordY');
        
        let isDragging = false;
        
        // Mouse events
        handle.addEventListener('mousedown', () => isDragging = true);
        document.addEventListener('mousemove', handleMove);
        document.addEventListener('mouseup', stopDrag);
        
        // Touch events
        handle.addEventListener('touchstart', (e) => {
            e.preventDefault();
            isDragging = true;
        });
        document.addEventListener('touchmove', handleTouchMove, { passive: false });
        document.addEventListener('touchend', stopDrag);
        
        function handleMove(e) {
            updatePosition(e.clientX, e.clientY);
        }
        
        function handleTouchMove(e) {
            e.preventDefault();
            updatePosition(e.touches[0].clientX, e.touches[0].clientY);
        }
        
        function updatePosition(clientX, clientY) {
            const rect = joystick.getBoundingClientRect();
            const centerX = rect.left + 100;
            const centerY = rect.top + 100;
            
            let deltaX = clientX - centerX;
            let deltaY = clientY - centerY;
            
            // Limit movement to joystick bounds
            const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
            if (distance > 70) {
                deltaX = (deltaX / distance) * 70;
                deltaY = (deltaY / distance) * 70;
            }
            
            // Update handle position
            handle.style.left = (70 + deltaX) + 'px';
            handle.style.top = (70 + deltaY) + 'px';
            
            // Normalize coordinates (-1 to 1)
            const x = deltaX / 70;
            const y = -deltaY / 70; // Invert Y axis
            
            coordX.textContent = x.toFixed(2);
            coordY.textContent = y.toFixed(2);
            
            // Send command to robot
            sendJoystickCommand(x, y);
        }
        
        function stopDrag() {
            isDragging = false;
            
            // Return to center
            handle.style.left = '70px';
            handle.style.top = '70px';
            coordX.textContent = '0';
            coordY.textContent = '0';
            
            // Stop movement
            sendJoystickCommand(0, 0);
        }
        
        function sendJoystickCommand(x, y) {
            fetch('/api/joystick', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: x, y: y })
            }).then(r => r.json()).then(data => {
                console.log('Joystick response:', data);
            });
        }
        
        // Keep existing functions
        function sendHome() {
            fetch('/api/home', { method: 'POST' }).then(r => r.json()).then(console.log);
        }
        
        function emergencyStop() {
            fetch('/api/stop', { method: 'POST' }).then(r => r.json()).then(console.log);
        }
        
        // Load existing sliders and status functionality
        // This will be filled by the existing JavaScript
    </script>
</body>
</html>
'''

def spin_ros(node):
    """Spin ROS in separate thread"""
    rclpy.spin(node)

def main():
    global robot_bridge
    
    rclpy.init()
    robot_bridge = RobotBridge()
    
    # Start ROS spinning in separate thread
    ros_thread = threading.Thread(target=spin_ros, args=(robot_bridge,), daemon=True)
    ros_thread.start()
    
    # Start Flask server
    print('🚀 Mobile Control Bridge starting on http://0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
