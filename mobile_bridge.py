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
    <!DOCTYPE html>
    <html>
    <head>
        <title>PAROL6 Mobile Control</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0;
                padding: 20px; 
                background: #1a1a2e;
                color: #eee;
            }
            .container { 
                max-width: 600px; 
                margin: 0 auto;
                background: #16213e;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            h1 { 
                text-align: center; 
                color: #0f3460;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 30px;
            }
            .joint-control {
                margin: 15px 0;
                padding: 15px;
                background: #0f3460;
                border-radius: 8px;
            }
            .joint-control label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #667eea;
            }
            .slider-container {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            input[type="range"] {
                flex-grow: 1;
                height: 8px;
                border-radius: 5px;
                background: #1a1a2e;
                outline: none;
                -webkit-appearance: none;
            }
            input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #667eea;
                cursor: pointer;
            }
            .value-display {
                min-width: 60px;
                text-align: right;
                font-family: monospace;
                color: #eee;
            }
            button {
                width: 100%;
                padding: 15px;
                margin: 10px 0;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .btn-move {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .btn-home {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            button:active {
                transform: translateY(0px);
            }
            #status {
                padding: 15px;
                margin: 20px 0;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
            }
            .status-online { background: #2ecc71; color: white; }
            .status-offline { background: #e74c3c; color: white; }
            .status-moving { background: #f39c12; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 PAROL6 Control</h1>
            <div id="status">Checking status...</div>
            
            <div class="joint-control">
                <label>Joint 1 (Base)</label>
                <div class="slider-container">
                    <input type="range" id="j1" min="-1.7" max="1.7" step="0.01" value="0">
                    <span class="value-display" id="j1-val">0.00</span>
                </div>
            </div>
            
            <div class="joint-control">
                <label>Joint 2 (Shoulder)</label>
                <div class="slider-container">
                    <input type="range" id="j2" min="-0.98" max="1.0" step="0.01" value="0">
                    <span class="value-display" id="j2-val">0.00</span>
                </div>
            </div>
            
            <div class="joint-control">
                <label>Joint 3 (Elbow)</label>
                <div class="slider-container">
                    <input type="range" id="j3" min="-2.0" max="1.3" step="0.01" value="0">
                    <span class="value-display" id="j3-val">0.00</span>
                </div>
            </div>
            
            <div class="joint-control">
                <label>Joint 4 (Wrist Pitch)</label>
                <div class="slider-container">
                    <input type="range" id="j4" min="-2.0" max="2.0" step="0.01" value="0">
                    <span class="value-display" id="j4-val">0.00</span>
                </div>
            </div>
            
            <div class="joint-control">
                <label>Joint 5 (Wrist Roll)</label>
                <div class="slider-container">
                    <input type="range" id="j5" min="-2.1" max="2.1" step="0.01" value="0">
                    <span class="value-display" id="j5-val">0.00</span>
                </div>
            </div>
            
            <div class="joint-control">
                <label>Joint 6 (End Effector)</label>
                <div class="slider-container">
                    <input type="range" id="j6" min="-3.14" max="3.14" step="0.01" value="0">
                    <span class="value-display" id="j6-val">0.00</span>
                </div>
            </div>
            
            <button class="btn-move" onclick="moveRobot()">Move Robot</button>
            <button class="btn-home" onclick="moveHome()">Go to Home</button>
        </div>
        
        <script>
            // Update slider value displays
            for (let i = 1; i <= 6; i++) {
                const slider = document.getElementById(`j${i}`);
                const display = document.getElementById(`j${i}-val`);
                slider.oninput = () => display.textContent = parseFloat(slider.value).toFixed(2);
            }
            
            function updateStatus() {
                fetch('/api/status')
                    .then(r => r.json())
                    .then(data => {
                        const status = document.getElementById('status');
                        if (data.status === 'online') {
                            status.textContent = '✓ Robot Online';
                            status.className = 'status-online';
                            
                            // Update sliders with current positions
                            if (data.joint_state && data.joint_state.positions) {
                                for (let i = 0; i < 6; i++) {
                                    const slider = document.getElementById(`j${i+1}`);
                                    const display = document.getElementById(`j${i+1}-val`);
                                    slider.value = data.joint_state.positions[i];
                                    display.textContent = data.joint_state.positions[i].toFixed(2);
                                }
                            }
                        } else {
                            status.textContent = '✗ Robot Offline';
                            status.className = 'status-offline';
                        }
                    })
                    .catch(() => {
                        const status = document.getElementById('status');
                        status.textContent = '✗ Connection Error';
                        status.className = 'status-offline';
                    });
            }
            
            function moveRobot() {
                const positions = [];
                for (let i = 1; i <= 6; i++) {
                    positions.push(parseFloat(document.getElementById(`j${i}`).value));
                }
                
                const status = document.getElementById('status');
                status.textContent = '⟳ Moving...';
                status.className = 'status-moving';
                
                fetch('/api/move', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ positions, duration: 2.0 })
                })
                .then(r => r.json())
                .then(data => {
                    setTimeout(updateStatus, 2000);
                })
                .catch(err => {
                    status.textContent = '✗ Move Failed';
                    status.className = 'status-offline';
                });
            }
            
            function moveHome() {
                const status = document.getElementById('status');
                status.textContent = '⟳ Moving to Home...';
                status.className = 'status-moving';
                
                fetch('/api/home', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        setTimeout(updateStatus, 3000);
                    })
                    .catch(err => {
                        status.textContent = '✗ Move Failed';
                        status.className = 'status-offline';
                    });
            }
            
            // Update status every 2 seconds
            updateStatus();
            setInterval(updateStatus, 2000);
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
