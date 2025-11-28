
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import threading
import time
import math
from flask import Flask, request, jsonify
from flask_cors import CORS

# Embedded HTML with joystick interface
JOYSTICK_HTML = '''
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PAROL6 Robot Control</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #f0f0f0;
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .joystick-area {
            width: 300px;
            height: 300px;
            background: #e0e0e0;
            border-radius: 50%;
            margin: 20px auto;
            position: relative;
            border: 2px solid #333;
            touch-action: none;
        }
        .joystick-handle {
            width: 80px;
            height: 80px;
            background: #007bff;
            border-radius: 50%;
            position: absolute;
            top: 110px;
            left: 110px;
            cursor: pointer;
        }
        .status {
            margin: 20px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        button {
            padding: 10px 20px;
            margin: 5px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .coordinates {
            font-family: monospace;
            background: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 PAROL6 Robot Joystick Control</h1>
        
        <div class="joystick-area" id="joystick">
            <div class="joystick-handle" id="handle"></div>
        </div>
        
        <div class="coordinates">
            X: <span id="x">0.00</span> | 
            Y: <span id="y">0.00</span>
        </div>
        
        <div>
            <button onclick="sendHome()">Home Position</button>
            <button onclick="emergencyStop()" style="background: #dc3545;">Emergency Stop</button>
        </div>
        
        <div class="status">
            <h3>Robot Status</h3>
            <div id="jointStatus">Loading...</div>
        </div>
    </div>

    <script>
        const joystick = document.getElementById('joystick');
        const handle = document.getElementById('handle');
        const xDisplay = document.getElementById('x');
        const yDisplay = document.getElementById('y');
        
        let isDragging = false;
        const maxMove = 100;
        
        // Mouse events
        handle.addEventListener('mousedown', startDrag);
        document.addEventListener('mousemove', drag);
        document.addEventListener('mouseup', stopDrag);
        
        // Touch events  
        handle.addEventListener('touchstart', (e) => {
            e.preventDefault();
            startDrag(e.touches[0]);
        });
        document.addEventListener('touchmove', (e) => {
            e.preventDefault();
            drag(e.touches[0]);
        });
        document.addEventListener('touchend', stopDrag);
        
        function startDrag(e) {
            isDragging = true;
            handle.style.background = '#0056b3';
        }
        
        function drag(e) {
            
            const rect = joystick.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            
            let deltaX = e.clientX - centerX;
            let deltaY = e.clientY - centerY;
            
            // Limit movement
            const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
            if (distance > maxMove) {
                deltaX = (deltaX / distance) * maxMove;
                deltaY = (deltaY / distance) * maxMove;
            }
            
            // Update handle position
            handle.style.left = (110 + deltaX) + 'px';
            handle.style.top = (110 + deltaY) + 'px';
            
            // Normalize coordinates
            const x = deltaX / maxMove;
            const y = -deltaY / maxMove; // Invert Y
            
            xDisplay.textContent = x.toFixed(2);
            yDisplay.textContent = y.toFixed(2);
            
            // Send command
            sendJoystickCommand(x, y);
        }
        
        function stopDrag() {
            isDragging = false;
            handle.style.background = '#007bff';
            
            // Return to center
            handle.style.left = '110px';
            handle.style.top = '110px';
            xDisplay.textContent = '0.00';
            yDisplay.textContent = '0.00';
            
            sendJoystickCommand(0, 0);
        }
        
        function sendJoystickCommand(x, y) {
            fetch('/api/joystick', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: x, y: y })
            }).then(r => r.json()).then(console.log);
        }
        
        function sendHome() {
            fetch('/api/home', { method: 'POST' }).then(r => r.json()).then(console.log);
        }
        
        function emergencyStop() {
            fetch('/api/stop', { method: 'POST' }).then(r => r.json()).then(console.log);
        }
        
        function updateStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    if (data.connected) {
                        let html = '';
                        data.joints.forEach((pos, i) => {
                            html += ;
                        });
                        document.getElementById('jointStatus').innerHTML = html;
                    }
                });
        }
        
        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>
'''

class RobotBridge(Node):
    def __init__(self):
        super().__init__('mobile_robot_bridge')
        
        self._action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/parol6_arm_controller/follow_joint_trajectory'
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.current_joint_positions = [0.0] * 6
        self.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']
        
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
        goal_msg.trajectory.points.append(point)
        
        self.get_logger().info(f'Sending goal: {positions}')
        self._action_client.send_goal_async(goal_msg)
        return True

    def joystick_to_joint_positions(self, joystick_x, joystick_y):
        current = self.current_joint_positions.copy()
        
        # Simple control: joystick X controls base rotation, Y controls shoulder
        scale = 0.5
        new_positions = [
            current[0] + joystick_x * scale,  # joint1 - base
            current[1] + joystick_y * scale,  # joint2 - shoulder  
            current[2],  # joint3 - keep current
            current[3], current[4], current[5]  # keep other joints
        ]
        
        # Simple joint limits
        limits = [(-3.14, 3.14), (-2.0, 2.0), (-2.0, 2.0), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)]
        for i in range(6):
            new_positions[i] = max(limits[i][0], min(limits[i][1], new_positions[i]))
        
        return new_positions

    def get_joint_positions(self):
        return self.current_joint_positions

# Flask app
app = Flask(__name__)
CORS(app)

robot_bridge = None

@app.route('/')
def index():
    return JOYSTICK_HTML

@app.route('/api/status')
def get_status():
    if robot_bridge:
        return jsonify({
            'connected': True,
            'joints': robot_bridge.get_joint_positions(),
            'joint_names': robot_bridge.joint_names
        })
    return jsonify({'connected': False})

@app.route('/api/joystick', methods=['POST'])
def joystick_control():
    if not robot_bridge:
        return jsonify({'success': False, 'error': 'Robot not connected'})
    
    try:
        x = float(request.json.get('x', 0))
        y = float(request.json.get('y', 0))
        print(f"Joystick: x={x:.2f}, y={y:.2f}")
        
        positions = robot_bridge.joystick_to_joint_positions(x, y)
        success = robot_bridge.send_joint_positions(positions, 0.3)
        
        return jsonify({'success': success, 'positions': positions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/home', methods=['POST'])
def home_robot():
    if robot_bridge:
        success = robot_bridge.send_joint_positions([0.0]*6, 3.0)
        return jsonify({'success': success})
    return jsonify({'success': False})

@app.route('/api/stop', methods=['POST'])
def emergency_stop():
    if robot_bridge:
        current = robot_bridge.get_joint_positions()
        robot_bridge.send_joint_positions(current, 0.1)
    return jsonify({'success': True})

def main():
    global robot_bridge
    
    rclpy.init()
    robot_bridge = RobotBridge()
    
    spin_thread = threading.Thread(target=lambda: rclpy.spin(robot_bridge))
    spin_thread.daemon = True
    spin_thread.start()
    
    print("🚀 Joystick Control Bridge starting on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
