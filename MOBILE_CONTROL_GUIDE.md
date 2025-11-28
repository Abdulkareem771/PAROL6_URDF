# PAROL6 Mobile Control System

## Overview

The mobile control system allows you to control the PAROL6 robot from any device (phone, tablet, computer) via a web interface.

## Architecture

```
[Web Browser] <--HTTP--> [Flask Server] <--ROS2--> [PAROL6 Robot]
```

- **Flask Server**: REST API bridge between web and ROS 2
- **Web Interface**: Modern, responsive UI with sliders for each joint
- **ROS 2 Integration**: Uses action clients for trajectory control

## Quick Start

### 1. Start the Simulation

```bash
# Terminal 1
./start_ignition.sh
```

Wait for Ignition Gazebo to load completely.

### 2. Start the Mobile Bridge

```bash
# Terminal 2
./start_mobile_bridge.sh
```

### 3. Open Web Interface

Open your browser and go to:
```
http://localhost:5000
```

Or from your phone (same WiFi network):
```
http://YOUR_COMPUTER_IP:5000
```

To find your IP:
```bash
hostname -I | awk '{print $1}'
```

## Features

### Current Features ✅
- **Real-time Joint Control**: Control all 6 joints with sliders
- **Visual Feedback**: See current joint positions
- **Home Position**: One-click return to home
- **Status Monitoring**: Real-time connection status
- **Mobile Responsive**: Works on phones and tablets

### API Endpoints

#### GET /api/status
Get robot status and current joint positions.

**Response:**
```json
{
  "status": "online",
  "joint_state": {
    "joints": ["L1", "L2", "L3", "L4", "L5", "L6"],
    "positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }
}
```

#### POST /api/move
Move robot to target position.

**Request:**
```json
{
  "positions": [0.5, -0.5, 0.5, 0.0, 0.0, 0.0],
  "duration": 2.0
}
```

**Response:**
```json
{
  "status": "moving",
  "target": [0.5, -0.5, 0.5, 0.0, 0.0, 0.0]
}
```

#### POST /api/home
Move robot to home position (all zeros).

**Response:**
```json
{
  "status": "moving_home"
}
```

## Future Enhancements (Planned)

### Phase 1: Advanced Control
- [ ] Cartesian space control (move end-effector in X/Y/Z)
- [ ] Saved positions (teach and replay)
- [ ] Speed control
- [ ] Emergency stop button

### Phase 2: Sensors & Vision
- [ ] Camera stream integration
- [ ] Depth sensor visualization
- [ ] Object detection overlay
- [ ] Point cloud viewer

### Phase 3: AI & Automation
- [ ] Voice control
- [ ] Gesture recognition
- [ ] Autonomous pick-and-place
- [ ] Path recording and playback

### Phase 4: Motor Control
- [ ] FOC (Field-Oriented Control) configuration
- [ ] Stepper motor tuning interface
- [ ] Real-time motor status monitoring
- [ ] Vibration analysis

### Phase 5: Multi-User & Collaboration
- [ ] WebRTC video streaming
- [ ] Multi-user access with permissions
- [ ] Collaborative control
- [ ] Remote diagnostics

## Development

### Adding New Features

The codebase is structured for easy extension:

**Backend** (`mobile_bridge.py`):
```python
@app.route('/api/your_feature', methods=['POST'])
def your_feature():
    # Your code here
    return jsonify({'status': 'success'})
```

**Frontend** (embedded in `mobile_bridge.py`):
Add HTML/CSS/JavaScript in the `index()` function.

### Testing

```bash
# Terminal 1: Start simulation
./start_ignition.sh

# Terminal 2: Start bridge
./start_mobile_bridge.sh

# Terminal 3: Test API
curl http://localhost:5000/api/status
```

### Custom Web Page

To create a custom web page:

1. Create HTML file: `mobile_control/web/custom.html`
2. Add route in `mobile_bridge.py`:
   ```python
   @app.route('/custom')
   def custom_page():
       return open('mobile_control/web/custom.html').read()
   ```

## Troubleshooting

### Bridge won't start
- Ensure simulation is running: `docker ps | grep parol6_dev`
- Check port 5000 is free: `lsof -i :5000`

### Can't connect from phone
- Ensure same WiFi network
- Check firewall: `sudo ufw allow 5000`
- Find computer IP: `hostname -I`

### Robot not moving
- Check action server: `docker exec parol6_dev bash -c "ros2 action list"`
- Verify controllers: `docker exec parol6_dev bash -c "ros2 control list_controllers"`

## Security Notes

**⚠️ Warning**: This is a development setup. For production:
- Add authentication (JWT tokens)
- Use HTTPS
- Implement rate limiting
- Add input validation
- Use environment variables for secrets

## Architecture for Future AI Integration

The system is designed with AI integration in mind:

```python
# Placeholder for AI module
class AIController:
    def __init__(self, robot_bridge):
        self.bridge = robot_bridge
        # Load vision models
        # Load control policies
        
    def autonomous_pick(self, object_coords):
        # AI-driven pick motion
        pass
```

Ready to add:
- Computer vision (OpenCV, YOLO)
- Reinforcement learning (PyTorch, TensorFlow)
- LLM integration (voice commands)

---

**Status**: ✅ Core system functional  
**Next**: Add camera streaming
