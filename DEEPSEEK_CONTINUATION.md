# CONTINUATION PROMPT FOR DEEPSEEK

Use this prompt to continue work on the PAROL6 mobile control system.

---

## CONTEXT SUMMARY

I am working on the PAROL6 robot project. The main branch has a functional Ignition Gazebo + MoveIt setup. I've created a **mobile-ros** branch with a complete mobile control system.

## CURRENT STATUS

**Branch:** `mobile-ros`  
**Location:** `/home/kareem/Desktop/PAROL6_URDF`

**What's Working:**
✅ Flask-based REST API server (`mobile_bridge.py`)
✅ Modern web interface with joint sliders
✅ Real-time joint state monitoring
✅ Mobile-responsive design
✅ Complete documentation (`MOBILE_CONTROL_GUIDE.md`, `MOBILE_ROS_SUMMARY.md`)

**Git Status:**
- All changes committed to `mobile-ros` branch
- Branch is clean and ready for new work

## HOW TO TEST CURRENT SYSTEM

```bash
# Terminal 1
./start_ignition.sh

# Terminal 2  
./start_mobile_bridge.sh

# Browser
open http://localhost:5000
```

## WHAT TO BUILD NEXT

I want to enhance the mobile control system with the following features (in priority order):

### Priority 1: Camera Streaming
Add live camera feed to the web interface so I can see what the robot sees while controlling it remotely.

**Requirements:**
- Stream camera from Gazebo simulation
- Display in web interface
- Low latency (< 500ms)
- Work on mobile devices

**Hints:**
- Use MJPEG streaming or WebRTC
- Subscribe to `/camera/image_raw` topic
- Convert ROS Image messages to JPEG
- Add `<img src="/api/camera/stream">` to web UI

### Priority 2: Saved Positions (Teach Mode)
Allow saving and replaying robot positions.

**Requirements:**
- Save current position with a name
- List all saved positions
- Load/execute saved positions
- Persist to JSON file

**Hints:**
- Add `/api/positions/save` endpoint
- Add `/api/positions/list` endpoint
- Add `/api/positions/load/<name>` endpoint
- Store in `saved_positions.json`

### Priority 3: Cartesian Control
Control end-effector position in X/Y/Z instead of joint angles.

**Requirements:**
- Input X/Y/Z coordinates
- Automatic inverse kinematics
- Visual feedback of reachability

**Hints:**
- Use MoveIt's `compute_ik` service
- Add sliders for X/Y/Z coordinates
- Show workspace bounds

### Priority 4: Advanced Features (Future)
- Emergency stop button
- Speed control
- Voice commands (Whisper API)
- Gesture recognition (MediaPipe)
- Object detection (YOLO)

## CODE STRUCTURE

**Main Files:**
- `mobile_bridge.py` - Flask server + ROS 2 bridge (start here)
- `start_mobile_bridge.sh` - Launch script
- `MOBILE_CONTROL_GUIDE.md` - Full documentation
- `MOBILE_ROS_SUMMARY.md` - Implementation details

**Key Classes:**
```python
class RobotBridge(Node):  # ROS 2 node for robot interface
    def __init__(self): ...
    def send_joint_positions(self, positions, duration): ...
    def get_joint_positions(self): ...
```

**Flask App Structure:**
```python
@app.route('/api/status')    # Get robot state
@app.route('/api/move')      # Send trajectory
@app.route('/api/home')      # Go to home
@app.route('/')              # Web interface (HTML embedded)
```

## SPECIFIC INSTRUCTIONS FOR CAMERA STREAMING

To add camera streaming (Priority 1), modify `mobile_bridge.py`:

```python
# Add to RobotBridge.__init__():
self.bridge = CvBridge()
self.camera_sub = self.create_subscription(
    Image,
    '/camera/image_raw',  # or appropriate camera topic
    self.camera_callback,
    10
)
self.latest_frame = None

def camera_callback(self, msg):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        ret, jpeg = cv2.imencode('.jpg', cv_image)
        self.latest_frame = jpeg.tobytes()
    except Exception as e:
        self.get_logger().error(f'Camera error: {e}')

# Add Flask route:
def generate_frames():
    while True:
        if robot_bridge and robot_bridge.latest_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   robot_bridge.latest_frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/api/camera/stream')
def camera_stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Add to HTML:
<img src="/api/camera/stream" style="width: 100%; border-radius: 8px;">
```

**Dependencies needed:**
```bash
docker exec parol6_dev pip install opencv-python
```

## TESTING CHECKLIST

Before claiming a feature is complete:

- [ ] Code runs without errors
- [ ] API endpoints return correct responses
- [ ] Web interface displays properly
- [ ] Works on mobile devices (test with phone)
- [ ] Documentation updated
- [ ] Git commit with clear message

## GIT WORKFLOW

```bash
# Make changes to files
git add .
git commit -m "feat: your feature description"

# When ready to merge back to main:
git checkout main
git merge mobile-ros
git push origin main
git push origin mobile-ros
```

## IMPORTANT NOTES

1. **Always test in Docker container** - Don't run directly on host
2. **Flask runs inside container** - Access via host's IP
3. **ROS 2 topics** - Use `ros2 topic list` to find camera topic name
4. **Port 5000** - Make sure it's not blocked by firewall

## TROUBLESHOOTING

**Flask won't start:**
```bash
docker exec parol6_dev pip install flask flask-cors
```

**Camera not working:**
```bash
# Check camera topic exists
docker exec parol6_dev bash -c "ros2 topic list | grep camera"

# Check if images are being published
docker exec parol6_dev bash -c "ros2 topic hz /camera/image_raw"
```

**Can't access from phone:**
```bash
# Find your IP
hostname -I

# Allow port through firewall
sudo ufw allow 5000
```

## FILE LOCATIONS

```
PAROL6_URDF/
├── mobile_bridge.py              # MAIN FILE - Flask + ROS bridge
├── start_mobile_bridge.sh        # Startup script
├── MOBILE_CONTROL_GUIDE.md       # User documentation
├── MOBILE_ROS_SUMMARY.md         # Technical summary
└── README.md                     # Updated with mobile control section
```

## SUCCESS MESSAGE

When you've successfully added camera streaming, you should see:
```
✓ Live camera feed visible in web interface
✓ Low latency (< 500ms)
✓ Works on mobile devices
✓ Smooth video playback
```

## YOUR TASK

**Primary**: Implement camera streaming (see code example above)

**Secondary**: If camera works, add saved positions feature

**Stretch**: Implement cartesian control with IK

Good luck! The foundation is solid, and the architecture is clean. Adding these features should be straightforward. 🚀
