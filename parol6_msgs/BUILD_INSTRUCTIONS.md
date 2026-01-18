# Building parol6_msgs Package - Step-by-Step Guide

This guide provides detailed instructions for building the custom message package.

---

## Prerequisites

✅ PAROL6 Docker container running  
✅ ROS2 Humble environment  
✅ Workspace at `/workspace` or `/home/osama/Desktop/PAROL6_URDF`

---

## Build Steps

### Step 1: Enter Docker Container

```bash
# Start container if not running
cd /home/osama/Desktop/PAROL6_URDF
./start_container.sh

# Enter container
docker exec -it parol6_dev bash
```

### Step 2: Navigate to Workspace

```bash
cd /workspace
```

### Step 3: Source ROS2

```bash
source /opt/ros/humble/setup.bash
```

### Step 4: Build parol6_msgs

```bash
# Build only the messages package
colcon build --packages-select parol6_msgs

# Expected output:
# Starting >>> parol6_msgs
# Finished <<< parol6_msgs [X.XXs]
# 
# Summary: 1 package finished
```

###Step 5: Source the Workspace

```bash
source install/setup.bash
```

### Step 6: Verify Message Generation

```bash
# List all parol6_msgs
ros2 interface list | grep parol6_msgs
```

**Expected output:**
```
parol6_msgs/msg/WeldLine
parol6_msgs/msg/WeldLine3D
parol6_msgs/msg/WeldLine3DArray
parol6_msgs/msg/WeldLineArray
```

### Step 7: Inspect Message Definitions

```bash
# View WeldLine structure
ros2 interface show parol6_msgs/msg/WeldLine

# View WeldLine3D structure
ros2 interface show parol6_msgs/msg/WeldLine3D
```

**Expected output for WeldLine:**
```
string id
float32 confidence
geometry_msgs/Point32[] pixels
geometry_msgs/Point bbox_min
geometry_msgs/Point bbox_max
std_msgs/Header header
```

---

## Testing the Messages

### Test 1: Import in Python

```bash
python3 << 'EOF'
from parol6_msgs.msg import WeldLine, WeldLineArray
from parol6_msgs.msg import WeldLine3D, WeldLine3DArray
print("✅ All messages imported successfully!")
EOF
```

### Test 2: Create Message Instance

```bash
python3 << 'EOF'
from parol6_msgs.msg import WeldLine
from geometry_msgs.msg import Point32

line = WeldLine()
line.id = "test_line"
line.confidence = 0.95
line.pixels = [Point32(x=100.0, y=200.0, z=0.0)]

print(f"✅ Created WeldLine: id={line.id}, confidence={line.confidence}")
print(f"   Pixels: {len(line.pixels)} points")
EOF
```

### Test 3: Publisher Test

```bash
# Terminal 1: Start listener
ros2 topic echo /test/weld_lines

# Terminal 2: Publish test message
ros2 topic pub --once /test/weld_lines parol6_msgs/msg/WeldLineArray \
  "{header: {frame_id: 'camera'}, lines: []}"
```

---

## Troubleshooting

### Issue: "colcon: command not found"

**Cause:** Not inside Docker container or ROS2 not sourced

**Solution:**
```bash
# Enter container
docker exec -it parol6_dev bash

# Source ROS2
source /opt/ros/humble/setup.bash
```

### Issue: "No packages found"

**Cause:** Wrong directory or package not in workspace

**Solution:**
```bash
# Verify you're in workspace root
pwd
# Should show: /workspace

# Verify parol6_msgs exists
ls parol6_msgs/
# Should show: CMakeLists.txt  msg/  package.xml  README.md
```

### Issue: "Module not found" when importing

**Cause:** Workspace not sourced

**Solution:**
```bash
source /workspace/install/setup.bash

# Add to ~/.bashrc for persistence
echo "source /workspace/install/setup.bash" >> ~/.bashrc
```

### Issue: Build errors

**Solution:**
```bash
# Clean build
rm -rf build/ install/ log/

# Rebuild
colcon build --packages-select parol6_msgs --cmake-clean-cache
```

---

## Next Steps

After successfully building `parol6_msgs`:

1. ✅ **Messages package ready**
2. 📝 **Review**: [parol6_msgs/README.md](file:///home/osama/Desktop/PAROL6_URDF/parol6_msgs/README.md)
3. 🔨 **Next**: Create `parol6_vision` package
4. 📖 **Reference**: [Vision Developer Guide](file:///home/osama/Desktop/PAROL6_URDF/docs/VISION_DEVELOPER_GUIDE.md)

---

## Quick Reference

```bash
# Build
colcon build --packages-select parol6_msgs

# Source
source install/setup.bash

# Verify
ros2 interface list | grep parol6_msgs

# Test import
python3 -c "from parol6_msgs.msg import WeldLine; print('OK')"
```

---

**Ready to proceed with vision package implementation!** 🚀
