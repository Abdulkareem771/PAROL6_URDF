#!/bin/bash
echo "=== XBOX CONTROLLER DEPLOYMENT ==="

# Install dependencies
echo "1. Installing dependencies..."
sudo apt update > /dev/null 2>&1
sudo apt install -y ros-humble-joy joystick jstest-gtk > /dev/null 2>&1
pip install pygame pyyaml > /dev/null 2>&1
echo "✓ Dependencies installed"

# Test controller
echo "2. Testing controller detection..."
CONTROLLERS=$(python3 -c "import pygame; pygame.init(); print(pygame.joystick.get_count())" 2>/dev/null)
echo "✓ Controllers found: $CONTROLLERS"

# Build package
echo "3. Building ROS package..."
cd /home/ros2_ws
colcon build --packages-select parol6_control > /dev/null 2>&1
source install/setup.bash
echo "✓ Package built"

# Final verification
echo "4. Final verification..."
ros2 pkg list | grep -q parol6_control && echo "✓ parol6_control package found" || echo "✗ Package missing"
which jstest > /dev/null && echo "✓ jstest installed" || echo "✗ jstest missing"

echo "=== DEPLOYMENT COMPLETE ==="
echo "Run: ros2 launch parol6_control xbox_control.launch.py"
