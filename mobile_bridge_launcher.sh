#!/bin/bash

# Source ROS 2
source /opt/ros/humble/setup.bash

# Source your workspace if it exists
if [ -f "/workspace/install/setup.bash" ]; then
    source /workspace/install/setup.bash
fi

# Install required Python packages
pip install flask flask-cors opencv-python --quiet

# Run the mobile bridge
cd /workspace
python3 mobile_bridge.py
