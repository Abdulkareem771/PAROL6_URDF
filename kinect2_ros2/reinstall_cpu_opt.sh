#!/bin/bash
set -e

echo "--- Installing CPU Performance Dependencies ---"
apt-get update
# Install OpenMP for CPU multi-threading
apt-get install -y libomp-dev

# Rebuild ROS Workspace for kinect2_calibration
echo "--- Rebuilding kinect2_calibration with OpenMP ---"
cd /workspace

# Force clean kinect2_calibration to ensure it picks up OpenMP
rm -rf build/kinect2_calibration install/kinect2_calibration

# Build only the calibration package (and its deps if needed)
# optimizing specifically for Release mode
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3 --packages-select kinect2_calibration

echo "--- Done! ---"
echo "You can now run the optimized calibration:"
echo "source install/setup.bash"
echo "ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 calibrate color"
