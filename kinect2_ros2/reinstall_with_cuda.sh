#!/bin/bash
set -e

echo "--- Installing Performance Dependencies ---"
apt-get update
# Install OpenMP for CPU multi-threading (kinect2_calibration)
apt-get install -y libomp-dev

# 1. Rebuild libfreenect2 with CUDA support
# Note: This assumes you have NVIDIA drivers and CUDA toolkit available in the container.
# If not, you might need: apt-get install -y nvidia-cuda-toolkit
echo "--- Rebuilding libfreenect2 with CUDA ---"
cd /workspace
if [ -d "libfreenect2/build" ]; then
    rm -rf "libfreenect2/build"
fi
mkdir -p libfreenect2/build && cd libfreenect2/build
cmake .. -DENABLE_CXX11=ON -DBUILD_OPENNI2_DRIVER=OFF -DENABLE_OPENCL=ON -DENABLE_CUDA=ON -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
make install
ldconfig

# 2. Rebuild ROS Workspace
echo "--- Rebuilding ROS Workspace ---"
cd /workspace
# Force clean kinect2_calibration to ensure it picks up OpenMP
rm -rf build/kinect2_calibration install/kinect2_calibration

colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3 --packages-select kinect2_bridge kinect2_calibration kinect2_registration

echo "--- Done! Source the workspace and run calibration. ---"
echo "source install/setup.bash"
echo "ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 calibrate color"
