FROM osrf/ros:humble-desktop

# Install Gazebo Classic, MoveIt, and ROS 2 integration
RUN apt-get update && apt-get install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-moveit \
    ros-humble-moveit-configs-utils \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-ros-visualization \
    ros-humble-rviz2 \
    gazebo \
    build-essential \
    cmake \
    pkg-config \
    libusb-1.0-0-dev \
    libturbojpeg0-dev \
    libopenni2-dev \
    git \
    ros-humble-image-view \
    ros-humble-compressed-image-transport \
    ros-humble-compressed-depth-image-transport \
    ros-humble-image-pipeline \
    && rm -rf /var/lib/apt/lists/*



# Install libfreenect2 (CPU-only, no GPU dependencies)
RUN cd /tmp && \
    git clone https://github.com/OpenKinect/libfreenect2.git && \
    cd libfreenect2 && \
    mkdir build && cd build && \
    cmake .. \
    -DENABLE_OPENGL=OFF \
    -DENABLE_CUDA=OFF \
    -DENABLE_OPENCL=OFF \
    -DENABLE_TLS=ON && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    mkdir -p /etc/udev/rules.d/ && \
    cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/ && \
    cd / && rm -rf /tmp/libfreenect2

# Build Kinect ROS2 bridge
SHELL ["/bin/bash", "-c"]
RUN mkdir -p /opt/kinect_ws/src && \
    cd /opt/kinect_ws/src && \
    git clone https://github.com/krepa098/kinect2_ros2.git && \
    cd /opt/kinect_ws && \
    source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

# Set working directory
WORKDIR /workspace

# Source ROS 2 and Kinect workspace automatically
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /opt/kinect_ws/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]

# Install dependencies for images extraction (RGB + Depth extraction tools)
RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-default-plugins \
    python3-opencv \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Install System & ROS Dependencies for Real Robot & Simulation (Layered for Cache)
RUN apt-get update && apt-get install -y \
    python3-serial \
    socat \
    python3-pip \
    python3-scipy \
    ros-humble-ros-ign-bridge \
    ros-humble-ros-ign-gazebo \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (YOLOv8, esptool)
RUN pip3 install ultralytics esptool

# Install libserial-dev for parol6_hardware (added last to preserve cache)
RUN apt-get update && apt-get install -y libserial-dev && rm -rf /var/lib/apt/lists/*

# Install Teensy Build Tools (PlatformIO + teensy_loader_cli)
RUN apt-get update && apt-get install -y teensy-loader-cli && rm -rf /var/lib/apt/lists/*
RUN pip3 install platformio

