FROM osrf/ros:humble-desktop

# Install Gazebo Classic, MoveIt, and ROS 2 integration
RUN apt-get update && apt-get install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-moveit \
    ros-humble-moveit-configs-utils \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-ros-visualization \
    ros-humble-moveit-servo \
    ros-humble-joy \
    ros-humble-joint-state-publisher \
    ros-humble-rviz2 \
    gazebo \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Source ROS 2 automatically
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
