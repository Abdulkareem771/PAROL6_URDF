FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV IDF_PATH=/opt/esp-idf
ENV PATH=/opt/esp-idf/tools:$PATH

# Core ROS + simulation + build dependencies (including ESP-IDF toolchain reqs)
RUN apt-get update && apt-get install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-moveit \
    ros-humble-moveit-configs-utils \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-ros-visualization \
    ros-humble-rviz2 \
    ros-humble-ros-ign-gazebo \
    ros-humble-ros-ign-bridge \
    ros-humble-ign-ros2-control \
    gazebo \
    git wget flex bison gperf python3 python3-pip python3-venv cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0 \
    build-essential python3-serial python3-empy python3-catkin-pkg python3-rosdep python3-colcon-common-extensions udev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install ESP-IDF v5.1 (with resume-friendly clone) and Python deps for micro-ROS builds
RUN mkdir -p ${IDF_PATH} && \
    git clone -b v5.1 --recursive https://github.com/espressif/esp-idf.git ${IDF_PATH} && \
    ${IDF_PATH}/install.sh all && \
    source ${IDF_PATH}/export.sh && \
    pip3 install --no-cache-dir catkin_pkg lark-parser colcon-common-extensions empy==3.3.4

# Build micro-ROS agent workspace once in the image so it is ready to use
RUN source /opt/ros/humble/setup.bash && \
    mkdir -p /microros_ws/src && \
    cd /microros_ws/src && \
    git clone -b humble https://github.com/micro-ROS/micro_ros_setup.git && \
    cd /microros_ws && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y && \
    colcon build && \
    source install/setup.bash && \
    ros2 run micro_ros_setup create_agent_ws.sh && \
    ros2 run micro_ros_setup build_agent.sh

# Persist environment conveniences for interactive shells
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /opt/esp-idf/export.sh" >> /root/.bashrc && \
    echo "source /microros_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export IDF_PATH=/opt/esp-idf" >> /root/.bashrc && \
    echo "export PATH=/opt/esp-idf/tools:\$PATH" >> /root/.bashrc

CMD ["/bin/bash"]
