#!/bin/bash
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && echo GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH"
