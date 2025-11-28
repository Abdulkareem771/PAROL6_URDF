#!/bin/bash
# Script to install dependencies using rosdep

echo "Checking for missing dependencies..."

# Check if we are inside the container
if [ ! -f "/.dockerenv" ]; then
    echo "⚠️  WARNING: You are running this script on the HOST machine."
    echo "It is recommended to run this INSIDE the Docker container."
    echo "Use: docker exec -it parol6_dev bash -c './setup_dependencies.sh'"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Source ROS 2
source /opt/ros/humble/setup.bash

# Initialize rosdep if not already done
if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then
    echo "Initializing rosdep..."
    sudo rosdep init
fi

# Update rosdep
echo "Updating rosdep database..."
rosdep update

# Install dependencies
echo "Installing dependencies..."
rosdep install --from-paths . --ignore-src -r -y

echo "✅ Dependencies installed!"
