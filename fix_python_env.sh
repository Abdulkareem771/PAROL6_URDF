#!/bin/bash
# Permanent fix for Python environment conflicts
# Run this ONCE inside the parol6_dev container

echo "════════════════════════════════════════════════════════"
echo "  Fixing Python Environment Conflicts (ESP-IDF vs ROS)"
echo "════════════════════════════════════════════════════════"
echo ""

# Add permanent environment variable to bashrc
if ! grep -q "PYTHON_EXECUTABLE=/usr/bin/python3" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Fix for ESP-IDF/ROS Python conflicts" >> ~/.bashrc
    echo "export PYTHON_EXECUTABLE=/usr/bin/python3" >> ~/.bashrc
    echo "✅ Added PYTHON_EXECUTABLE to ~/.bashrc"
else
    echo "ℹ️  PYTHON_EXECUTABLE already set in ~/.bashrc"
fi

# Set for current session
export PYTHON_EXECUTABLE=/usr/bin/python3
echo "✅ Set PYTHON_EXECUTABLE for current session"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Fix Applied Successfully!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "What this does:"
echo "  - Forces ROS colcon to use system Python (/usr/bin/python3)"
echo "  - Prevents ESP-IDF Python from interfering with ROS builds"
echo "  - Persists across container restarts"
echo ""
echo "Verification:"
echo "  Run: echo \$PYTHON_EXECUTABLE"
echo "  Expected: /usr/bin/python3"
echo ""
echo "Now you can build ROS packages without errors!"
echo ""
