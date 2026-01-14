#!/bin/bash
# Post-Build Verification Script for Day 1 SIL
# Run this after building to verify installation

set -e

echo "=============================================="
echo "Day 1 SIL Post-Build Verification"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ISSUES=0

# Source workspace
if [ ! -f "install/setup.bash" ]; then
    echo -e "${RED}❌ install/setup.bash not found!${NC}"
    echo "Build the package first:"
    echo "  colcon build --packages-select parol6_hardware"
    exit 1
fi

source install/setup.bash

# Check 1: Package installed
echo -n "✓ Package installed... "
if ros2 pkg prefix parol6_hardware &>/dev/null; then
    PREFIX=$(ros2 pkg prefix parol6_hardware)
    echo -e "${GREEN}OK${NC} ($PREFIX)"
else
    echo -e "${RED}FAIL${NC}"
    ISSUES=$((ISSUES+1))
fi

# Check 2: Plugin XML installed
echo -n "✓ Plugin XML installed... "
PLUGIN_XML="$PREFIX/share/parol6_hardware/parol6_hardware_plugin.xml"
if [ -f "$PLUGIN_XML" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "  Expected: $PLUGIN_XML"
    ISSUES=$((ISSUES+1))
fi

# Check 3: Launch files installed
echo -n "✓ Launch files installed... "
if [ -d "$PREFIX/share/parol6_hardware/launch" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    ISSUES=$((ISSUES+1))
fi

# Check 4: Config files installed
echo -n "✓ Config files installed... "
if [ -f "$PREFIX/share/parol6_hardware/config/parol6_controllers.yaml" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    ISSUES=$((ISSUES+1))
fi

# Check 5: URDF files installed
echo -n "✓ URDF files installed... "
if [ -f "$PREFIX/share/parol6_hardware/urdf/parol6.urdf.xacro" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    ISSUES=$((ISSUES+1))
fi

# Check 6: Shared library installed
echo -n "✓ Shared library installed... "
if [ -f "$PREFIX/lib/libparol6_hardware.so" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    ISSUES=$((ISSUES+1))
fi

# Check 7: Plugin discoverable
echo -n "✓ Plugin discoverable... "
if ros2 pkg xml parol6_hardware | grep -q "hardware_interface"; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo "  Plugin export may not be configured correctly"
fi

# Check 8: URDF can be processed
echo -n "✓ URDF processing... "
if xacro "$PREFIX/share/parol6_hardware/urdf/parol6.urdf.xacro" > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "  URDF/xacro processing failed"
    ISSUES=$((ISSUES+1))
fi

echo ""
echo "=============================================="
echo "Installation Structure:"
echo "=============================================="
tree -L 2 "$PREFIX/share/parol6_hardware" 2>/dev/null || ls -R "$PREFIX/share/parol6_hardware"

echo ""
echo "=============================================="
if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "Ready to launch with:"
    echo "  ros2 launch parol6_hardware real_robot.launch.py"
    echo ""
    echo "Validate with:"
    echo "  ros2 control list_controllers"
    echo "  ros2 topic hz /joint_states"
    exit 0
else
    echo -e "${RED}❌ Found $ISSUES issue(s)${NC}"
    echo "Check the build output for errors."
    exit 1
fi
