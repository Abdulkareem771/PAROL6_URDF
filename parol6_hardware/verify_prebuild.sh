#!/bin/bash
# Pre-Build Verification Script for Day 1 SIL
# Run this before building to catch common issues early

set -e

echo "=============================================="
echo "Day 1 SIL Pre-Build Verification"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ISSUES=0

# Check 1: ROS 2 environment
echo -n "✓ Checking ROS 2 environment... "
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${RED}FAIL${NC}"
    echo "  ROS 2 environment not sourced!"
    echo "  Run: source /opt/ros/humble/setup.bash"
    ISSUES=$((ISSUES+1))
else
    echo -e "${GREEN}OK${NC} (ROS_DISTRO=${ROS_DISTRO})"
fi

# Check 2: In correct directory
echo -n "✓ Checking workspace directory... "
if [ ! -f "parol6_hardware/package.xml" ]; then
    echo -e "${RED}FAIL${NC}"
    echo "  Not in workspace root!"
    echo "  Expected: /workspace or ~/Desktop/PAROL6_URDF"
    ISSUES=$((ISSUES+1))
else
    echo -e "${GREEN}OK${NC}"
fi

# Check 3: Dependencies installed
echo -n "✓ Checking serial package... "
if dpkg -l | grep -q libserial-dev; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo "  libserial-dev not found"
    echo "  Install: sudo apt-get install libserial-dev"
    ISSUES=$((ISSUES+1))
fi

# Check 4: parol6 package exists
echo -n "✓ Checking parol6 package... "
if ros2 pkg prefix parol6 &>/dev/null; then
    echo -e "${GREEN}OK${NC}"
elif [ -d "PAROL6" ]; then
    echo -e "${YELLOW}WARNING${NC}"
    echo "  parol6 package not built yet"
    echo "  Run: colcon build --packages-select parol6"
else
    echo -e "${RED}FAIL${NC}"
    echo "  PAROL6 directory not found!"
    ISSUES=$((ISSUES+1))
fi

# Check 5: Critical files exist
echo -n "✓ Checking package files... "
MISSING=""
for file in \
    "parol6_hardware/package.xml" \
    "parol6_hardware/CMakeLists.txt" \
    "parol6_hardware/parol6_hardware_plugin.xml" \
    "parol6_hardware/src/parol6_system.cpp" \
    "parol6_hardware/include/parol6_hardware/parol6_system.hpp" \
    "parol6_hardware/urdf/parol6.urdf.xacro" \
    "parol6_hardware/urdf/parol6.ros2_control.xacro"
do
    if [ ! -f "$file" ]; then
        MISSING="$MISSING\n  - $file"
    fi
done

if [ -z "$MISSING" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo -e "  Missing files:$MISSING"
    ISSUES=$((ISSUES+1))
fi

# Check 6: XML syntax
echo -n "✓ Checking XML files syntax... "
XML_ERRORS=""
for file in parol6_hardware/parol6_hardware_plugin.xml parol6_hardware/package.xml; do
    if ! xmllint --noout "$file" 2>/dev/null; then
        XML_ERRORS="$XML_ERRORS\n  - $file"
    fi
done

if [ -z "$XML_ERRORS" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo -e "  XML syntax issues (or xmllint not installed):$XML_ERRORS"
fi

# Check 7: Plugin export in CMakeLists
echo -n "✓ Checking plugin export... "
if grep -q "pluginlib_export_plugin_description_file" parol6_hardware/CMakeLists.txt; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "  pluginlib_export_plugin_description_file not found in CMakeLists.txt"
    ISSUES=$((ISSUES+1))
fi

# Check 8: Plugin installation
echo -n "✓ Checking plugin XML installation... "
if grep -q "FILES parol6_hardware_plugin.xml" parol6_hardware/CMakeLists.txt; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "  Plugin XML not installed in CMakeLists.txt"
    ISSUES=$((ISSUES+1))
fi

echo ""
echo "=============================================="
if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo "Ready to build with:"
    echo "  colcon build --packages-select parol6_hardware"
    exit 0
else
    echo -e "${RED}❌ Found $ISSUES issue(s)${NC}"
    echo "Fix the issues above before building."
    exit 1
fi
