#!/bin/bash
# Quick verification script for Kinect2 improvements

echo "======================================"
echo "Kinect2 Improvements Verification"
echo "======================================"

# Test 1: OpenMP
echo ""
echo "Test 1: Checking OpenMP linkage..."
if docker exec parol6_dev ldd /opt/kinect_ws/install/kinect2_bridge/lib/kinect2_bridge/kinect2_bridge_node 2>/dev/null | grep -q "libgomp"; then
    echo "✅ PASS: OpenMP is linked"
else
    echo "❌ FAIL: OpenMP not found"
fi

# Test 2: Check if bridge can start
echo ""
echo "Test 2: Checking if bridge can start..."
docker exec parol6_dev bash -c "source /opt/kinect_ws/install/setup.bash && timeout 5 ros2 run kinect2_bridge kinect2_bridge 2>&1" | grep -q "error" 
if [ $? -eq 1 ]; then
    echo "✅ PASS: Bridge starts without errors"
else
    echo "❌ FAIL: Bridge has startup errors"
fi

# Test 3: Check build artifacts
echo ""
echo "Test 3: Checking build artifacts..."
if docker exec parol6_dev test -f /opt/kinect_ws/install/kinect2_registration/lib/libkinect2_registration.so; then
    echo "✅ PASS: kinect2_registration library exists"
else
    echo "❌ FAIL: kinect2_registration library missing"
fi

# Test 4: Check source modifications
echo ""
echo "Test 4: Checking source code modifications..."
if docker exec parol6_dev grep -q "fillHoles" /opt/kinect_ws/src/kinect2_ros2/kinect2_registration/src/depth_registration_cpu.h 2>/dev/null; then
    echo "✅ PASS: Hole filling code present"
else
    echo "❌ FAIL: Hole filling code not found"
fi

if docker exec parol6_dev grep -q "cv::inpaint" /opt/kinect_ws/src/kinect2_ros2/kinect2_registration/src/depth_registration_cpu.cpp 2>/dev/null; then
    echo "✅ PASS: Inpainting function present"
else
    echo "❌ FAIL: Inpainting function not found"
fi

echo ""
echo "======================================"
echo "Verification Complete"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Run: docker exec -it parol6_dev bash"
echo "2. Run: source /opt/kinect_ws/install/setup.bash"
echo "3. Run: ros2 run kinect2_bridge kinect2_bridge"
echo "4. In another terminal, run: ros2 topic hz /kinect2/qhd/image_depth_rect"
echo "5. Expected: 25-30 Hz (vs 15-20 Hz before)"
