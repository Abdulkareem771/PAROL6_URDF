# Depth Accuracy Testing Instructions

## Quick Method (Command Line)

### 1. Echo depth topic and measure center point:
```bash
# In container, start kinect
source /opt/kinect_ws/install/setup.bash
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml fps_limit:=5

# In another terminal, check depth value
ros2 topic echo /kinect2/sd/image_depth_rect --once
```

## Better Method (Visual Tool)

### 1. Copy the test script to the container:
```bash
# From host
docker cp /home/helsof/Desktop/PAROL6_URDF/vision_work/test_depth_accuracy.py parol6_dev:/workspace/
```

### 2. Run the depth tester inside the container:
```bash
# Enter container
docker exec -it parol6_dev bash

# Install dependencies if needed
pip3 install opencv-python

# Run the test
cd /workspace
python3 test_depth_accuracy.py
```

## How to Test:

### Setup:
1. Place a flat object (wall, book, box) in front of the camera
2. Use a tape measure to measure the actual distance from the camera lens to the object
3. Make sure the object is perpendicular to the camera
4. Ensure good lighting (not too dark)

### Test Procedure:
1. Run the script - a window will open showing the depth image
2. Point the camera at your test surface
3. The green crosshair shows where it's measuring (center by default)
4. **Click anywhere** on the image to measure depth at that point
5. Compare the displayed distance with your tape measure reading

### Expected Accuracy:
- **Close range (0.5-2m)**: ±10-20mm typical error
- **Mid range (2-4m)**: ±20-50mm typical error  
- **Far range (4-5m)**: ±50-100mm typical error

### Tips for Best Results:
- ✅ Use a flat, matte surface (not shiny/reflective)
- ✅ Measure at least 5 different distances
- ✅ Keep the surface perpendicular to camera
- ✅ Avoid edges and corners
- ❌ Don't test on glass, mirrors, or very dark surfaces
- ❌ Avoid direct sunlight or IR interference

## Recording Your Results:

Create a test table:

| Measured Distance (m) | Camera Reading (m) | Error (mm) | Error (%) |
|-----------------------|-------------------|------------|-----------|
| 0.50                  |                   |            |           |
| 1.00                  |                   |            |           |
| 1.50                  |                   |            |           |
| 2.00                  |                   |            |           |
| 3.00                  |                   |            |           |

## Troubleshooting:

**"No valid depth data"**: 
- Surface too close (< 0.5m) or too far (> 4.5m)
- Surface is too reflective/absorptive
- IR emitter blocked

**High standard deviation (> 50mm)**:
- Surface not flat
- Measuring at an edge or corner  
- Camera not stable (vibration)

**Systematic offset (always too short/long)**:
- May need to recalibrate depth
- Check if using correct depth topic (sd/qhd/hd)
