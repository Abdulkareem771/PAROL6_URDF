# Kinect v2 Calibration Guide for ROS 2

**Complete calibration workflow for kinect2_ros2 on ROS 2 Humble**

This guide walks you through calibrating your Kinect v2 sensor for accurate depth-to-color registration and improved point cloud quality. The process takes approximately 1-2 hours.

---

## Why Calibrate?

**Without calibration:**
- Depth and RGB images are misaligned
- Point clouds have color artifacts at edges
- Depth measurements have systematic offset (~24mm)

**After calibration:**
- Perfect depth-to-RGB alignment
- Clean point clouds with accurate color mapping
- Depth measurements corrected to ground truth

---

## Prerequisites

### Hardware Required
- ✅ Kinect v2 sensor (connected via USB 3.0)
- ✅ **Two tripods** (one for Kinect, one for calibration pattern)
- ✅ Printed calibration pattern (see below)
- ✅ **Flat mounting surface** for pattern (foam board, cardboard, etc.)

### Software Setup
```bash
# Inside Docker container
docker exec -it parol6_dev bash
source /opt/kinect_ws/install/setup.bash
```

### Verify kinect2_bridge is working
```bash
# Start bridge with low FPS to reduce CPU load during calibration
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml fps_limit:=2.0
```

Press `Ctrl+C` to stop once you confirm it starts without errors.

---

## Step 1: Prepare Calibration Pattern

### Download and Print Pattern

The calibration tool supports multiple patterns. **Recommended: chess5x7x0.03**

**Available patterns in the package:**
```bash
docker exec parol6_dev bash -c "ls /opt/kinect_ws/src/kinect2_ros2/kinect2_calibration/patterns/"
```

- `chess5x7x0.03.pdf` ⭐ **Recommended** (fits A4 paper)
- `chess7x9x0.025.pdf` (larger pattern)
- `chess9x11x0.02.pdf` (smaller squares)

**Alternative patterns from OpenCV:**
- [Chessboard pattern](http://docs.opencv.org/2.4.2/_downloads/pattern.png)
- [Asymmetric circle grid](http://docs.opencv.org/2.4.2/_downloads/acircles_pattern.png)

### Print Instructions

1. **Print on good quality laser printer** (inkjet may blur edges)
2. **Print at 100% scale** (disable "Fit to page")
3. **Verify dimensions with caliper**:
   - For chess5x7x0.03: squares should be **exactly 3.0 cm × 3.0 cm**
   - Printers sometimes scale documents—this will break calibration!

4. **Mount pattern on flat surface**:
   - Use foam board, plexiglass, or thick cardboard
   - Glue pattern flat with **no bubbles or wrinkles**
   - Ensure surface is rigid and won't bend during calibration

> [!IMPORTANT]
> Pattern flatness is critical! Any warping will introduce calibration errors.

---

## Step 2: Setup Calibration Environment

### Tripod Setup

**Kinect Tripod:**
- Use ball head for easy positioning
- Height: ~1.5m (eye level)
- Stable surface (no vibration)

**Pattern Tripod:**
- Adjustable height
- Clamp or tape to hold pattern board
- Easy to move and rotate

### Lighting Conditions

- **Avoid direct sunlight** (causes IR saturation)
- **Indoor lighting is fine** for color camera
- **IR camera works in any lighting** (uses its own illumination)

### Test Visibility

```bash
# Start bridge
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml fps_limit:=2.0

# In another terminal, view color image
ros2 run rqt_image_view rqt_image_view /kinect2/hd/image_color

# View IR image  
ros2 run rqt_image_view rqt_image_view /kinect2/sd/image_ir
```

Check that the pattern is clearly visible in both images.

---

## Step 3: Create Calibration Directory

```bash
# Inside container
mkdir -p ~/kinect_cal_data
cd ~/kinect_cal_data
```

All calibration files will be saved here.

---

## Step 4: Start kinect2_bridge with Low FPS

**In Terminal 1:**
```bash
docker exec -it parol6_dev bash
source /opt/kinect_ws/install/setup.bash
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml fps_limit:=2.0
```

> **Why low FPS?** Reduces CPU/GPU load during calibration, preventing frame drops.

---

## Step 5: Calibrate Color Camera (Intrinsics)

### Record Color Images

**In Terminal 2:**
```bash
docker exec -it parol6_dev bash
source /opt/kinect_ws/install/setup.bash
cd ~/kinect_cal_data

# Record color images
ros2 run kinect2_calibration kinect2_calibration chess5x7x0.03 record color
```

**How to record images:**
1. Position pattern in different locations in the camera view
2. Pattern should be detected (colored lines overlay on pattern)
3. Press **SPACEBAR** to capture image when:
   - Pattern is fully visible and detected
   - Image is sharp (not blurred)
   - Kinect is stable (not moving)

**Recording strategy (aim for 100+ images):**

- **Close distance** (pattern fills most of frame):
  - Center position, straight-on
  - Tilt pattern vertically (toward/away from camera)
  - Tilt pattern horizontally (left/right)
  
- **Medium distance** (pattern covers ~30% of frame):
  - Move pattern to **all 9 grid positions**:
    ```
    Top-left    Top-center    Top-right
    Mid-left    Center        Mid-right
    Bottom-left Bottom-center Bottom-right
    ```
  - For each position, tilt pattern in different orientations

- **Far distance** (pattern is small):
  - Repeat several positions with tilts

Press **ESC** when finished (100+ images recommended).

### Calibrate Color Intrinsics

```bash
ros2 run kinect2_calibration kinect2_calibration chess5x7x0.03 calibrate color
```

**Expected output:**
```
[Info] [LOAD] 104 color images
[Info] Calibrating color camera...
[Info] RMS error: 0.234 pixels
[Info] Calibration saved to: calib_color.yaml
```

> **Target RMS error**: < 0.5 pixels (excellent), < 1.0 pixels (acceptable)

---

## Step 6: Calibrate IR Camera (Intrinsics)

### Enable IR Auto-Exposure (Important!)

The IR camera has limited dynamic range. For better pattern detection, use digital auto-exposure:

**Stop the bridge** (Terminal 1: `Ctrl+C`), then restart with:
```bash
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml \
  fps_limit:=2.0 \
  ir_auto_exposure:=true
```

### Record IR Images

**In Terminal 2:**
```bash
cd ~/kinect_cal_data
ros2 run kinect2_calibration kinect2_calibration chess5x7x0.03 record ir
```

Use the same recording strategy as color (100+ images, all positions/tilts).

**Tips:**
- IR sees better in low light (can turn off room lights if pattern isn't detected)
- Make sure pattern is detected (colored lines overlay)
- Press SPACEBAR for each good image

### Calibrate IR Intrinsics

```bash
ros2 run kinect2_calibration kinect2_calibration chess5x7x0.03 calibrate ir
```

**Expected output:**
```
[Info] [LOAD] 107 ir images
[Info] Calibrating ir camera...
[Info] RMS error: 0.189 pixels
[Info] Calibration saved to: calib_ir.yaml
```

---

## Step 7: Calibrate Extrinsics (Camera-to-Camera Transform)

This step finds the transformation between the color and IR cameras.

### Record Synchronized Images

```bash
cd ~/kinect_cal_data
ros2 run kinect2_calibration kinect2_calibration chess5x7x0.03 record sync
```

**Important:**
- Pattern must be detected in **both** color and IR simultaneously
- Use the same 100+ image strategy
- Ensure Kinect is **perfectly still** when pressing SPACEBAR
  - Tripod with ball head helps here—lock position, then capture

### Calibrate Extrinsics

```bash
ros2 run kinect2_calibration kinect2_calibration chess5x7x0.03 calibrate sync
```

**Expected output:**
```
[Info] [LOAD] 112 sync images
[Info] Calibrating extrinsics (color to ir transform)...
[Info] Calibration saved to: calib_pose.yaml
```

---

## Step 8: Calibrate Depth Measurements

This step corrects systematic depth offset (~24mm found in testing).

```bash
cd ~/kinect_cal_data
ros2 run kinect2_calibration kinect2_calibration chess5x7x0.03 calibrate depth
```

**Expected output:**
```
[Info] Using 112 sync images for depth calibration
[Info] Mean depth offset: -24.3 mm
[Info] Std deviation: 2.1 mm
[Info] Calibration saved to: calib_depth.yaml
[Info] Plot data saved to: plot.dat
```

### Optional: Visualize Depth Calibration

If you have gnuplot installed:
```bash
cd ~/kinect_cal_data
gnuplot

# In gnuplot:
set xlabel "Measured distance"
set ylabel "Computed distance"
plot 'plot.dat' using 3:4 with dots title "Depth calibration"
```

---

## Step 9: Install Calibration Files

### Find Your Kinect Serial Number

Look at the kinect2_bridge startup output (Terminal 1):
```
[Info] [Freenect2Impl] found valid Kinect v2 @2:3 with serial 018436651247
                                                              ^^^^^^^^^^^^^^
```

Your serial number is shown here (e.g., `018436651247`).

### Create Calibration Directory

```bash
# Inside container
export SERIAL=018436651247  # Replace with YOUR serial number
mkdir -p /opt/kinect_ws/install/kinect2_bridge/share/kinect2_bridge/data/$SERIAL
```

### Copy Calibration Files

```bash
cd ~/kinect_cal_data
cp calib_color.yaml calib_depth.yaml calib_ir.yaml calib_pose.yaml \
   /opt/kinect_ws/install/kinect2_bridge/share/kinect2_bridge/data/$SERIAL/

# Verify
ls -la /opt/kinect_ws/install/kinect2_bridge/share/kinect2_bridge/data/$SERIAL/
```

You should see all 4 files:
- `calib_color.yaml`
- `calib_ir.yaml`
- `calib_pose.yaml`
- `calib_depth.yaml`

---

## Step 10: Test Calibrated Bridge

### Restart Bridge

**Terminal 1** (stop old bridge with `Ctrl+C`, then):
```bash
source /opt/kinect_ws/install/setup.bash
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml
```

**Look for this in startup:**
```
[Info] [DepthRegistration] Using calibration data from: .../data/018436651247/
[Info] [DepthRegistration] calib_color.yaml loaded
[Info] [DepthRegistration] calib_ir.yaml loaded
[Info] [DepthRegistration] calib_pose.yaml loaded
[Info] [DepthRegistration] calib_depth.yaml loaded
```

✅ If you see these messages, calibration is loaded!

---

## Step 11: Verify Calibration in RViz2

### Launch RViz2

**Terminal 2:**
```bash
rviz2
```

### Add Displays

1. **Add PointCloud2**:
   - Click "Add" → "By topic"
   - Select `/kinect2/qhd/points` → "PointCloud2"
   - Set "Color Transformer" to "RGB8"

2. **Add Camera**:
   - Click "Add" → "By display type" → "Camera"
   - Set "Image Topic" to `/kinect2/qhd/image_color_rect`

3. **Adjust view**:
   - Set "Fixed Frame" to `kinect2_link`
   - Use mouse to rotate point cloud

### What to Check

**Before calibration:**
- Color and depth are misaligned at edges
- Depth "bleeds" onto background at object boundaries
- Point cloud colors are offset

**After calibration:**
- Sharp color transitions at edges
- Depth aligns perfectly with RGB
- Clean point clouds with accurate colors

---

## Step 12: Commit Container (Persist Calibration)

Your calibration is now in `/opt/kinect_ws/install/`, but this will be lost if the container is recreated. **Commit the container** to save it:

```bash
# On host machine (not inside container)
docker commit parol6_dev parol6-ultimate:latest
```

---

## Advanced Features

### Depth Hole Filling

Fill black holes in depth maps using neighboring pixels:

```bash
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml hole_fill_radius:=2
```

**Values:**
- `0` = Disabled (default)
- `1` = Light filling
- `2-3` = Moderate filling (recommended)
- `4+` = Aggressive filling (may introduce artifacts)

### IR Auto-Exposure

Enable software dynamic range compression for IR:

```bash
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml ir_auto_exposure:=true
```

**Use cases:**
- Indoor calibration (mixed lighting)
- Environments with bright spots
- Better pattern detection

---

## Troubleshooting

### Pattern Not Detected

**Problem:** Colored lines don't appear on pattern

**Solutions:**
1. Ensure pattern is flat and not warped
2. Try different lighting (IR works better in low light)
3. For IR: Enable `ir_auto_exposure:=true`
4. Check pattern dimensions with caliper (should be exact)
5. Clean pattern surface (no glare or reflections)

### Too Few Images Accepted

**Problem:** Calibration fails with "Not enough images"

**Solution:**
- Need 30+ images minimum, 100+ recommended
- Make sure pattern is detected before pressing SPACEBAR
- Cover all areas of the camera view
- Include various pattern orientations (tilts)

### High RMS Error

**Problem:** RMS error > 1.0 pixels

**Causes:**
- Pattern is not flat (warped or bubbles)
- Camera moved during sync image capture
- Incorrect pattern dimensions (printer scaling)

**Solution:**
- Re-print pattern and verify dimensions
- Use more rigid backing board
- Ensure tripod is stable (lock ball head before capture)

### Calibration Files Not Loaded

**Problem:** Bridge doesn't show "loaded" messages

**Check:**
```bash
export SERIAL=018436651247  # Your serial
ls -la /opt/kinect_ws/install/kinect2_bridge/share/kinect2_bridge/data/$SERIAL/
```

Should show all 4 YAML files. If missing, revisit Step 9.

### Pattern Size Different from Standard

If using a custom pattern (e.g., circle8x7x0.02):

```bash
ros2 run kinect2_calibration kinect2_calibration circle8x7x0.02 record color
ros2 run kinect2_calibration kinect2_calibration circle8x7x0.02 calibrate color
# ... repeat for ir, sync, depth
```

Pattern format: `<type><cols>x<rows>x<size>`
- `type`: chess, circle, asymcircle
- `cols`: Number of feature columns
- `rows`: Number of feature rows
- `size`: Feature spacing in meters

---

## Verification Checklist

- [ ] All 4 calibration files copied to correct serial number directory
- [ ] Bridge shows "loaded" messages on startup
- [ ] Point cloud in RViz2 has aligned colors
- [ ] No color bleeding at object edges
- [ ] Tested with hole_fill_radius for improved depth quality
- [ ] Container committed to persist calibration

---

## Performance Tips

**During calibration:**
- Use `fps_limit:=2.0` to reduce CPU load
- Close unnecessary applications
- Use `reg_method:=cpu` for stability

**After calibration:**
- Remove `fps_limit` for full 30 Hz
- Use `depth_method:=cuda` + `reg_method:=cpu` for GPU acceleration
- Enable `hole_fill_radius:=2` for cleaner depth maps

---

## Example Results

**Uncalibrated vs Calibrated:**

| Aspect | Before | After |
|--------|--------|-------|
| Color-depth alignment | Offset by 5-10 pixels | < 1 pixel error |
| Edge artifacts | Visible "ghosting" | Clean edges |
| Depth accuracy | ±24mm systematic error | < 2mm error |
| Point cloud quality | Blurry color edges | Sharp, accurate |

---

## Next Steps

After successful calibration:

1. **Integrate with your robot**: Use calibrated point clouds for perception
2. **Test vision pipeline**: Run red line detection with clean depth data
3. **Optimize settings**: Experiment with hole_fill_radius and GPU acceleration
4. **Document serial number**: Note which Kinect corresponds to which robot

---

**Calibration complete!** Your Kinect v2 is now ready for production use. 🎉

For questions or issues, refer to the [kinect2_ros2 repository](https://github.com/krepa098/kinect2_ros2) or check the troubleshooting section above.
