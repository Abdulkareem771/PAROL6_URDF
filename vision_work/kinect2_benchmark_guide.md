# Kinect v2 Performance Benchmark Guide

## Test Setup

**Hardware:**
- NVIDIA RTX 3050 Ti Laptop GPU
- Kinect v2 sensor (serial: 018436651247)
- Docker container: parol6-ultimate:latest

**Test Duration:** 60 seconds per configuration

---

## Test Configurations

### Test 1: CUDA Depth + CPU Registration (Default)
**Configuration:**
```yaml
depth_method: cuda
reg_method: cpu
```

**Why test:** This is our recommended production config

---

### Test 2: CPU-only (Fallback)
**Configuration:**
```yaml
depth_method: cpu
reg_method: cpu
```

**Why test:** Baseline for systems without GPU

---

## How to Run Tests

### Terminal Setup (3 terminals needed)

**Terminal 1: Bridge** (run each test config)
**Terminal 2: FPS Monitor**
**Terminal 3: Resource Monitor**

---

## Test 1: CUDA + CPU (Current Default)

### Terminal 1 - Launch Bridge
```bash
docker exec -it parol6_dev bash
source /opt/kinect_ws/install/setup.bash
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml
```

### Terminal 2 - Monitor FPS
```bash
docker exec -it parol6_dev bash
source /opt/kinect_ws/install/setup.bash

# Monitor depth topic
echo "=== QHD Depth FPS ==="
ros2 topic hz /kinect2/qhd/image_depth_rect

# In parallel, check color FPS (Ctrl+C after 60s, then run this)
echo "=== QHD Color FPS ==="
ros2 topic hz /kinect2/qhd/image_color_rect
```

### Terminal 3 - Monitor Resources
```bash
docker exec -it parol6_dev bash

# GPU monitoring
watch -n 1 nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu --format=csv

# OR combined CPU+GPU (if you have both)
htop  # Watch kinect2_bridge_node process CPU%
```

**Record for 60 seconds, then note:**
- Average FPS (depth)
- Average FPS (color)
- GPU utilization %
- GPU memory used
- CPU % (from htop)

---

## Test 2: CPU-only

### Terminal 1 - Launch Bridge (CPU mode)
```bash
# Stop previous bridge (Ctrl+C), then:
docker exec -it parol6_dev bash
source /opt/kinect_ws/install/setup.bash
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml \
  depth_method:=cpu \
  reg_method:=cpu
```

### Terminal 2 & 3 - Same monitoring as Test 1

**Record for 60 seconds, then note:**
- Average FPS (depth)
- Average FPS (color)
- GPU utilization % (should be 0%)
- CPU % (will be higher)

---

## Data Collection Template

### Test 1: CUDA + CPU
```
Depth FPS: _____
Color FPS: _____
GPU Util: _____%
GPU Mem: _____ MB
CPU %: _____%
Temperature: _____°C
```

### Test 2: CPU + CPU
```
Depth FPS: _____
Color FPS: _____
GPU Util: _____%
GPU Mem: _____ MB
CPU %: _____%
Temperature: _____°C
```

---

## Expected Results

| Config | Depth FPS | GPU Util | CPU % | Status |
|--------|-----------|----------|-------|--------|
| CUDA+CPU | ~30 Hz | ~15-30% | ~20% | ✅ Optimal |
| CPU+CPU | ~5-15 Hz | 0% | ~60-80% | ⚠️ Slow |

---

## Quick Test Commands (All-in-One)

**Test CUDA+CPU:**
```bash
# Terminal 1
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml &
sleep 5
# Terminal 2 (auto-collect for 30 seconds)
timeout 30 ros2 topic hz /kinect2/qhd/image_depth_rect
```

**Test CPU+CPU:**
```bash
# Terminal 1 (stop previous, start CPU mode)
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml depth_method:=cpu reg_method:=cpu &
sleep 5
# Terminal 2
timeout 30 ros2 topic hz /kinect2/qhd/image_depth_rect
```
