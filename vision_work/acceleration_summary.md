# Kinect v2 Hardware Acceleration Setup

## 1. Objective
The goal was to enable **NVIDIA GPU acceleration** for the Kinect v2 driver (`libfreenect2` and `kinect2_ros2`) running inside a Docker container. This moves the heavy depth registration processing from the CPU to the GPU.

## 2. Key Challenges & Fixes

### A. Missing OpenCL Driver
*   **Issue**: `kinect2_bridge` initially failed to find any OpenCL devices.
*   **Fix**: Created `/etc/OpenCL/vendors/nvidia.icd` containing `libnvidia-opencl.so.1`. This allows the Installable Client Driver (ICD) loader to find the NVIDIA runtime.

### B. "Initialization Failed" (Missing Kernel)
*   **Issue**: OpenCL was initializing but failing silently or with generic errors.
*   **Root Cause**: The build script (`CMakeLists.txt`) was not defining `REG_OPENCL_FILE`, so the code couldn't find the `.cl` kernel source file at runtime.
*   **Fix**: Patched `kinect2_registration/CMakeLists.txt` to define `-DREG_OPENCL_FILE` pointing to the absolute path of `depth_registration.cl`.

### C. "OpenCL registration not available" (Build Regression)
*   **Issue**: After a clean rebuild, the bridge node refused to use OpenCL, claiming it wasn't available.
*   **Root Cause**: `kinect2_bridge` was compiled without the `-DDEPTH_REG_OPENCL` flag, causing the preprocessor to strip out the OpenCL logic.
*   **Fix**:
    1.  Updated `kinect2_bridge/CMakeLists.txt` to explicitly add `-DDEPTH_REG_OPENCL` and `-DDEPTH_REG_CPU` to the compiler definitions.
    2.  Modified `kinect2_registration.cpp` to remove `#ifdef` guards, forcing the compiler to include the OpenCL code.

### D. OpenGL Failure
*   **Issue**: `depth_method:=opengl` fails with `failed to load driver: nouveau`.
*   **Root Cause**: The Docker container lacks the NVIDIA **Graphics** libraries (`libGLX_nvidia.so`), containing only the **Compute** libraries (OpenCL/CUDA). This is a limitation of the current container environment.

## 3. Validated Modes

| Depth Method | Reg Method | Status | Performance | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **`opencl`** | **`opencl`** | ✅ **PASS** | **~600 Hz** | **Recommended.** Full GPU acceleration. Very fast. |
| `opencl` | `cpu` | ✅ **PASS** | ~3 Hz | Working CPU fallback. Useful for debugging only. |
| `cpu` | `cpu` | ✅ **PASS** | < 1 Hz | extremely slow. Baseline. |
| `opengl` | `*` | ❌ **FAIL** | N/A | Requires host GLX driver injection. |

## 4. Usage Commands

### Standard Execution (Fastest)
Run this command to start the driver with full acceleration:
```bash
docker exec -it parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /opt/kinect_ws/install/setup.bash && ros2 launch kinect2_bridge kinect2_bridge_launch.yaml depth_method:=opencl reg_method:=opencl"
```

### Debugging (CPU Fallback)
```bash
docker exec -it parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /opt/kinect_ws/install/setup.bash && ros2 launch kinect2_bridge kinect2_bridge_launch.yaml depth_method:=opencl reg_method:=cpu"
```
