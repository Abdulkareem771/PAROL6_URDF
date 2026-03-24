---
trigger: always_on
glob: "*"
description: "General rules for the PAROL6 ROS2/MoveIt2 codebase"
---

1. **ROS2 Humble Standards**: Always follow ROS2 Humble conventions for Python and C++ nodes. Use modern `rclpy`/`rclcpp` APIs and appropriate logging levels (`get_logger().info()`, etc.).
2. **Safety & Motion Planning (MoveIt2)**:
   - Always prioritize collision-free Cartesian trajectory generation.
   - Implement strict fail-safe stop conditions for TF timeouts, kinematic singularities, and joint limits.
3. **Performance & Hardware Optimization**:
   - The system utilizes an Intel Xeon processor and a Quadro GPU. Always prioritize GPU-accelerated computing (CUDA/OpenCL) and OpenMP parallelization for vision pipelines and path optimization where possible.
4. **Containerized Environment**:
   - Assume development happens inside the Docker environment (`parol6_dev`). Do not introduce dependencies that break the Docker setup unless explicitly asked.
5. **Code Modularity and Clarity**:
   - Ensure nodes are modular, well-documented, and error-handled gracefully to prevent silent crashes in the robotic pipeline.
