# PAROL6 Demos Package

Demo applications for the PAROL6 vision-guided welding/gluing system.

## Phase 1: Cartesian Path Validation

**Purpose**: Validate smooth Cartesian path following before adding vision complexity.

**Demo**: `demo_cartesian_path.py`

### What it Tests:
- ✅ Straight-line motion (welding seam simulation)
- ✅ L-shaped path (corner transitions)
- ✅ Rectangular path (multi-corner, path closure)
- ✅ Smooth acceleration profiles
- ✅ mkservo42c closed-loop operation

### Running the Demo:

**Option 1: Real Robot**
```bash
# Terminal 1: Start system
./bringup.sh real

# Terminal 2: Run demo
docker exec -it parol6_dev bash
source /workspace/install/setup.bash
ros2 run parol6_demos demo_cartesian_path
```

**Option 2: Simulation**
```bash
# Terminal 1: Start Gazebo
./start_ignition.sh

# Terminal 2: Add MoveIt
./add_moveit.sh

#Terminal 3: Run demo
docker exec -it parol6_dev bash
source /workspace/install/setup.bash
ros2 run parol6_demos demo_cartesian_path
```

### Expected Behavior:
1. Robot moves to HOME position
2. Executes straight line (10cm forward)
3. Returns to HOME
4. Executes L-shape (forward 8cm, right 6cm)
5. Returns to HOME
6. Executes rectangle (8cm x 6cm)
7. Returns to HOME
8. Prints summary

### Success Criteria:
- ✅ All paths planned successfully (>90% completion)
- ✅ Smooth motion (no jerks at waypoints)
- ✅ Accurate positioning (visual verification)
- ✅ No servo step loss (closed-loop verification)

### Next Steps:
Once this passes → Proceed to Phase 2 (Vision Integration)

---

## Future Demos

**Phase 2**: `demo_vision_detection.py` - Test workpiece detection
**Phase 3**: `demo_path_generation.py` - Test B-spline smoothing
**Phase 4**: `demo_vision_guided.py` - Complete integrated system
