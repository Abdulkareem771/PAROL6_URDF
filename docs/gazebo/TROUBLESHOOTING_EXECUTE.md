# Gazebo + MoveIt Execute Troubleshooting

This guide fixes the case where planning works in RViz but execution does not move the robot in Gazebo.

---

## ✅ Correct, Minimal Fix

### Symptoms
- RViz shows a planned trajectory
- Gazebo robot does not move on Execute
- TF warnings like `TF_OLD_DATA` or "jump back in time"

### Fix (Do Exactly in This Order)

1. **Restart the container (clean state):**
   ```bash
   docker restart parol6_dev
   ```

2. **Launch Gazebo first (Terminal 1):**
   ```bash
   docker exec -it parol6_dev bash
   cd /workspace && source install/setup.bash
   ros2 launch parol6 ignition.launch.py
   ```

3. **Launch MoveIt second (Terminal 2):**
   ```bash
   docker exec -it parol6_dev bash
   cd /workspace && source install/setup.bash
   ros2 launch parol6_moveit_config demo.launch.py
   ```

4. **Enable sim time for MoveIt and RViz:**
   ```bash
   docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 param set /move_group use_sim_time true"
   docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 param set /rviz2 use_sim_time true"
   ```

5. **Verify `/clock` exists:**
   ```bash
   docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 topic list | grep /clock"
   ```

6. **Verify controllers are active:**
   ```bash
   docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 control list_controllers"
   ```
   Expected:
   ```
   joint_state_broadcaster  ...  active
   parol6_arm_controller    ...  active
   ```

7. **Test execution:**
   In RViz: Plan → Execute.

   If still no motion, check if a trajectory is being published:
   ```bash
   docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 topic echo /parol6_arm_controller/joint_trajectory --once"
   ```

---

## Root Cause (Most Common)

Multiple ROS instances or time desync caused TF to jump backwards. Restarting the container and enabling sim time fixes it.
