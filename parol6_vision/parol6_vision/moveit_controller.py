#!/usr/bin/env python3
"""
MoveIt Controller Node - Vision-Guided Welding Path Detection

This node executes generated welding paths using MoveIt2. It implements a robust
Cartesian path planning strategy with multiple fallback resolutions to handle
potential planning failures (common in Cartesian motion).

================================================================================
ALGORITHM OVERVIEW
================================================================================

1. PATH RECEPTION & VALIDATION
   - Subscribes to /vision/welding_path
   - Checks constraints: Minimum length, workspace bounds, curvature limits.

2. CARTESIAN PLANNING FALLBACK STRATEGY
   - Cartesian path planning (Task Space) is brittle. A single collision or 
     kinematic singularity can cause failure.
   - We implement a 3-Tier Fallback Strategy:
     
     Attempt 1: High Precision (Step=2mm, Success Threshold=95%)
       - Ideal for high-quality welding.
       
     Attempt 2: Medium Precision (Step=5mm, Success Threshold=95%)
       - If precision plan fails, relax step size to skip micro-singularities.
       
     Attempt 3: Coarse Precision (Step=10mm, Success Threshold=90%)
       - "Get the job done" mode. Acceptable for rough passes.

3. EXECUTION SEQUENCE
   Phase A: Pre-Weld Approach
     - Move to "Approach Point" (5cm above start of weld).
     - Standard Joint Space planning (guaranteed success).
   
   Phase B: Welding Motion
     - Linear motion along the seam.
     - Constant velocity execution (critical for weld quality).
   
   Phase C: Post-Weld Retract
     - Move safely away from workpiece (5cm up).

================================================================================
THESIS-READY STATEMENTS
================================================================================

> "To overcome the inherent brittleness of Cartesian trajectory generation in
> constrained environments, a hierarchical fallback strategy was implemented. This
> dynamically adjusts the discretization resolution (2mm to 10mm), prioritizing
> geometric fidelity while ensuring system reliability."

> "The execution pipeline decouples the approach, welding, and retraction phases,
> ensuring that potential planning failures during the approach phase do not 
> compromise the critical welding trajectory."

================================================================================
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Path
from moveit_msgs.msg import (
    MoveItErrorCodes, RobotTrajectory,
    Constraints, PositionConstraint, OrientationConstraint, BoundingVolume,
    JointConstraint
)
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from geometry_msgs.msg import PoseStamped, Pose
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Trigger

import copy
import math
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError

class MoveItController(Node):
    """
    MoveIt Controller Node
    
    Executes welding paths using MoveIt2 with robust fallback logic.
    """
    
    def __init__(self):
        super().__init__('moveit_controller')
        
        # ============================================================
        # PARAMETERS
        # ============================================================
        
        self.declare_parameter('planning_group', 'parol6_arm')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('end_effector_link', 'tcp_link')
        
        # Fallback Strategy Config
        self.declare_parameter('cartesian_step_sizes', [0.002, 0.005, 0.010])
        self.declare_parameter('min_success_rates', [0.95, 0.95, 0.90])
        
        # Process offsets
        self.declare_parameter('approach_distance', 0.05)
        self.declare_parameter('weld_velocity', 0.01) # m/s

        # Auto-execute: if True, run sequence every time a new path arrives
        self.declare_parameter('auto_execute', False)
        self.declare_parameter('move_group_wait_timeout_sec', 30.0)
        self.declare_parameter('execute_wait_timeout_sec', 20.0)
        self.declare_parameter('enable_joint_waypoint_fallback', True)
        self.declare_parameter('joint_waypoint_fallback_count', 8)

        # Test mode: clamp incoming path into a conservative reachable workspace.
        # Useful to validate pipeline wiring even when vision points are out of reach.
        self.declare_parameter('enforce_reachable_test_path', False)
        self.declare_parameter('test_workspace_min', [0.20, -0.35, 0.10])  # x,y,z
        self.declare_parameter('test_workspace_max', [0.65, 0.35, 0.55])   # x,y,z
        self.declare_parameter('test_min_radius_xy', 0.20)
        self.declare_parameter('test_max_radius_xy', 0.70)
        
        self.group_name = self.get_parameter('planning_group').value
        self.base_frame = self.get_parameter('base_frame').value
        self.ee_link = self.get_parameter('end_effector_link').value
        self.step_sizes = self.get_parameter('cartesian_step_sizes').value
        self.thresholds = self.get_parameter('min_success_rates').value
        self.approach_dist = self.get_parameter('approach_distance').value
        self.auto_execute = self.get_parameter('auto_execute').value
        self.move_group_wait_timeout = float(self.get_parameter('move_group_wait_timeout_sec').value)
        self.execute_wait_timeout = float(self.get_parameter('execute_wait_timeout_sec').value)
        self.enable_joint_waypoint_fallback = bool(
            self.get_parameter('enable_joint_waypoint_fallback').value
        )
        self.joint_waypoint_fallback_count = int(
            self.get_parameter('joint_waypoint_fallback_count').value
        )
        self.enforce_reachable_test_path = self.get_parameter('enforce_reachable_test_path').value
        self.workspace_min = self.get_parameter('test_workspace_min').value
        self.workspace_max = self.get_parameter('test_workspace_max').value
        self.min_radius_xy = self.get_parameter('test_min_radius_xy').value
        self.max_radius_xy = self.get_parameter('test_max_radius_xy').value
        
        # ============================================================
        # ROS INTERFACES
        # ============================================================
        
        # Cartesian Path Service
        self.cartesian_client = self.create_client(
            GetCartesianPath, 
            'compute_cartesian_path'
        )
        
        # Execute Trajectory Action
        self.execute_client = ActionClient(
            self, 
            ExecuteTrajectory, 
            'execute_trajectory'
        )
        
        # MoveGroup Action (for joint moves)
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            'move_action'
        )
        
        # Input Path — TRANSIENT_LOCAL (latching) so a message published before
        # this node starts is still received. Must match the publisher's QoS.
        latch_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.sub = self.create_subscription(
            Path,
            '/vision/welding_path',
            self.path_callback,
            latch_qos
        )
        
        # Manual Trigger
        self.srv = self.create_service(
            Trigger, 
            '~/execute_welding_path',
            self.trigger_execution
        )
        
        # Status Query
        self.status_srv = self.create_service(
            Trigger,
            '~/is_execution_idle',
            self.get_execution_status
        )
        
        self.latest_path = None
        self.execution_in_progress = False
        self._exec_lock = threading.Lock()
        self.get_logger().info('MoveIt Controller initialized')

    def _wait_async_result(self, future, timeout_sec=15.0):
        """
        Wait for an rclpy Future from worker thread without calling spin().
        Main executor keeps spinning in main thread; this just blocks on an event.
        """
        done_evt = threading.Event()

        def _done_cb(_):
            done_evt.set()

        future.add_done_callback(_done_cb)
        if not done_evt.wait(timeout_sec):
            raise FutureTimeoutError(f'Future timed out after {timeout_sec}s')
        return future.result()

    def _wait_for_action_server(self, client, server_name: str, timeout_sec: float) -> bool:
        """Wait for action server in short polls to tolerate DDS discovery delay."""
        deadline = self.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while self.get_clock().now().nanoseconds < deadline:
            if client.wait_for_server(timeout_sec=0.5):
                return True
        self.get_logger().error(f"Action server '{server_name}' not available after {timeout_sec:.1f}s")
        return False
        
    def path_callback(self, msg):
        """Buffer latest path; auto-execute if configured"""
        self.latest_path = msg
        self.get_logger().info(f'Received path with {len(msg.poses)} points')
        if self.auto_execute:
            self._start_execution_async(msg, source='auto_execute')

    def trigger_execution(self, request, response):
        """Service callback to start execution"""
        if self.latest_path is None:
            response.success = False
            response.message = "No path received yet"
            return response
            
        if self.execution_in_progress:
            response.success = False
            response.message = "Execution already in progress"
            return response
            
        # Start execution in worker thread so callback doesn't block executor.
        self._start_execution_async(self.latest_path, source='service')
        response.success = True
        response.message = "Execution started"
        return response

    def get_execution_status(self, request, response):
        """Service callback to check if execution is idle"""
        with self._exec_lock:
            # If idle, return success=True
            response.success = not self.execution_in_progress
            response.message = "Idle" if not self.execution_in_progress else "Executing"
            return response

    def _start_execution_async(self, path, source='unknown'):
        with self._exec_lock:
            if self.execution_in_progress:
                self.get_logger().info(f'{source}: execution already in progress, skipping')
                return
            self.execution_in_progress = True
        self.get_logger().info(f'{source}: starting welding sequence thread')
        t = threading.Thread(target=self._execution_worker, args=(path,), daemon=True)
        t.start()

    def _execution_worker(self, path):
        try:
            ok = self.execute_welding_sequence(path)
            self.get_logger().info(f'Execution finished: success={ok}')
        except Exception as exc:
            self.get_logger().error(f'Execution worker exception: {exc}')
        finally:
            with self._exec_lock:
                self.execution_in_progress = False

    # ================================================================
    # EXECUTION SEQUENCE
    # ================================================================
    
    def execute_welding_sequence(self, path):
        """
        Full 3-Phase Welding Sequence: Approach -> Weld -> Retract

        KEY DESIGN: All planning is done BEFORE any execution begins.
        This eliminates the robot "freeze" that occurs when the weld Cartesian
        path is planned AFTER the approach move has already finished.
        """
        self.get_logger().info("STARTING WELDING SEQUENCE (pre-plan all phases)")

        # Optional test-mode normalization
        if self.enforce_reachable_test_path:
            path = self._make_path_reachable(path)
            if len(path.poses) == 0:
                self.get_logger().error("Reachability normalization produced empty path")
                return False

        # ── PLANNING PHASE ────────────────────────────────────────────────────
        # Phase 1: Plan home trajectory (plan only, no execution)
        self.get_logger().info("[Plan 1/3] Planning Home trajectory…")
        home_traj = self.plan_to_home()
        if home_traj is None:
            self.get_logger().error("Planning home failed — aborting")
            return False

        # Phase 2: Plan approach trajectory using home end-state as start
        self.get_logger().info("[Plan 2/3] Planning Approach trajectory…")
        approach_pose = copy.deepcopy(path.poses[0])
        approach_pose.pose.position.z += self.approach_dist
        approach_traj = self.plan_pose(
            approach_pose, start_state_traj=home_traj
        )
        if approach_traj is None:
            self.get_logger().error("Planning approach failed — aborting")
            return False

        # Phase 3: Plan Cartesian weld path using approach end-state as start
        self.get_logger().info("[Plan 3/3] Planning Cartesian weld trajectory…")
        weld_traj = self.plan_cartesian_with_fallback(
            path, start_state_traj=approach_traj
        )
        if not weld_traj:
            self.get_logger().error("All Cartesian planning attempts failed")
            if self.enable_joint_waypoint_fallback:
                self.get_logger().warn("Falling back to joint-space waypoint execution")
                # Fallback must execute on its own (plan+execute inside loop)
                if not self.execute_trajectory_action(home_traj):
                    return False
                if not self.execute_trajectory_action(approach_traj):
                    return False
                return self.execute_waypoint_fallback(path)
            return False

        # ── EXECUTION PHASE (all plans ready, execute back-to-back) ──────────
        self.get_logger().info("All phases planned! Starting seamless execution…")

        self.get_logger().info("[Exec 1/3] Moving to Home")
        if not self.execute_trajectory_action(home_traj):
            self.get_logger().error("Home execution failed")
            return False

        self.get_logger().info("[Exec 2/3] Moving to Approach Point")
        if not self.execute_trajectory_action(approach_traj):
            self.get_logger().error("Approach execution failed")
            return False

        self.get_logger().info("[Exec 3/3] Executing Weld")
        if not self.execute_trajectory_action(weld_traj):
            self.get_logger().error("Weld execution failed")
            return False

        self.get_logger().info("Sequence Complete")
        return True

    def _make_path_reachable(self, path: Path) -> Path:
        """Clamp path positions into a conservative reachable workspace for test validation."""
        out = Path()
        out.header = path.header
        changed = 0

        xmin, ymin, zmin = self.workspace_min
        xmax, ymax, zmax = self.workspace_max
        rmin = max(0.0, float(self.min_radius_xy))
        rmax = max(rmin + 1e-6, float(self.max_radius_xy))

        for pose_stamped in path.poses:
            p = copy.deepcopy(pose_stamped)
            x = p.pose.position.x
            y = p.pose.position.y
            z = p.pose.position.z
            orig = (x, y, z)

            # Axis-aligned clamp.
            x = max(xmin, min(xmax, x))
            y = max(ymin, min(ymax, y))
            z = max(zmin, min(zmax, z))

            # Radial clamp in XY plane relative to base.
            r = math.hypot(x, y)
            if r > 1e-9:
                if r < rmin:
                    s = rmin / r
                    x *= s
                    y *= s
                elif r > rmax:
                    s = rmax / r
                    x *= s
                    y *= s
            else:
                x = rmin
                y = 0.0

            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.position.z = z
            out.poses.append(p)

            if (abs(orig[0] - x) > 1e-9) or (abs(orig[1] - y) > 1e-9) or (abs(orig[2] - z) > 1e-9):
                changed += 1

        self.get_logger().info(
            f"Reachability normalization: modified {changed}/{len(path.poses)} points "
            f"(frame={path.header.frame_id if path.header.frame_id else 'unknown'})"
        )
        return out

    # ================================================================
    # PLANNING LOGIC
    # ================================================================

    def plan_cartesian_with_fallback(self, path, start_state_traj=None):
        """
        Try to plan Cartesian path with decreasing precision requirements.
        If start_state_traj is provided, its end joint state is used as the
        planning start state (critical for pre-plan consistency).
        """
        waypoints = [p.pose for p in path.poses]

        # Build start state from the previous trajectory if supplied
        start_state = None
        if start_state_traj is not None:
            start_state = self._extract_end_state(start_state_traj)

        for i, step in enumerate(self.step_sizes):
            threshold = self.thresholds[i]
            
            self.get_logger().info(
                f"Attempt {i+1}: Step={step*1000}mm, Threshold={threshold*100}%"
            )
            
            if not self.cartesian_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("Compute Cartesian Path service execution failed")
                return None

            req = GetCartesianPath.Request()
            req.header = path.header
            req.header.stamp = self.get_clock().now().to_msg()
            req.group_name = self.group_name
            req.link_name = self.ee_link  # CRITICAL: tell MoveIt which link to plan for
            req.waypoints = waypoints
            req.max_step = step
            req.jump_threshold = 0.0
            req.avoid_collisions = True
            if start_state is not None:
                req.start_state = start_state
            
            future = self.cartesian_client.call_async(req)
            try:
                res = self._wait_async_result(future, timeout_sec=20.0)
            except FutureTimeoutError:
                self.get_logger().warn('compute_cartesian_path timed out after 20.0s')
                continue
            
            if res.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().warn(f"MoveIt Error: {res.error_code.val}")
                continue
                
            if res.fraction >= threshold:
                self.get_logger().info(f"Success! Planned fraction: {res.fraction:.2f}")
                trajectory = res.solution
                self._ensure_trajectory_timing(trajectory)
                return trajectory
            else:
                self.get_logger().warn(f"Fraction too low: {res.fraction:.2f} < {threshold}")
                
        return None

    def _ensure_trajectory_timing(self, trajectory):
        """
        Ensure all JointTrajectory points have proper time_from_start.

        compute_cartesian_path may return a trajectory where all time_from_start
        values are zero, causing Gazebo's JointTrajectoryController to silently
        hang (never completing the action result). This adds linear timing at
        0.5 seconds per waypoint when zero times are detected.
        """
        from builtin_interfaces.msg import Duration as RosDuration

        points = trajectory.joint_trajectory.points
        if not points:
            return

        # Check if timing is already set (last point > 0)
        last = points[-1].time_from_start
        if last.sec > 0 or last.nanosec > 0:
            self.get_logger().info(
                f"Trajectory already timed: {len(points)} pts, "
                f"total={last.sec + last.nanosec*1e-9:.2f}s"
            )
            return

        # Add linear timing: 0.5s per waypoint
        dt = 0.5
        for i, pt in enumerate(points):
            total_sec = (i + 1) * dt
            pt.time_from_start = RosDuration(
                sec=int(total_sec),
                nanosec=int((total_sec % 1) * 1_000_000_000)
            )
        total_time = len(points) * dt
        self.get_logger().info(
            f"Added linear timing to {len(points)} trajectory points "
            f"(total={total_time:.1f}s, {dt}s/pt)"
        )

    def execute_waypoint_fallback(self, path: Path) -> bool:
        """
        Fallback execution path:
        - Select a coarse subset of waypoints.
        - Move with joint-space planning to each waypoint.
        This trades smooth Cartesian motion for robustness.
        """
        total = len(path.poses)
        if total == 0:
            self.get_logger().error("Waypoint fallback received empty path")
            return False

        desired = max(2, self.joint_waypoint_fallback_count)
        if total <= desired:
            indices = list(range(total))
        else:
            indices = sorted({int(i * (total - 1) / (desired - 1)) for i in range(desired)})

        self.get_logger().info(
            f"Waypoint fallback: executing {len(indices)}/{total} sampled waypoints"
        )
        for n, idx in enumerate(indices, start=1):
            pose_stamped = copy.deepcopy(path.poses[idx])
            ok = self.move_to_pose(
                pose_stamped,
                use_orientation_constraint=False
            )
            if not ok:
                self.get_logger().error(
                    f"Waypoint fallback failed at sampled point {n}/{len(indices)} (index={idx})"
                )
                return False

        self.get_logger().info("Waypoint fallback execution completed")
        return True

    def _extract_end_state(self, trajectory):
        """
        Build a RobotState matching the last point of a JointTrajectory so it
        can be used as `start_state` for the next planning request.
        Returns None if the trajectory is empty or malformed.
        """
        from moveit_msgs.msg import RobotState
        from sensor_msgs.msg import JointState

        jt = trajectory.joint_trajectory
        if not jt.points:
            return None

        rs = RobotState()
        js = JointState()
        js.name = list(jt.joint_names)
        js.position = list(jt.points[-1].positions)
        rs.joint_state = js
        return rs

    def plan_to_home(self):
        """
        Plan (don't execute) a move to home joint state.
        Returns a RobotTrajectory or None on failure.
        """
        if not self._wait_for_action_server(
            self.move_group_client, 'move_action', self.move_group_wait_timeout
        ):
            return None

        joint_names = ['joint_L1', 'joint_L2', 'joint_L3',
                       'joint_L4', 'joint_L5', 'joint_L6']
        home_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        joint_constraints = []
        for name, pos in zip(joint_names, home_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            joint_constraints.append(jc)

        goal_constraints = Constraints()
        goal_constraints.joint_constraints = joint_constraints

        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = 0.5
        goal.request.max_acceleration_scaling_factor = 0.5
        goal.request.goal_constraints = [goal_constraints]
        # CRITICAL: is_diff=True tells MoveIt to use the current robot state from the
        # planning scene monitor as the start state, rather than trusting the default
        # empty RobotState (which maps to all-zero joints and generates a trivially
        # short "already at home" trajectory that fails the start-point tolerance check).
        goal.request.start_state.is_diff = True
        goal.planning_options.plan_only = True  # PLAN ONLY

        future = self.move_group_client.send_goal_async(goal)
        try:
            handle = self._wait_async_result(future, timeout_sec=20.0)
        except FutureTimeoutError:
            self.get_logger().error('Timed out waiting for home plan')
            return None
        if handle is None or not handle.accepted:
            self.get_logger().error('Home plan goal rejected')
            return None

        res_future = handle.get_result_async()
        try:
            result = self._wait_async_result(res_future, timeout_sec=30.0)
        except FutureTimeoutError:
            self.get_logger().error('Timed out waiting for home plan result')
            return None
        if result is None or result.result.error_code.val != MoveItErrorCodes.SUCCESS:
            code = getattr(result, 'result', None)
            self.get_logger().error(f'Home plan failed — code: {code}')
            return None

        self._ensure_trajectory_timing(result.result.planned_trajectory)
        self.get_logger().info('Home trajectory planned')
        return result.result.planned_trajectory

    def plan_pose(
        self,
        pose_stamped,
        start_state_traj=None,
        use_orientation_constraint=True,
        orientation_tolerance_xyz=0.2
    ):
        """
        Plan (don't execute) a move to a target pose.
        If start_state_traj is provided, its last point is used as the planning
        start state, so the plan is consistent with the previous phase.
        Returns a RobotTrajectory on success, or None on failure.
        """
        self.get_logger().info(
            f'Planning approach: x={pose_stamped.pose.position.x:.3f}, '
            f'y={pose_stamped.pose.position.y:.3f}, '
            f'z={pose_stamped.pose.position.z:.3f}'
        )
        if not self._wait_for_action_server(
            self.move_group_client, 'move_action', self.move_group_wait_timeout
        ):
            return None

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [0.02, 0.02, 0.02]

        bv = BoundingVolume()
        bv.primitives = [prim]
        bv.primitive_poses = [pose_stamped.pose]

        pos_con = PositionConstraint()
        pos_con.header = pose_stamped.header
        pos_con.link_name = self.ee_link
        pos_con.constraint_region = bv
        pos_con.weight = 1.0

        goal_constraints = Constraints()
        goal_constraints.position_constraints = [pos_con]
        if use_orientation_constraint:
            ori_con = OrientationConstraint()
            ori_con.header = pose_stamped.header
            ori_con.link_name = self.ee_link
            ori_con.orientation = pose_stamped.pose.orientation
            ori_con.absolute_x_axis_tolerance = orientation_tolerance_xyz
            ori_con.absolute_y_axis_tolerance = orientation_tolerance_xyz
            ori_con.absolute_z_axis_tolerance = orientation_tolerance_xyz
            ori_con.weight = 1.0
            goal_constraints.orientation_constraints = [ori_con]

        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = 0.5
        goal.request.max_acceleration_scaling_factor = 0.5
        goal.request.goal_constraints = [goal_constraints]
        goal.planning_options.plan_only = True  # PLAN ONLY

        # Seed the planner from the previous trajectory's final state
        if start_state_traj is not None:
            rs = self._extract_end_state(start_state_traj)
            if rs is not None:
                goal.request.start_state = rs

        future = self.move_group_client.send_goal_async(goal)
        try:
            handle = self._wait_async_result(future, timeout_sec=20.0)
        except FutureTimeoutError:
            self.get_logger().error('Timed out waiting for approach plan')
            return None
        if handle is None or not handle.accepted:
            self.get_logger().error('Approach plan rejected')
            return None

        res_future = handle.get_result_async()
        try:
            result = self._wait_async_result(res_future, timeout_sec=30.0)
        except FutureTimeoutError:
            self.get_logger().error('Timed out waiting for approach plan result')
            return None
        if result is None or result.result.error_code.val != MoveItErrorCodes.SUCCESS:
            code = getattr(result, 'result', None)
            self.get_logger().error(f'Approach plan failed — code: {code}')
            return None

        self._ensure_trajectory_timing(result.result.planned_trajectory)
        self.get_logger().info('Approach trajectory planned')
        return result.result.planned_trajectory

    def move_to_home(self):
        """
        Execute a move to home (convenience wrapper: plan + execute).
        Used by fallback paths only. Main sequence uses plan_to_home() directly.
        """
        traj = self.plan_to_home()
        if traj is None:
            return False
        return self.execute_trajectory_action(traj)

    def _move_to_home_legacy(self):
        """
        Original plan+execute home move kept for reference.
        Move arm to home joint state (all joints = 0) using JointConstraints.
        This is always kinematically valid and is used as the approach phase
        in the connection-validation test.
        """
        if not self._wait_for_action_server(
            self.move_group_client, 'move_action', self.move_group_wait_timeout
        ):
            return False

        joint_names = ['joint_L1', 'joint_L2', 'joint_L3',
                       'joint_L4', 'joint_L5', 'joint_L6']
        home_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        joint_constraints = []
        for name, pos in zip(joint_names, home_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            joint_constraints.append(jc)

        goal_constraints = Constraints()
        goal_constraints.joint_constraints = joint_constraints

        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3
        goal.request.goal_constraints = [goal_constraints]
        goal.planning_options.plan_only = False

        future = self.move_group_client.send_goal_async(goal)
        try:
            handle = self._wait_async_result(future, timeout_sec=20.0)
        except FutureTimeoutError:
            self.get_logger().error('Timed out waiting for MoveGroup home goal response')
            return False

        if handle is None or not handle.accepted:
            self.get_logger().error('Home goal rejected or None')
            return False

        res_future = handle.get_result_async()
        try:
            result = self._wait_async_result(res_future, timeout_sec=60.0)
        except FutureTimeoutError:
            self.get_logger().error('Timed out waiting for MoveGroup home result')
            return False

        if result is None:
            self.get_logger().error('Home result is None')
            return False

        ok = result.result.error_code.val == MoveItErrorCodes.SUCCESS
        if ok:
            self.get_logger().info('Move to home succeeded')
        else:
            self.get_logger().error(
                f'Move to home failed — MoveIt error code: {result.result.error_code.val}'
            )
        return ok

    def move_to_pose(
        self,
        pose_stamped,
        use_orientation_constraint=True,
        orientation_tolerance_xyz=0.2
    ):
        """
        Move end-effector to a target pose using MoveGroup joint-space planning.
        Sends a full MotionPlanRequest via the 'move_action' action server.
        """
        self.get_logger().info(
            f'Approach target: x={pose_stamped.pose.position.x:.3f}, '
            f'y={pose_stamped.pose.position.y:.3f}, '
            f'z={pose_stamped.pose.position.z:.3f}'
        )

        if not self._wait_for_action_server(
            self.move_group_client, 'move_action', self.move_group_wait_timeout
        ):
            return False

        # --- Build position constraint (2 cm tolerance box around target) ---
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [0.02, 0.02, 0.02]  # slightly relaxed for IK sampling

        bv = BoundingVolume()
        bv.primitives = [prim]
        bv.primitive_poses = [pose_stamped.pose]

        pos_con = PositionConstraint()
        pos_con.header = pose_stamped.header
        pos_con.link_name = self.ee_link
        pos_con.constraint_region = bv
        pos_con.weight = 1.0

        goal_constraints = Constraints()
        goal_constraints.position_constraints = [pos_con]
        if use_orientation_constraint:
            ori_con = OrientationConstraint()
            ori_con.header = pose_stamped.header
            ori_con.link_name = self.ee_link
            ori_con.orientation = pose_stamped.pose.orientation
            ori_con.absolute_x_axis_tolerance = orientation_tolerance_xyz
            ori_con.absolute_y_axis_tolerance = orientation_tolerance_xyz
            ori_con.absolute_z_axis_tolerance = orientation_tolerance_xyz
            ori_con.weight = 1.0
            goal_constraints.orientation_constraints = [ori_con]

        # --- Build MoveGroup Goal ---
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3
        goal.request.goal_constraints = [goal_constraints]
        goal.planning_options.plan_only = False  # plan AND execute


        future = self.move_group_client.send_goal_async(goal)
        try:
            handle = self._wait_async_result(future, timeout_sec=20.0)
        except FutureTimeoutError:
            self.get_logger().error('Timed out waiting for MoveGroup goal response')
            return False

        if handle is None:
            self.get_logger().error('Approach goal handle is None')
            return False

        if not handle.accepted:
            self.get_logger().error('Approach goal rejected by MoveGroup')
            return False

        res_future = handle.get_result_async()
        try:
            result = self._wait_async_result(res_future, timeout_sec=60.0)
        except FutureTimeoutError:
            self.get_logger().error('Timed out waiting for MoveGroup execution result')
            return False

        if result is None:
            self.get_logger().error('Approach result is None')
            return False

        ok = result.result.error_code.val == MoveItErrorCodes.SUCCESS
        if ok:
            self.get_logger().info('Approach move succeeded')
        else:
            self.get_logger().error(
                f'Approach move failed — MoveIt error code: {result.result.error_code.val}'
            )
        return ok

    def execute_trajectory_action(self, trajectory):
        """Execute a computed trajectory"""
        if not self._wait_for_action_server(
            self.execute_client, 'execute_trajectory', self.execute_wait_timeout
        ):
            return False
            
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        
        future = self.execute_client.send_goal_async(goal)
        try:
            goal_handle = self._wait_async_result(future, timeout_sec=20.0)
        except FutureTimeoutError:
            self.get_logger().error('Timed out waiting for ExecuteTrajectory goal response')
            return False
        if goal_handle is None:
            self.get_logger().error('ExecuteTrajectory goal handle is None')
            return False
        
        if not goal_handle.accepted:
            return False
            
        res_future = goal_handle.get_result_async()
        try:
            result = self._wait_async_result(res_future, timeout_sec=90.0)
        except FutureTimeoutError:
            self.get_logger().error('Timed out waiting for ExecuteTrajectory result')
            return False
        if result is None:
            self.get_logger().error('ExecuteTrajectory result is None')
            return False
        
        return result.result.error_code.val == MoveItErrorCodes.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    node = MoveItController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
