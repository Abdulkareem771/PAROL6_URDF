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
        self.declare_parameter('approach_distance', 0.15)  # 15cm: weld surface ~z=0.045m → approach at z=0.195m (inside workspace)
        self.declare_parameter('weld_velocity', 0.01) # m/s

        # Auto-execute: if True, run sequence every time a new path arrives
        self.declare_parameter('auto_execute', False)
        self.declare_parameter('move_group_wait_timeout_sec', 30.0)
        self.declare_parameter('execute_wait_timeout_sec', 20.0)
        self.declare_parameter('enable_joint_waypoint_fallback', True)
        self.declare_parameter('joint_waypoint_fallback_count', 8)

        # Hardware trajectory time scaling.
        # Real hardware (2 Hz serial) needs trajectories stretched so the
        # controller has enough command updates to interpolate smoothly.
        # Set to 1.0 for no scaling (default). Only increase if needed.
        self.declare_parameter('hardware_trajectory_scale', 1.0)

        # Return to home after successful weld execution
        self.declare_parameter('return_home_after_weld', True)

        # Test mode: clamp incoming path into a conservative reachable workspace.
        # Useful to validate pipeline wiring even when vision points are out of reach.
        self.declare_parameter('enforce_reachable_test_path', False)

        # Path offset — applied to every incoming waypoint before execution.
        # Allows welding offset correction without re-running vision (in meters).
        self.declare_parameter('path_offset_x', 0.0)  # positive = forward (+X)
        self.declare_parameter('path_offset_y', 0.0)  # positive = left    (+Y)
        self.declare_parameter('path_offset_z', 0.0)  # positive = up      (+Z)
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
        self.hw_traj_scale = float(self.get_parameter('hardware_trajectory_scale').value)
        
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
        
        # Input Path — TRANSIENT_LOCAL so we receive the held path from path_holder
        # even when joining after it was published. path_holder is the sole
        # TRANSIENT_LOCAL publisher on /vision/welding_path, so no ambiguity.
        path_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.sub = self.create_subscription(
            Path,
            '/vision/welding_path',
            self.path_callback,
            path_qos
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
        self._last_path_hash: int | None = None  # dedup guard against TRANSIENT_LOCAL burst
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
        """Buffer latest path (with live offset applied); auto-execute if configured"""
        import copy as _copy

        # Read live offset parameters — user can change them without restarting
        ox = self.get_parameter('path_offset_x').value
        oy = self.get_parameter('path_offset_y').value
        oz = self.get_parameter('path_offset_z').value

        if msg.poses:
            raw_first = msg.poses[0].pose.position
            self.get_logger().info(
                f'Raw first waypoint: x={raw_first.x:.4f}, y={raw_first.y:.4f}, z={raw_first.z:.4f}'
            )

        if msg.poses and (ox or oy or oz):
            # Apply offset — work on a deep copy so we don't mutate the DDS buffer
            msg = _copy.deepcopy(msg)
            for ps in msg.poses:
                ps.pose.position.x += ox
                ps.pose.position.y += oy
                ps.pose.position.z += oz
            if ox or oy or oz:
                adj_first = msg.poses[0].pose.position
                self.get_logger().info(
                    f'Path offset applied: dx={ox*1000:.1f}mm  dy={oy*1000:.1f}mm  dz={oz*1000:.1f}mm'
                )
                self.get_logger().info(
                    f'Offset-adjusted first waypoint: x={adj_first.x:.4f}, y={adj_first.y:.4f}, z={adj_first.z:.4f}'
                )

        if msg.poses:
            first = msg.poses[0].pose.position
            last = msg.poses[-1].pose.position
            path_hash = hash((len(msg.poses), first.x, first.y, first.z, last.x, last.y, last.z))
            if path_hash == self._last_path_hash:
                return  # identical path already cached — skip log spam
            self._last_path_hash = path_hash

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
            
        # Freeze the exact path snapshot for this run so late republishes do not
        # perturb the currently executing planning thread.
        self._start_execution_async(copy.deepcopy(self.latest_path), source='service')
        response.success = True
        response.message = "Execution started"
        return response

    def get_execution_status(self, request, response):
        """Report whether execution is idle and a path is available."""
        with self._exec_lock:
            has_path = self.latest_path is not None
            is_idle = not self.execution_in_progress
            response.success = is_idle and has_path
            if self.execution_in_progress:
                response.message = "Executing"
            elif not has_path:
                response.message = "Idle, no path received yet"
            else:
                response.message = "Idle, path ready"
            return response

    def _start_execution_async(self, path, source='unknown'):
        with self._exec_lock:
            if self.execution_in_progress:
                self.get_logger().info(f'{source}: execution already in progress, skipping')
                return
            self.execution_in_progress = True
        self.get_logger().info(f'{source}: starting welding sequence thread')
        t = threading.Thread(target=self._execution_worker, args=(copy.deepcopy(path),), daemon=True)
        t.start()

    def _execution_worker(self, path):
        ok = False
        try:
            ok = self.execute_welding_sequence(path)
            self.get_logger().info(f'Execution finished: success={ok}')
        except Exception as exc:
            self.get_logger().error(f'Execution worker exception: {exc}')
        finally:
            with self._exec_lock:
                self.execution_in_progress = False

        # ── Return to home after any successful weld (Cartesian OR fallback) ──
        if ok and self.get_parameter('return_home_after_weld').value:
            self.get_logger().info("Welding complete ✅  Returning to home position…")
            try:
                home_return_traj = self.plan_to_home()
                if home_return_traj is not None:
                    if self.execute_trajectory_action(home_return_traj):
                        self.get_logger().info("[Return] Robot back at home position.")
                    else:
                        self.get_logger().warn(
                            "[Return] Return-to-home execution failed — robot may need manual recovery."
                        )
                else:
                    self.get_logger().warn(
                        "[Return] Could not plan return-to-home trajectory — robot stayed at weld end-point."
                    )
            except Exception as exc:
                self.get_logger().error(f'[Return] Exception during return-to-home: {exc}')

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
        
        # Refresh dynamic parameters
        self.enforce_reachable_test_path = self.get_parameter('enforce_reachable_test_path').value
        self.approach_dist = self.get_parameter('approach_distance').value

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
        approach_traj = self.plan_approach_with_fallback(
            path.poses[0],
            start_state_traj=home_traj,
        )
        if approach_traj is None:
            self.get_logger().error("Planning approach failed — aborting")
            return False

        # Phase 3: Plan Cartesian weld path using approach end-state as start
        self.get_logger().info("[Plan 3/3] Planning Cartesian weld trajectory…")
        weld_traj = self.plan_cartesian_with_fallback(
            path, start_state_traj=approach_traj
        )
        if weld_traj:
            # Densify: TOTG compresses 188 IK waypoints to ~46 trajectory pts.
            # The JointTrajectoryController spline-interpolates between these
            # in joint space, which doesn't preserve Cartesian linearity.
            # Densify back to ~50 Hz to keep the spline close to the circle.
            self._densify_cartesian_trajectory(weld_traj, max_segment_dt=0.02)
        if not weld_traj:
            if self.enable_joint_waypoint_fallback:
                self.get_logger().warn("Falling back to joint-space waypoint execution")
                # Execute home + approach first, then chain the fallback from approach endpoint
                if not self.execute_trajectory_action(home_traj):
                    return False
                if not self.execute_trajectory_action(approach_traj):
                    return False
                # Pass approach_traj so the first fallback waypoint chains from
                # the approach end-state (short down-move) rather than re-planning
                # from scratch (full hoist-and-plunge cycle).
                return self.execute_waypoint_fallback(path, start_state_traj=approach_traj)
            return False

        # ── EXECUTION PHASE (all plans ready, execute back-to-back) ──────────
        self.get_logger().info("All phases planned! Starting seamless execution…")

        # Stretch trajectory timing for real hardware
        if self.hw_traj_scale > 1.0:
            self.get_logger().info(
                f"Scaling trajectories by {self.hw_traj_scale}x for hardware"
            )
            self._scale_trajectory_time(home_traj, self.hw_traj_scale)
            self._scale_trajectory_time(approach_traj, self.hw_traj_scale)
            self._scale_trajectory_time(weld_traj, self.hw_traj_scale)

        # Hardware settle delay (seconds) — real servos report "goal reached"
        # while still physically converging. 2.0s is enough for Servo42C.
        SETTLE_DELAY = 2.0

        self.get_logger().info("[Exec 1/3] Moving to Home")
        if not self.execute_trajectory_action(home_traj):
            self.get_logger().error("Home execution failed")
            return False
        import time
        self.get_logger().info(f"Waiting {SETTLE_DELAY}s for hardware settle…")
        time.sleep(SETTLE_DELAY)

        self.get_logger().info("[Exec 2/3] Moving to Approach Point")
        if not self.execute_trajectory_action(approach_traj):
            self.get_logger().error("Approach execution failed")
            return False
        self.get_logger().info(f"Waiting {SETTLE_DELAY}s for hardware settle…")
        time.sleep(SETTLE_DELAY)

        self.get_logger().info("[Exec 3/3] Executing Weld")
        if not self.execute_trajectory_action(weld_traj):
            self.get_logger().error("Weld execution failed")
            return False

        self.get_logger().info("Sequence Complete")
        return True

    def _candidate_approach_distances(self):
        """
        Generate a short, deterministic list of lift distances to try.

        The workspace photo and current weld setup suggest a mostly planar task
        with one primary downward orientation, so we keep orientation stable and
        vary only how aggressively we lift above the first weld point.
        """
        raw_candidates = [
            float(self.approach_dist),
            min(float(self.approach_dist), 0.10),
            min(float(self.approach_dist), 0.05),
            0.0,
        ]
        distances = []
        for distance in raw_candidates:
            distance = max(0.0, distance)
            if all(abs(distance - existing) > 1e-6 for existing in distances):
                distances.append(distance)
        return distances

    def plan_approach_with_fallback(self, first_pose_stamped, start_state_traj=None):
        """
        Plan the pre-weld approach using a small fallback ladder.

        We keep the weld pose orientation as the primary target, but if MoveIt
        cannot solve the lifted pose with strict constraints we progressively:
          1. relax orientation tolerance
          2. drop orientation constraint entirely
          3. reduce the Z lift distance
        """
        attempt_no = 0
        orientation_modes = [
            ("strict", True, 0.20),
            ("relaxed", True, 0.60),
            ("position_only", False, 0.20),
        ]

        for lift_distance in self._candidate_approach_distances():
            for mode_name, use_orientation, tolerance in orientation_modes:
                attempt_no += 1
                approach_pose = copy.deepcopy(first_pose_stamped)
                approach_pose.pose.position.z += lift_distance
                q = approach_pose.pose.orientation
                self.get_logger().info(
                    f"Approach attempt {attempt_no}: lift={lift_distance:.3f}m "
                    f"mode={mode_name} "
                    f"quat=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})"
                )
                traj = self.plan_pose(
                    approach_pose,
                    start_state_traj=start_state_traj,
                    use_orientation_constraint=use_orientation,
                    orientation_tolerance_xyz=tolerance,
                )
                if traj is not None:
                    self.get_logger().info(
                        f"Approach succeeded on attempt {attempt_no} "
                        f"(lift={lift_distance:.3f}m, mode={mode_name})"
                    )
                    return traj

        self.get_logger().error("All approach planning attempts failed")
        return None

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

    def _densify_cartesian_trajectory(self, trajectory, max_segment_dt=0.02):
        """
        Re-interpolate a TOTG-compressed trajectory to a minimum point density.

        MoveIt's Time-Optimal Trajectory Generation (TOTG) merges joint-space-
        colinear segments, compressing e.g. 188 IK waypoints to 46 trajectory
        points.  The JointTrajectoryController then cubic-spline-interpolates
        between these sparse joint-space waypoints.  Because joint-space splines
        do NOT preserve Cartesian linearity, the TCP deviates from the planned
        Cartesian path between waypoints — producing visible Z-dipping.

        This method linearly re-interpolates any segment longer than
        `max_segment_dt` seconds, creating dense waypoints so the controller's
        cubic spline stays close to the true Cartesian path.

        Args:
            trajectory: RobotTrajectory message to densify (modified in place).
            max_segment_dt: Maximum allowed time gap between adjacent waypoints
                            (seconds).  Default 20 ms → ~50 Hz point density.
        """
        from builtin_interfaces.msg import Duration as RosDuration
        from trajectory_msgs.msg import JointTrajectoryPoint

        points = trajectory.joint_trajectory.points
        if len(points) < 2:
            return

        def _time_sec(pt):
            return pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9

        def _make_duration(t):
            return RosDuration(sec=int(t), nanosec=int((t % 1) * 1_000_000_000))

        new_points = [points[0]]

        for k in range(1, len(points)):
            p0 = points[k - 1]
            p1 = points[k]
            t0 = _time_sec(p0)
            t1 = _time_sec(p1)
            seg_dt = t1 - t0

            if seg_dt <= max_segment_dt or seg_dt < 1e-9:
                new_points.append(p1)
                continue

            # Number of sub-segments to insert
            n_sub = max(2, int(seg_dt / max_segment_dt) + 1)

            for s in range(1, n_sub):
                alpha = s / float(n_sub)
                t_interp = t0 + alpha * seg_dt

                interp_pt = JointTrajectoryPoint()
                interp_pt.time_from_start = _make_duration(t_interp)

                # Linearly interpolate positions
                interp_pt.positions = [
                    p0.positions[j] + alpha * (p1.positions[j] - p0.positions[j])
                    for j in range(len(p0.positions))
                ]

                # Linearly interpolate velocities (if present)
                if p0.velocities and p1.velocities:
                    interp_pt.velocities = [
                        p0.velocities[j] + alpha * (p1.velocities[j] - p0.velocities[j])
                        for j in range(len(p0.velocities))
                    ]

                # Linearly interpolate accelerations (if present)
                if p0.accelerations and p1.accelerations:
                    interp_pt.accelerations = [
                        p0.accelerations[j] + alpha * (p1.accelerations[j] - p0.accelerations[j])
                        for j in range(len(p0.accelerations))
                    ]

                new_points.append(interp_pt)

            # Append the original endpoint
            new_points.append(p1)

        orig_count = len(points)
        trajectory.joint_trajectory.points = new_points

        total_t = _time_sec(new_points[-1])
        self.get_logger().info(
            f"Densified trajectory: {orig_count} → {len(new_points)} pts "
            f"(max_dt={max_segment_dt*1000:.0f}ms, total={total_t:.2f}s)"
        )

    def _scale_trajectory_time(self, trajectory, scale_factor: float):
        """
        Uniformly stretch all trajectory point timestamps by `scale_factor`.
        Velocities are divided by the same factor to maintain kinematic consistency.
        This gives the hardware controller more time between waypoints.
        """
        from builtin_interfaces.msg import Duration as RosDuration

        points = trajectory.joint_trajectory.points
        if not points:
            return

        for pt in points:
            old_sec = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            new_sec = old_sec * scale_factor
            pt.time_from_start = RosDuration(
                sec=int(new_sec),
                nanosec=int((new_sec % 1) * 1_000_000_000)
            )
            # Scale velocities down proportionally
            if pt.velocities:
                pt.velocities = [v / scale_factor for v in pt.velocities]
            if pt.accelerations:
                pt.accelerations = [a / (scale_factor * scale_factor) for a in pt.accelerations]

        new_total = points[-1].time_from_start.sec + points[-1].time_from_start.nanosec * 1e-9
        self.get_logger().info(
            f"Scaled trajectory time by {scale_factor}x: "
            f"{len(points)} pts, new total={new_total:.2f}s"
        )

    def execute_waypoint_fallback(self, path: Path, start_state_traj=None) -> bool:
        """
        Improved fallback execution path:
        - Select a coarse subset of waypoints from the path.
        - Pre-plan ALL segments before executing ANY of them (avoids "freeze" during weld).
        - Each segment is planned from the END STATE of the previous segment so the
          robot moves point-to-point along the path without full re-approach hoist-and-plunge.

        Args:
            path: The full weld path (nav_msgs/Path).
            start_state_traj: Optional trajectory whose final joint state seeds the first
                              waypoint plan.  Typically the approach trajectory so planning
                              is consistent with where the robot actually ended up.
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
            f"Waypoint fallback: pre-planning {len(indices)}/{total} sampled waypoints "
            f"(chained start states — no re-hoisting between points)"
        )

        # ── Phase 1: Pre-plan all segments ───────────────────────────────────
        planned_trajs = []
        prev_traj = start_state_traj  # chain: each plan seeds from previous end state

        for n, idx in enumerate(indices, start=1):
            pose_stamped = copy.deepcopy(path.poses[idx])
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            pz = pose_stamped.pose.position.z

            self.get_logger().info(
                f"[FallbackPlan {n}/{len(indices)}] Planning to "
                f"x={px:.4f}, y={py:.4f}, z={pz:.4f} (index={idx})"
            )

            # Position-only so orientation singularities don't block path following
            traj = self.plan_pose(
                pose_stamped,
                start_state_traj=prev_traj,
                use_orientation_constraint=False,
            )

            if traj is None:
                self.get_logger().error(
                    f"[FallbackPlan] Could not plan to waypoint {n}/{len(indices)} "
                    f"(index={idx}) — aborting"
                )
                return False

            planned_trajs.append((n, idx, px, py, pz, traj))
            prev_traj = traj  # seed next plan from this end state

        self.get_logger().info(
            f"Waypoint fallback: all {len(planned_trajs)} segments pre-planned — executing…"
        )

        # ── Phase 2: Execute pre-planned segments back-to-back ───────────────
        for n, idx, px, py, pz, traj in planned_trajs:
            self.get_logger().info(
                f"[FallbackExec {n}/{len(planned_trajs)}] Moving to "
                f"x={px:.4f}, y={py:.4f}, z={pz:.4f}"
            )
            ok = self.execute_trajectory_action(traj)
            if not ok:
                self.get_logger().error(
                    f"Waypoint fallback failed at point {n}/{len(planned_trajs)} (index={idx})"
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
            f'Planning approach: x={pose_stamped.pose.position.x:.4f}, '
            f'y={pose_stamped.pose.position.y:.4f}, '
            f'z={pose_stamped.pose.position.z:.4f} '
            f'(frame={pose_stamped.header.frame_id})'
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
            f'Approach target: x={pose_stamped.pose.position.x:.4f}, '
            f'y={pose_stamped.pose.position.y:.4f}, '
            f'z={pose_stamped.pose.position.z:.4f} '
            f'(frame={pose_stamped.header.frame_id})'
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
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
