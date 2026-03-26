#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseArray, PoseStamped
from moveit_msgs.srv import GetCartesianPath
from control_msgs.action import FollowJointTrajectory
from action_msgs.msg import GoalStatus

class VisionTrajectoryExecutor(Node):
    """
    Subscribes to 3D Cartesian paths from the vision pipeline.
    Transforms them to the base_link frame, computes a sparse/optimized Cartesian trajectory,
    and executes it using the standard JointTrajectoryController.
    """
    def __init__(self):
        super().__init__('vision_trajectory_executor')
        
        # Parameters
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('arm_group', 'parol6_arm')
        # Optimized for Intel Xeon E3-1505M / Quadro M1000M limits (larger step sizes to reduce load)
        self.declare_parameter('step_size', 0.02) # 2 cm resolution computes much faster on CPUs
        self.declare_parameter('jump_threshold', 1.5)
        
        self.base_frame = self.get_parameter('base_frame').value
        
        # TF2 Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ROS Interfaces
        self.path_sub = self.create_subscription(PoseArray, '/vision/path', self.path_callback, 10)
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.trajectory_action_client = ActionClient(self, FollowJointTrajectory, '/parol6_arm_controller/follow_joint_trajectory')
        
        # State tracking
        self.is_executing = False
        self.active_goal_handle = None
        
        self.get_logger().info('Vision Trajectory Executor Initialized. Waiting for /vision/path...')

    def path_callback(self, msg: PoseArray):
        """Receives the Kinect vision generated sequence of waypoints."""
        if self.is_executing:
            self.get_logger().warn("Currently executing a path. Ignoring new sequence to save compute.")
            return

        self.get_logger().info(f"Received vision path containing {len(msg.poses)} waypoints.")
        
        source_frame = msg.header.frame_id
        path_time = msg.header.stamp

        # 1. Coordinate Transformation
        transformed_poses = []
        try:
            # Stop Condition 1: Abort if TF times out or is unavailable
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                path_time,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except tf2_ros.TransformException as ex:
            self.get_logger().error(f"ABORT PLANNING: TF lookup from {source_frame} to {self.base_frame} failed: {ex}")
            return

        for pose in msg.poses:
            pose_stamp = PoseStamped()
            pose_stamp.pose = pose
            target_pose = do_transform_pose(pose_stamp.pose, transform)
            transformed_poses.append(target_pose)

        # 2. Cartesian Planning
        self.get_logger().info("Requesting MoveIt2 Cartesian Path calculation...")
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/compute_cartesian_path service not available. Start MoveIt2.')
            return

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = self.get_parameter('arm_group').value
        req.waypoints = transformed_poses
        req.max_step = self.get_parameter('step_size').value
        req.jump_threshold = self.get_parameter('jump_threshold').value
        req.avoid_collisions = True

        future = self.cartesian_client.call_async(req)
        future.add_done_callback(self.cartesian_response_callback)

    def cartesian_response_callback(self, future):
        """Handles response from the Cartesian Planner."""
        try:
            response = future.result()
            
            # Stop Condition 2: Completion fraction < 1.0
            if response.fraction < 1.0:
                self.get_logger().error(f"HALT EXECUTION: Cartesian path fraction < 1.0 ({response.fraction:.2f}). Kinematic singularity or collision detected.")
                return
            
            self.get_logger().info("Dense trajectory planned completely. Initiating execution...")
            self.execute_trajectory(response.solution)
            
        except Exception as e:
            self.get_logger().error(f"Cartesian planning call failed: {e}")

    def execute_trajectory(self, moveit_trajectory):
        """Sends payload to standard JointTrajectoryController."""
        self.is_executing = True
        
        if not self.trajectory_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server /parol6_arm_controller/follow_joint_trajectory not available! Aborting execution.")
            self.is_executing = False
            return

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = moveit_trajectory.joint_trajectory

        send_goal_future = self.trajectory_action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Action goal acceptance state."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory execution rejected by the JointTrajectoryController.")
            self.is_executing = False
            return

        self.active_goal_handle = goal_handle
        self.get_logger().info("Trajectory accepted. Monitoring execution...")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Checks for successful conclusion of trajectory execution."""
        status = future.result().status
        
        # Stop Condition 3: Terminate successfully only when final waypoint is reached
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("SUCCESS! End-effector successfully reached the final vision sequence waypoint.")
        else:
            self.get_logger().error(f"Trajectory execution stopped prematurely (Status: {status})")
            
        self.is_executing = False
        self.active_goal_handle = None

def main(args=None):
    rclpy.init(args=args)
    node = VisionTrajectoryExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
