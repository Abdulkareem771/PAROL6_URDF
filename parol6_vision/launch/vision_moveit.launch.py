"""
vision_moveit.launch.py
=======================
Unified launch: Vision Pipeline + MoveIt (fake hardware) + RViz

Combines:
  - ros2 bag play  (replays Kinect data — set use_bag:=false for live camera)
  - red_line_detector  →  /vision/weld_lines_2d
  - depth_matcher      →  /vision/weld_lines_3d
  - path_generator     →  /vision/welding_path
  - move_group         (fake hardware — robot model + motion planning panel)
  - ros2_control_node  (FakeSystem — no ESP32 needed)
  - RViz with vision_debug.rviz  (all overlays + MotionPlanning panel)

Usage:
  # Bag replay (default):
  ros2 launch parol6_vision vision_moveit.launch.py

  # Live Kinect + vision (container must have kinect2_bridge running):
  ros2 launch parol6_vision vision_moveit.launch.py use_bag:=false
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # ── Launch Arguments ───────────────────────────────────────────────
    use_bag_arg = DeclareLaunchArgument(
        'use_bag', default_value='true',
        description='true = replay bag; false = use live Kinect'
    )
    use_bag = LaunchConfiguration('use_bag')

    # ── MoveIt Config ──────────────────────────────────────────────────
    pkg_parol6            = get_package_share_directory('parol6')
    pkg_moveit            = get_package_share_directory('parol6_moveit_config')
    pkg_vision            = get_package_share_directory('parol6_vision')

    moveit_config = (
        MoveItConfigsBuilder("parol6")
        .robot_description(file_path=os.path.join(pkg_parol6, 'urdf', 'PAROL6.urdf'))
        .robot_description_semantic(file_path=os.path.join(pkg_moveit, 'config', 'parol6.srdf'))
        .trajectory_execution(file_path=os.path.join(pkg_moveit, 'config', 'moveit_controllers.yaml'))
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )

    ros2_controllers_yaml = os.path.join(pkg_moveit, 'config', 'ros2_controllers.yaml')

    # ── 1. Bag Player (conditional) ────────────────────────────────────
    bag_path = '/workspace/rosbag2_2026_01_26-23_26_59'
    play_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--loop',
             '--remap', '/tf_static:=/tf_static_bag_discard',
             '--remap', '/tf:=/tf_bag_discard'],
        output='screen',
        condition=IfCondition(use_bag)
    )

    # ── 2. Static TFs ──────────────────────────────────────────────────
    static_tf_world = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_world',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'world', '--child-frame-id', 'base_link'],
    )

    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_camera',
        # Camera is 1.2 m in front of robot base, 1.0 m up, looking back+down
        # yaw=pi  → faces toward -X (toward robot)
        # pitch=+0.52 rad (+30° = tilts DOWN after yaw=π flip) → tilts down toward workspace
        arguments=['--x', '1.2', '--y', '0.0', '--z', '0.65',
                   '--yaw', '1.5708', '--pitch', '0.0', '--roll', '-1.5708',
                   '--frame-id', 'base_link', '--child-frame-id', 'kinect2_link'],
        output='screen'
    )

    static_tf_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_optical',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '-1.5708', '--pitch', '0', '--yaw', '-1.5708',
                   '--frame-id', 'kinect2_link', '--child-frame-id', 'kinect2_rgb_optical_frame'],
        output='screen'
    )

    # ── 3. Vision Nodes ────────────────────────────────────────────────
    red_line_detector = Node(
        package='parol6_vision',
        executable='red_line_detector',
        name='red_line_detector',
        output='screen',
        parameters=[{
            'publish_debug_images': True,
            'use_sim_time': True,
        }]
    )

    depth_matcher = Node(
        package='parol6_vision',
        executable='depth_matcher',
        name='depth_matcher',
        output='screen',
        parameters=[{
            'sync_time_tolerance': 0.5,
            'min_depth_quality': 0.05,
            'max_depth': 5000.0,
            'min_depth': 100.0,
            'min_valid_points': 2,
            'use_sim_time': True,
        }]
    )

    path_generator = Node(
        package='parol6_vision',
        executable='path_generator',
        name='path_generator',
        output='screen',
        parameters=[{
            'spline_degree': 3,
            'spline_smoothing': 0.005,
            'waypoint_spacing': 0.005,
            'approach_angle_deg': 45.0,
            'auto_generate': True,
            'min_points_for_path': 3,
            'use_sim_time': True,
        }]
    )

    # ── 4. Point Cloud (RViz 3D view) ──────────────────────────────────
    point_cloud_xyzrgb = Node(
        package='depth_image_proc',
        executable='point_cloud_xyzrgb_node',
        name='point_cloud_xyzrgb',
        output='screen',
        remappings=[
            ('/rgb/camera_info',             '/kinect2/qhd/camera_info'),
            ('/rgb/image_rect_color',        '/kinect2/qhd/image_color_rect'),
            ('/depth_registered/image_rect', '/kinect2/qhd/image_depth_rect'),
        ],
        parameters=[{'use_sim_time': True}]
    )

    # ── 5. MoveIt – move_group (fake hardware, no ESP32 needed) ────────
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict()],
    )

    # ── 6. ros2_control (FakeSystem) ───────────────────────────────────
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[moveit_config.robot_description, ros2_controllers_yaml],
        output='both',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['parol6_arm_controller'],
        output='screen',
    )

    # ── 7. Robot State Publisher ────────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[moveit_config.robot_description],
    )

    # ── 8. RViz  (vision_debug.rviz = vision overlays + MotionPlanning) ─
    rviz_config = os.path.join(pkg_vision, 'config', 'vision_debug.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
        output='screen',
    )

    return LaunchDescription([
        use_bag_arg,
        # Bag
        play_bag,
        # TFs
        static_tf_world,
        static_tf_camera,
        static_tf_optical,
        # Vision pipeline
        red_line_detector,
        depth_matcher,
        path_generator,
        point_cloud_xyzrgb,
        # MoveIt
        move_group_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        robot_state_publisher,
        # Visualization
        rviz_node,
    ])
