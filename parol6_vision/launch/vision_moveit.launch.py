"""
vision_moveit.launch.py
=======================
Unified launch: Vision Pipeline + MoveIt Controller (for external MoveIt/Gazebo)

Combines:
  - ros2 bag play  (replays Kinect data — set use_bag:=false for live camera)
  - red_line_detector  →  /vision/weld_lines_2d
  - depth_matcher      →  /vision/weld_lines_3d
  - path_generator     →  /vision/welding_path
  - moveit_controller  (consumes /vision/welding_path and calls MoveIt services/actions)

Expected external stack (started separately):
  - Gazebo + controller_manager
  - move_group + RViz (e.g. parol6_moveit_config demo.launch.py use_fake_hardware:=false)

Usage:
  # Bag replay vision pipeline (default):
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

    # =========================================================================================
    # HARDWARE INJECTION: Connect the real PAROL6System plugin to the flashed ESP32.
    # The PAROL6.urdf hardcodes the IgnitionSystem, which crashes if Gazebo isn't running.
    # We replace it with PAROL6System, and point it to /dev/ttyUSB0.
    # =========================================================================================
    moveit_config.robot_description['robot_description'] = \
        moveit_config.robot_description['robot_description'].replace(
            'ign_ros2_control/IgnitionSystem',
            'parol6_hardware/PAROL6System</plugin>\n      <param name="serial_port">/dev/ttyUSB0</param>\n      <param name="baud_rate">115200</param><plugin>'
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
        # TEST POSITION: Camera tilted down (pitch=-0.52) to bring weld pixels into reach.
        # Red line is at v≈125, cy≈270: without pitch it adds +0.29m to base_z.
        # Pitch=-0.52 rad + z=0.10m → back-projects to z≈0.30–0.40m, x≈0.35m ✓
        # Real position (restore later): x=1.2, z=0.65, pitch=0.0
        arguments=['--x', '1.44', '--y', '0.0', '--z', '0.10',
                   '--yaw', '1.5708', '--pitch', '-0.52', '--roll', '-1.5708',
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

    # ── 3b. MoveIt Controller (path follower) ──────────────────────────
    moveit_controller = Node(
        package='parol6_vision',
        executable='moveit_controller',
        name='moveit_controller',
        output='screen',
        parameters=[{
            'planning_group': 'parol6_arm',
            'base_frame': 'base_link',
            'end_effector_link': 'L6',
            'approach_distance': 0.05,
            'weld_velocity': 0.01,
            # Auto execute is enabled for pipeline wiring validation.
            # New /vision/welding_path messages will trigger execution directly.
            'auto_execute': True,
            'use_sim_time': True,
            # Test-mode workspace clamp to validate path->MoveIt->Gazebo pipeline wiring.
            # Set false to use raw path points.
            'enforce_reachable_test_path': True,
            'test_workspace_min': [0.20, -0.35, 0.10],
            'test_workspace_max': [0.65, 0.35, 0.55],
            'test_min_radius_xy': 0.20,
            'test_max_radius_xy': 0.70,
        }],
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
        moveit_controller,
        point_cloud_xyzrgb,
    ])
