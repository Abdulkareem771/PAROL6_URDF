"""
vision_moveit.launch.py
=======================
Unified launch: Vision Pipeline + MoveIt (fake hardware) + RViz + MoveIt Controller

Combines:
  - ros2 bag play  (replays Kinect data — set use_bag:=false for live camera)
  - path_optimizer     →  /vision/weld_lines_2d
  - depth_matcher      →  /vision/weld_lines_3d
  - path_generator     →  /vision/welding_path
  - moveit_controller  (consumes /vision/welding_path and calls MoveIt services/actions)
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
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # ── Launch Arguments ───────────────────────────────────────────────
    use_bag_arg = DeclareLaunchArgument(
        'use_bag', default_value='true',
        description='true = replay bag; false = use live Kinect'
    )
    use_bag = LaunchConfiguration('use_bag')

    single_frame_detection_arg = DeclareLaunchArgument(
        'single_frame_detection', default_value='true',
        description='true = process one camera frame then stop detector subscription'
    )
    single_frame_detection = LaunchConfiguration('single_frame_detection')

    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame', default_value='kinect2_rgb_optical_frame',
        description='The TF frame ID for the camera optical frame'
    )
    camera_frame = LaunchConfiguration('camera_frame')


    # ── MoveIt Config ──────────────────────────────────────────────────
    pkg_parol6            = get_package_share_directory('parol6')
    pkg_moveit            = get_package_share_directory('parol6_moveit_config')
    pkg_vision            = get_package_share_directory('parol6_vision')
    pkg_hardware          = get_package_share_directory('parol6_hardware')

    moveit_config = (
        MoveItConfigsBuilder("parol6")
        .robot_description(file_path=os.path.join(pkg_hardware, 'urdf', 'parol6.urdf.xacro'), mappings={"use_ros2_control": "true", "allow_spoofing": "true"})
        .robot_description_semantic(file_path=os.path.join(pkg_moveit, 'config', 'parol6.srdf'))
        .trajectory_execution(file_path=os.path.join(pkg_moveit, 'config', 'moveit_controllers.yaml'))
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )

    # For fake hardware execution, we must swap the hardcoded Gazebo plugin with the mock system.
    fake_robot_description = {
        "robot_description": moveit_config.robot_description["robot_description"].replace(
            "ign_ros2_control/IgnitionSystem", 
            "mock_components/GenericSystem"
        )
    }

    ros2_controllers_yaml = os.path.join(pkg_moveit, 'config', 'ros2_controllers_sim.yaml')

    # ── 1. Bag Player (conditional) ────────────────────────────────────
    bag_path = '/workspace/rosbag2_2026_01_26-23_26_59'
    play_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--loop',
             '--remap', '/tf_static:=/tf_static_bag_discard',
             '--remap', '/tf:=/tf_bag_discard',
             '--clock'],
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
        
        arguments=['--x', '0.646', '--y', '0.1225', '--z', '1.015',
                   '--yaw', '1.603684', '--pitch', '0.0', '--roll', '-3.14159',
                   '--frame-id', 'base_link', '--child-frame-id', 'kinect2_link'],
        output='screen'
    )

    static_tf_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_optical',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '-1.5708', '--pitch', '0', '--yaw', '-1.5708',
                   '--frame-id', 'kinect2_link', '--child-frame-id', camera_frame],
        output='screen'
    )

    # ── 3. Vision Nodes ────────────────────────────────────────────────
    path_optimizer = Node(
        package='parol6_vision',
        executable='path_optimizer',
        name='path_optimizer',
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

    # ── 3b. Vision Trajectory Executor ─────────────────────────────────
    vision_trajectory_executor = Node(
        package='parol6_vision',
        executable='vision_trajectory_executor',
        name='vision_trajectory_executor',
        output='screen',
        parameters=[{
            'planning_group': 'parol6_arm',
            'base_frame': 'base_link',
            'end_effector_link': 'L6',
            'step_size': 0.02,
            'jump_threshold': 1.5,
            'auto_execute': True,
            'use_sim_time': True,
        }],
    )

    # ── 4. Point Cloud (RViz 3D view) ──────────────────────────────────
    point_cloud_xyzrgb = Node(
        package='depth_image_proc',
        executable='point_cloud_xyzrgb_node',
        name='point_cloud_xyzrgb',
        output='screen',
        remappings=[
            ('/rgb/camera_info',             '/kinect2/sd/camera_info'),
            ('/rgb/image_rect_color',        '/kinect2/sd/image_color_rect'),
            ('/depth_registered/image_rect', '/kinect2/sd/image_depth_rect'),
        ],
        parameters=[{'use_sim_time': True}]
    )

    # ── 5. MoveIt – move_group (fake hardware, no ESP32 needed) ────────
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            {**moveit_config.to_dict(), **fake_robot_description},
            {"use_sim_time": True},
        ],
    )

    # ── 6. ros2_control (FakeSystem) ───────────────────────────────────
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[fake_robot_description, ros2_controllers_yaml],
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
        parameters=[fake_robot_description],
    )

    # ── 8. RViz  (vision_debug.rviz = vision overlays + MotionPlanning) ─
    rviz_config = os.path.join(pkg_vision, 'config', 'vision_debug.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[
            fake_robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": True},
        ],
        output='screen',
    )

    return LaunchDescription([
        use_bag_arg,
        single_frame_detection_arg,
        camera_frame_arg,
        # Bag
        play_bag,
        # TFs
        static_tf_world,
        static_tf_camera,
        static_tf_optical,
        # Vision pipeline
        path_optimizer,
        depth_matcher,
        path_generator,
        vision_trajectory_executor,
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
