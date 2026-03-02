
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    pkg_vision = get_package_share_directory('parol6_vision')

    # Load Robot Description (URDF) for Visualization
    moveit_config = (
        MoveItConfigsBuilder("parol6")
        .robot_description(file_path=os.path.join(
            get_package_share_directory("parol6"),
            "urdf",
            "PAROL6.urdf"
        ))
        .robot_description_semantic(file_path=os.path.join(
            get_package_share_directory("parol6_moveit_config"),
            "config",
            "parol6.srdf"
        ))
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # -----------------------------------------------------------------
    # 1. Play ROS Bag (Looping)
    # -----------------------------------------------------------------
    bag_path = '/workspace/rosbag2_2026_01_26-23_26_59'
    play_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--loop',
             '--remap', '/tf_static:=/tf_static_bag_discard',
             '--remap', '/tf:=/tf_bag_discard'],
        output='screen'
    )

    # -----------------------------------------------------------------
    # 2. Static TFs  (matches camera_setup.launch.py calibration)
    # -----------------------------------------------------------------
    static_tf_world = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_world',
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.0',
                   '--yaw', '0.0', '--pitch', '0.0', '--roll', '0.0',
                   '--frame-id', 'world', '--child-frame-id', 'base_link'],
    )

    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_camera',
        # Camera 1.2 m in front of robot, 1.0 m high, looking back+down at workspace
        arguments=['--x', '1.2', '--y', '0.0', '--z', '0.65',
                   '--yaw', '1.5708', '--pitch', '0.0', '--roll', '-1.5708',
                   '--frame-id', 'base_link', '--child-frame-id', 'kinect2_link'],
        output='screen'
    )

    static_tf_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_optical',
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.0',
                   '--roll', '-1.5708', '--pitch', '0.0', '--yaw', '-1.5708',
                   '--frame-id', 'kinect2_link', '--child-frame-id', 'kinect2_rgb_optical_frame'],
        output='screen'
    )

    # -----------------------------------------------------------------
    # 3. Red Line Detector  →  /vision/weld_lines_2d
    # -----------------------------------------------------------------
    red_line_detector = Node(
        package='parol6_vision',
        executable='red_line_detector',
        name='red_line_detector',
        output='screen',
        parameters=[{
            'debug_image_topic': '/red_line_detector/debug_image',
            'publish_debug_images': True,
            'use_sim_time': True
        }]
    )

    # -----------------------------------------------------------------
    # 4. Depth Matcher  →  /vision/weld_lines_3d
    # -----------------------------------------------------------------
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
            'use_sim_time': True
        }]
    )

    # -----------------------------------------------------------------
    # 5. Path Generator  →  /vision/welding_path  ← NEW
    # -----------------------------------------------------------------
    path_generator = Node(
        package='parol6_vision',
        executable='path_generator',
        name='path_generator',
        output='screen',
        parameters=[{
            'spline_degree': 3,
            'spline_smoothing': 0.005,
            'waypoint_spacing': 0.005,   # 5 mm
            'approach_angle_deg': 45.0,
            'auto_generate': True,
            'min_points_for_path': 3,    # Lowered for bag data
            'use_sim_time': True
        }]
    )

    # -----------------------------------------------------------------
    # 6. Point Cloud XYZRGB (for RViz depth view)
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # 7. Robot State Publisher + Dummy Joint Publisher
    # -----------------------------------------------------------------
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[moveit_config.robot_description],
    )

    joint_state_publisher = Node(
        package='parol6_vision',
        executable='dummy_joint_publisher',
        name='dummy_joint_publisher',
        output='screen',
    )

    # -----------------------------------------------------------------
    # 8. RViz2
    # -----------------------------------------------------------------
    rviz_config_file = os.path.join(pkg_vision, 'config', 'vision_debug.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
        output='screen'
    )

    return LaunchDescription([
        play_bag,
        static_tf_world,
        static_tf_camera,
        static_tf_optical,
        red_line_detector,
        depth_matcher,
        path_generator,        # ← added
        point_cloud_xyzrgb,
        robot_state_publisher,
        joint_state_publisher,
        rviz_node,
    ])
