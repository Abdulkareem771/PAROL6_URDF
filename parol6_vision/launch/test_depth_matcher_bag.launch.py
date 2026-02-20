
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
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

    # 1. Play ROS Bag (Looping)
    # Adjust path to the bag file as needed or pass as argument
    # BAG PATH: /home/kareem/Desktop/PAROL6_URDF/rosbag2_2026_01_26-23_26_59
    bag_path = '/workspace/rosbag2_2026_01_26-23_26_59'
    
    play_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--loop',
             '--remap', '/tf_static:=/tf_static_bag_discard',
             '--remap', '/tf:=/tf_bag_discard'],
        output='screen'
    )

    # 2. Static TF (World -> Base Link) - Optional but good for RViz context if robot model is loaded
    static_tf_world = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher_world",
        arguments=["--x", "0.0", "--y", "0.0", "--z", "0.0", "--yaw", "0.0", "--pitch", "0.0", "--roll", "0.0", "--frame-id", "world", "--child-frame-id", "base_link"],
    )

    # 3. Static TF (Base Link -> Kinect Base)
    # COPIED FROM camera_setup.launch.py to match calibration
    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_camera',
        # MATCHES camera_setup.launch.py: roll=90° (qx=0.7071, qw=0.7071)
        # This correctly orients the Kinect so it points forward along base_link +X
        arguments=['--x', '0.3', '--y', '0.0', '--z', '0.45',
                   '--qx', '0.0', '--qy', '0.7071', '--qz', '0.0', '--qw', '0.7071',
                   '--frame-id', 'base_link', '--child-frame-id', 'kinect2_link'],
        output='screen'
    )

    # 4. Static TF (Kinect Link -> RGB Optical Frame)
    # COPIED FROM camera_setup.launch.py
    static_tf_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_optical',
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.0',
                   '--roll', '-1.5708', '--pitch', '0.0', '--yaw', '-1.5708',
                   '--frame-id', 'kinect2_link', '--child-frame-id', 'kinect2_rgb_optical_frame'],
        output='screen'
    )

    # 5. Red Line Detector
    red_line_detector = Node(
        package='parol6_vision',
        executable='red_line_detector',
        name='red_line_detector',
        output='screen',
        # Parameters can be loaded from yaml if needed, defaults are usually fine for testing
        parameters=[{
            'debug_image_topic': '/red_line_detector/debug_image',
            'publish_debug_images': True,
            'use_sim_time': True
        }]
    )

    # 6. Depth Matcher
    depth_matcher = Node(
        package='parol6_vision',
        executable='depth_matcher',
        name='depth_matcher',
        output='screen',
        parameters=[{
            'sync_time_tolerance': 0.5,
            'min_depth_quality': 0.05,  # Very low for testing
            'max_depth': 5000.0,         # 5 meters
            'min_depth': 100.0,          # 10 cm
            'min_valid_points': 2,       # Detector only sends skeleton endpoints
            'use_sim_time': True
        }]
    )

    # 6b. Point Cloud XYZRGB — fuses color + depth into /points (PointCloud2)
    # Executable name requires the _node suffix (depth_image_proc >= 3.x)
    point_cloud_xyzrgb = Node(
        package='depth_image_proc',
        executable='point_cloud_xyzrgb_node',
        name='point_cloud_xyzrgb',
        output='screen',
        remappings=[
            ('/rgb/camera_info',        '/kinect2/qhd/camera_info'),
            ('/rgb/image_rect_color',   '/kinect2/qhd/image_color_rect'),
            ('/depth_registered/image_rect', '/kinect2/qhd/image_depth_rect'),
        ],
        parameters=[{'use_sim_time': True}]
    )

    # 7. RViz2
    # Using the existing vision_debug.rviz or default
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

    # 8. Robot State Publisher (For visualizing the robot mesh)
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # 9. Dummy Joint State Publisher (To keep the robot from collapsing in RViz)
    # Reusing the one from camera_setup if available, or just publishing zeroes
    # Assuming 'dummy_joint_publisher' exists in parol6_vision as seen in camera_setup.launch.py
    joint_state_publisher = Node(
        package="parol6_vision",
        executable="dummy_joint_publisher",
        name="dummy_joint_publisher",
        output="screen",
    )

    return LaunchDescription([
        play_bag,
        static_tf_world,
        static_tf_camera,
        static_tf_optical,
        red_line_detector,
        depth_matcher,
        point_cloud_xyzrgb,
        rviz_node,
        robot_state_publisher,
        joint_state_publisher
    ])
