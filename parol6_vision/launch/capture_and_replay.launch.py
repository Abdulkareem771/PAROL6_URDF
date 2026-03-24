"""
capture_and_replay.launch.py — Two-stage offline vision pipeline.

Stage 1  capture_images
    Listens to live Kinect2 topics and saves matched colour + depth PNG pairs
    to parol6_vision/data/images_captured/ on demand (keyboard or timed).

Stage 2  read_image
    Watches the same folder for new pairs and republishes them as ROS topics:
        /vision/captured_image_color  (sensor_msgs/Image, bgr8)
        /vision/captured_image_depth  (sensor_msgs/Image, 16UC1)

Downstream nodes are remapped to consume these replay topics instead of
the live camera topics:
    red_line_detector : /kinect2/sd/image_color_rect  →  /vision/captured_image_color
    depth_matcher     : /kinect2/sd/image_depth_rect  →  /vision/captured_image_depth

Usage
-----
    # Default (keyboard capture, auto-detect save_dir):
    ros2 launch parol6_vision capture_and_replay.launch.py

    # Timed mode, save every 30 s:
    ros2 launch parol6_vision capture_and_replay.launch.py \
        capture_mode:=timed frame_time:=30.0

    # Custom image folder:
    ros2 launch parol6_vision capture_and_replay.launch.py \
        save_dir:=/path/to/my/images
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    pkg_share = FindPackageShare('parol6_vision')
    detection_config = PathJoinSubstitution(
        [pkg_share, 'config', 'detection_params.yaml']
    )

    # ── Launch arguments ─────────────────────────────────────────────
    save_dir_arg = DeclareLaunchArgument(
        'save_dir',
        default_value='parol6_vision/data/images_captured',
        description='Folder for saving / reading image pairs'
    )
    capture_mode_arg = DeclareLaunchArgument(
        'capture_mode',
        default_value='keyboard',
        description="Capture trigger: 'keyboard' (press s+Enter) or 'timed'"
    )
    frame_time_arg = DeclareLaunchArgument(
        'frame_time',
        default_value='60.0',
        description='Seconds between auto-saves (timed mode only)'
    )
    poll_rate_arg = DeclareLaunchArgument(
        'poll_rate',
        default_value='1.0',
        description='Hz — how often read_image polls the folder for new files'
    )

    # ── Stage 1: Capture_images ──────────────────────────────────────
    capture_node = Node(
        package='parol6_vision',
        executable='capture_images',
        name='capture_images',
        parameters=[{
            'save_dir':       LaunchConfiguration('save_dir'),
            'capture_mode':   LaunchConfiguration('capture_mode'),
            'frame_time':     LaunchConfiguration('frame_time'),
        }],
        output='screen',
    )

    # ── Stage 2: Read_image ──────────────────────────────────────────
    read_node = Node(
        package='parol6_vision',
        executable='read_image',
        name='read_image',
        parameters=[{
            'save_dir':   LaunchConfiguration('save_dir'),
            'poll_rate':  LaunchConfiguration('poll_rate'),
            'frame_id':   'kinect2_rgb_optical_frame',
        }],
        output='screen',
    )

    # ── Red Line Detector ────────────────────────────────────────────
    # Subscribes to /vision/captured_image_color (hardcoded in node source)
    detector_node = Node(
        package='parol6_vision',
        executable='red_line_detector',
        name='red_line_detector',
        parameters=[detection_config, {
            'publish_debug_images': True,
        }],
        output='screen',
    )

    # ── Depth Matcher ─────────────────────────────────────────────────
    # Subscribes to /vision/captured_image_depth (hardcoded in node source)
    depth_node = Node(
        package='parol6_vision',
        executable='depth_matcher',
        name='depth_matcher',
        parameters=[detection_config],
        output='screen',
    )

    # ── Static TFs (required by depth_matcher for 3-D projection) ────
    static_tf_world = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--yaw', '0.0', '--pitch', '0.0', '--roll', '0.0',
            '--frame-id', 'world', '--child-frame-id', 'base_link',
        ],
        output='log',
    )

    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_camera',
        arguments=[
            '--x', '0.3', '--y', '0.0', '--z', '0.45',
            '--qx', '0.7071', '--qy', '0.0', '--qz', '0.0', '--qw', '0.7071',
            '--frame-id', 'base_link', '--child-frame-id', 'kinect2_link',
        ],
        output='log',
    )

    static_tf_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_optical',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '-1.5708', '--pitch', '0.0', '--yaw', '-1.5708',
            '--frame-id', 'kinect2_link',
            '--child-frame-id', 'kinect2_rgb_optical_frame',
        ],
        output='log',
    )

    return LaunchDescription([
        # Args
        save_dir_arg,
        capture_mode_arg,
        frame_time_arg,
        poll_rate_arg,
        # TFs
        static_tf_world,
        static_tf_camera,
        static_tf_optical,
        # Pipeline nodes
        capture_node,
        read_node,
        detector_node,
        depth_node,
    ])
