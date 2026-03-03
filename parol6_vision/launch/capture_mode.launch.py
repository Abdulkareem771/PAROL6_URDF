"""
capture_mode.launch.py — Convenience launch for PAROL6 capture-mode testing.

Launches:
  - Static TF publishers (world→base_link, base_link→kinect2_link,
    kinect2_link→kinect2_rgb_optical_frame)
  - red_line_detector  with capture_mode:=true
  - depth_matcher      with capture_mode:=true

NOTE: The Kinect2 bridge (or any RGB/depth camera driver) must already
be running separately before this launch is used.  Once launched, use
    ros2 run parol6_vision capture_gui
to operate the pipeline interactively.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('parol6_vision')
    config    = PathJoinSubstitution([pkg_share, 'config', 'detection_params.yaml'])

    # ── Static TFs ──────────────────────────────────────────────────
    static_tf_world = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world',
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.0',
                   '--yaw', '0.0', '--pitch', '0.0', '--roll', '0.0',
                   '--frame-id', 'world', '--child-frame-id', 'base_link'],
        output='log',
    )

    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_camera',
        arguments=['--x', '0.3', '--y', '0.0', '--z', '0.45',
                   '--qx', '0.7071', '--qy', '0.0', '--qz', '0.0', '--qw', '0.7071',
                   '--frame-id', 'base_link', '--child-frame-id', 'kinect2_link'],
        output='log',
    )

    static_tf_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_optical',
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.0',
                   '--roll', '-1.5708', '--pitch', '0.0', '--yaw', '-1.5708',
                   '--frame-id', 'kinect2_link',
                   '--child-frame-id', 'kinect2_rgb_optical_frame'],
        output='log',
    )

    # ── Red Line Detector — capture mode ────────────────────────────
    # Input is remapped so it reads from the GUI frozen-frame topic.
    detector_node = Node(
        package='parol6_vision',
        executable='red_line_detector',
        name='red_line_detector',
        parameters=[config, {
            'capture_mode': True,
            'publish_debug_images': True,
        }],
        remappings=[
            ('/kinect2/qhd/image_color_rect',
             '/capture_gui/frozen_frame'),
        ],
        output='screen',
    )

    # ── Depth Matcher — capture mode ─────────────────────────────────
    depth_node = Node(
        package='parol6_vision',
        executable='depth_matcher',
        name='depth_matcher',
        parameters=[config, {
            'capture_mode': True,
        }],
        output='screen',
    )

    return LaunchDescription([
        static_tf_world,
        static_tf_camera,
        static_tf_optical,
        detector_node,
        depth_node,
    ])
