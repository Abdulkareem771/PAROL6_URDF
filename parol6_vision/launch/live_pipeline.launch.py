"""
live_pipeline.launch.py — PAROL6 Live Vision Pipeline (no bag)
================================================================
Launches the complete live vision pipeline using the real Kinect camera.

Nodes started:
  1. capture_images   — captures live Kinect frames (keyboard or timed)
  2. crop_image       — always-active ROI crop relay (loads ~/.parol6/crop_config.json)
  3. path_optimizer   — detects red marker lines in the processed image
  4. depth_matcher    — lifts 2-D weld lines to 3-D using depth data
  5. path_generator   — produces the final Nav2 Path message for MoveIt

Also sets up the required static TF root so depth_matcher can project
into 3-D: world → base_link → kinect2 → kinect2_link → kinect2_rgb_optical_frame.
A point-cloud node (depth_image_proc) fuses colour + depth for RViz.

Usage
-----
  ros2 launch parol6_vision live_pipeline.launch.py

  # Capture mode options:
  ros2 launch parol6_vision live_pipeline.launch.py capture_mode:=keyboard
  ros2 launch parol6_vision live_pipeline.launch.py capture_mode:=timed frame_time:=15.0

  # Disable RViz:
  ros2 launch parol6_vision live_pipeline.launch.py use_rviz:=false

Note
----
The Kinect camera (kinect2_bridge) must be started SEPARATELY before this
launch file, as it lives in a different workspace:
  ros2 launch ~/Desktop/PAROL6_URDF/kinect2_bridge_gpu.yaml
"""

import os
import yaml
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as _Rot

# ── Camera TF: read from calibration yaml or use defaults ────────────────────
_CAMERA_TF_PATH = Path.home() / '.parol6' / 'camera_tf.yaml'
_CAM_DEFAULTS = {
    'frame_id':       'base_link',
    'child_frame_id': 'kinect2',
    'x': 0.646, 'y': 0.1225, 'z': 1.015,
    'roll': -3.14159, 'pitch': 0.0, 'yaw': 1.603684,
}
_cam_cfg = dict(_CAM_DEFAULTS)
if _CAMERA_TF_PATH.exists():
    try:
        with open(_CAMERA_TF_PATH) as _f:
            _loaded = yaml.safe_load(_f) or {}
        _cam_cfg.update({k: v for k, v in _loaded.items() if v is not None and not k.startswith('_')})
        print(f"[live_pipeline] Using calibrated camera TF from {_CAMERA_TF_PATH}")
        print(f"  {_cam_cfg['frame_id']} → {_cam_cfg['child_frame_id']}")
        print(f"  xyz=({_cam_cfg['x']:.4f}, {_cam_cfg['y']:.4f}, {_cam_cfg['z']:.4f})")
    except Exception as _e:
        print(f"[live_pipeline] WARNING: Could not load {_CAMERA_TF_PATH}: {_e} — using defaults")
else:
    print(f"[live_pipeline] No camera_tf.yaml found — using hardcoded defaults")

# Build static_transform_publisher positional args from config.
# We support both quaternion (qx/qy/qz/qw) and euler (roll/pitch/yaw) formats.
def _build_camera_tf_args(cfg: dict) -> list:
    x = str(float(cfg.get('x', 0.0)))
    y = str(float(cfg.get('y', 0.0)))
    z = str(float(cfg.get('z', 0.0)))
    if 'qx' in cfg:
        qx = str(float(cfg['qx']))
        qy = str(float(cfg['qy']))
        qz = str(float(cfg['qz']))
        qw = str(float(cfg['qw']))
        return [x, y, z, qx, qy, qz, qw, cfg['frame_id'], cfg['child_frame_id']]
    else:
        roll  = float(cfg.get('roll',  0.0))
        pitch = float(cfg.get('pitch', 0.0))
        yaw   = float(cfg.get('yaw',   0.0))
        q = _Rot.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        return [x, y, z, str(q[0]), str(q[1]), str(q[2]), str(q[3]),
                cfg['frame_id'], cfg['child_frame_id']]


def generate_launch_description():
    pkg_vision = get_package_share_directory('parol6_vision')
    pkg_share   = FindPackageShare('parol6_vision')

    # ── Launch arguments ──────────────────────────────────────────────
    capture_mode_arg = DeclareLaunchArgument(
        'capture_mode',
        default_value='keyboard',
        description="Capture trigger: 'keyboard' (press s+Enter) or 'timed'"
    )
    frame_time_arg = DeclareLaunchArgument(
        'frame_time',
        default_value='10.0',
        description='Seconds between auto-captures (timed mode only)'
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz for visualisation'
    )

    # ── Config files ──────────────────────────────────────────────────
    detection_params = PathJoinSubstitution(
        [pkg_share, 'config', 'detection_params.yaml']
    )
    path_params = PathJoinSubstitution(
        [pkg_share, 'config', 'path_params.yaml']
    )
    rviz_config = os.path.join(pkg_vision, 'config', 'vision_debug.rviz')

    # ── Static TFs ────────────────────────────────────────────────────
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
        # TF values are loaded from ~/.parol6/camera_tf.yaml at launch time.
        # If that file is missing, hardcoded defaults are used (see top of file).
        # Use GUI "Cam Calibrate" tab or camera_tf_enforcer node for live override.
        arguments=_build_camera_tf_args(_cam_cfg),
        output='log',
    )

    # ── Stage 1: Capture Images ───────────────────────────────────────
    # Always publishes to /vision/captured_image_raw; crop_image relays downstream.
    capture_node = Node(
        package='parol6_vision',
        executable='capture_images',
        name='capture_images',
        parameters=[{
            'capture_mode':  LaunchConfiguration('capture_mode'),
            'frame_time':    LaunchConfiguration('frame_time'),
            'output_topic':  '/vision/captured_image_raw',
        }],
        output='screen',
    )

    # ── Stage 1b: Crop Image (always-active) ─────────────────────────
    crop_node = Node(
        package='parol6_vision',
        executable='crop_image',
        name='crop_image',
        output='screen',
    )

    # ── Stage 2: Path Optimizer ───────────────────────────────────────
    path_optimizer_node = Node(
        package='parol6_vision',
        executable='path_optimizer',
        name='path_optimizer',
        parameters=[detection_params, {
            'publish_debug_images': True,
        }],
        output='screen',
    )

    # ── Stage 3: Depth Matcher ────────────────────────────────────────
    depth_matcher_node = Node(
        package='parol6_vision',
        executable='depth_matcher',
        name='depth_matcher',
        parameters=[detection_params, {
            'sync_time_tolerance': 0.5,
            'min_depth_quality':   0.05,
            'max_depth':           5000.0,
            'min_depth':           100.0,
            'min_valid_points':    2,
        }],
        output='screen',
    )

    # ── Stage 4: Path Generator ───────────────────────────────────────
    path_generator_node = Node(
        package='parol6_vision',
        executable='path_generator',
        name='path_generator',
        parameters=[path_params],
        output='screen',
    )

    # ── Stage 4b: Inject Path Node ────────────────────────────────────
    # Receives test paths from GUI and publishes to staging topic.
    inject_path_node = Node(
        package='parol6_vision',
        executable='inject_path',
        name='inject_path_node',
        output='screen',
    )

    # ── Stage 4c: Path Holder (Authoritative Publisher) ───────────────
    # Single authoritative TRANSIENT_LOCAL publisher on /vision/welding_path.
    path_holder_node = Node(
        package='parol6_vision',
        executable='path_holder',
        name='path_holder',
        parameters=[{'active_source': 'generated'}],
        output='screen',
    )

    # ── Point Cloud (for RViz depth viz) ─────────────────────────────
    point_cloud_node = Node(
        package='depth_image_proc',
        executable='point_cloud_xyzrgb_node',
        name='point_cloud_xyzrgb',
        remappings=[
            ('/rgb/camera_info',             '/kinect2/qhd/camera_info'),
            ('/rgb/image_rect_color',         '/kinect2/qhd/image_color_rect'),
            ('/depth_registered/image_rect',  '/kinect2/qhd/image_depth_rect'),
        ],
        output='screen',
    )

    # ── RViz (optional) ───────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription([
        # Args
        capture_mode_arg,
        frame_time_arg,
        use_rviz_arg,
        # TFs
        static_tf_world,
        static_tf_camera,
        # Pipeline
        capture_node,
        crop_node,
        path_optimizer_node,
        depth_matcher_node,
        path_generator_node,
        inject_path_node,
        path_holder_node,
        point_cloud_node,
        rviz_node,
    ])
