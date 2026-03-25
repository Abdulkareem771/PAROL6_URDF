from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    pkg_vision = FindPackageShare('parol6_vision')
    
    # 1. Vision Pipeline
    # Include the main launch file
    pipeline_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_vision, 'launch', 'vision_pipeline.launch.py'])
        ])
    )
    
    # 2. Mock Camera Publisher (Installed script)
    mock_camera_node = Node(
        package='parol6_vision',
        executable='mock_camera_publisher.py', # Name as in scripts=[]
        name='mock_camera',
        output='screen'
    )
    
    # 3. Static TF (Simulate Camera mounting)
    # Camera Looking Down: Rotate -90 X (to make +Z forward -> +Y down), then something to point Z down?
    # Kinects usually: Z forward (depth). 
    # If looking down at table: Z points -Z_world.
    # Base_link Z is Up.
    # Rotation (Roll=0, Pitch=90deg) -> Z points forward (X_world).
    # Pitch=180 -> Z points down.
    # Let's just assume identity for connection to catch the data. 
    # The pipeline transforms to 'base_link'.
    # Default params in pipeline use 'base_link'.
    # If we don't publish TF, depth_matcher transform lookup fails properly (or waits).
    # We need a transform from 'kinect2_rgb_optical_frame' (mock) to 'base_link'.
    # Let's define it.
    
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.5', '0', '1.0', '0.0', '0.707', '0.0', '0.707', 'base_link', 'kinect2_rgb_optical_frame'],
        output='screen'
    )
    # Quaternion: 90 deg around Y? 
    # If not perfect, 3D points will just be rotated. Path Generator should still work (PCA).
    
    # 4. Path Checker (Verification Oracle)
    # Checks if /vision/welding_path receives message
    path_checker = Node(
        package='parol6_vision',
        executable='check_path.py', 
        name='path_checker',
        output='screen'
    )

    return LaunchDescription([
        static_tf,
        mock_camera_node,
        pipeline_launch,
        # Delay checker to allow pipeline to startup
        TimerAction(
            period=5.0,
            actions=[path_checker]
        )
    ])
