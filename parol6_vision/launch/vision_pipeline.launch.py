from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Package share directory
    pkg_share = FindPackageShare('parol6_vision')
    
    # Launch arguments
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to launch RViz for visualization'
    )
    
    # Config files  
    detection_params = PathJoinSubstitution([pkg_share, 'config', 'detection_params.yaml'])
    camera_params = PathJoinSubstitution([pkg_share, 'config', 'camera_params.yaml'])
    path_params = PathJoinSubstitution([pkg_share, 'config', 'path_params.yaml'])
    
    # 1. Red Line Detector Node
    detector_node = Node(
        package='parol6_vision',
        executable='red_line_detector',
        name='red_line_detector',
        parameters=[detection_params, camera_params],
        output='screen'
    )
    
    # 2. Depth Matcher Node
    matcher_node = Node(
        package='parol6_vision',
        executable='depth_matcher',
        name='depth_matcher',
        parameters=[detection_params, camera_params],
        output='screen'
    )

    # 3. Path Generator Node
    generator_node = Node(
        package='parol6_vision',
        executable='path_generator',
        name='path_generator',
        parameters=[path_params],
        output='screen'
    )

    # 4. Capture Images Node — publishes to raw topic; crop node relays downstream
    capture_node = Node(
        package='parol6_vision',
        executable='capture_images',
        name='capture_images',
        parameters=[{'output_topic': '/vision/captured_image_raw'}],
        output='screen',
    )

    # 5. Crop Image Node (always-active) — auto-loads ~/.parol6/crop_config.json
    crop_node = Node(
        package='parol6_vision',
        executable='crop_image',
        name='crop_image',
        output='screen',
    )

    # RViz Node (Optional)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        # arguments=['-d', PathJoinSubstitution([pkg_share, 'config', 'vision.rviz'])],
        condition=LaunchConfiguration('use_rviz'), # This line handles the conditional launch
        output='screen'
    )

    # 6. Optional: Camera Setup (TFs, Robot State Publisher)
    # This solves the "missing TF" issue for depth_matcher
    launch_setup_arg = DeclareLaunchArgument(
        'launch_setup',
        default_value='true',
        description='Whether to launch the camera/robot setup (TF, Rsp)'
    )
    
    camera_setup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_share, 'launch', 'camera_setup.launch.py'])
        ]),
        launch_arguments={'use_rviz': LaunchConfiguration('use_rviz')}.items(),
        condition=IfCondition(LaunchConfiguration('launch_setup'))
    )

    return LaunchDescription([
        use_rviz_arg,
        launch_setup_arg,
        camera_setup_launch,
        capture_node,
        crop_node,      # always-active crop relay
        detector_node,
        matcher_node,
        generator_node,
        # rviz_node (Handled by camera_setup if launch_setup is true)
    ])
