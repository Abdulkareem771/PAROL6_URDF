from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
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
    
    # 2. Depth Matcher Node (Commented out until implemented)
    # depth_matcher_node = Node(
    #     package='parol6_vision',
    #     executable='depth_matcher',
    #     name='depth_matcher',
    #     parameters=[detection_params, camera_params],
    #     output='screen'
    # )

    # RViz Node (Optional)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        # arguments=['-d', PathJoinSubstitution([pkg_share, 'config', 'vision.rviz'])],
        condition=LaunchConfiguration('use_rviz'), # This line handles the conditional launch
        output='screen'
    )

    return LaunchDescription([
        use_rviz_arg,
        detector_node,
        # depth_matcher_node,
        # rviz_node
    ])
