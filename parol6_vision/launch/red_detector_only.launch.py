from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_share = FindPackageShare('parol6_vision')
    
    config_file = PathJoinSubstitution([pkg_share, 'config', 'detection_params.yaml'])
    
    return LaunchDescription([
        Node(
            package='parol6_vision',
            executable='red_line_detector',
            name='red_line_detector',
            parameters=[config_file, {'publish_debug_images': True}],
            output='screen',
            remappings=[
                ('/vision/weld_lines_2d', '/vision/weld_lines_2d'),
                ('/kinect2/qhd/image_color_rect', '/kinect2/qhd/image_color_rect')
            ]
        )
    ])
