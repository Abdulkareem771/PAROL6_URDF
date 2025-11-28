from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # ROS Bridge Server for WebSocket connections
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_websocket',
            parameters=[{'port': 9090}]
        ),
        
        # Web Video Server for camera streams
        Node(
            package='web_video_server',
            executable='web_video_server',
            name='web_video_server'
        ),
        
        # Mobile control bridge
        Node(
            package='mobile_control',
            executable='mobile_bridge',
            name='mobile_bridge'
        ),
    ])
