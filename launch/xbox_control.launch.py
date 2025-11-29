from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Joy node for Xbox controller
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'dev': '/dev/input/js0',
                'deadzone': 0.1,
                'autorepeat_rate': 20.0
            }]
        ),
        
        # Xbox controller node
        Node(
            package='parol6',
            executable='xbox_controller_node.py',
            name='xbox_controller_node',
            output='screen',
            emulate_tty=True
        )
    ])
