from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # 1. Joy Node
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'deadzone': 0.05,
                'autorepeat_rate': 20.0,
            }]
        ),
        
        # 2. Industrial Xbox Controller
        Node(
            package='xbox_industrial_controller', # We will run this as an executable script
            executable='xbox_industrial_controller.py', # This assumes it's installed, but we'll run it directly
            name='xbox_controller',
            parameters=[{
                'use_sim_time': True,
                'sensitivity': 0.08,
                'deadzone': 0.15,
                'max_speed': 0.5,
                'control_rate': 10.0,
                'trajectory_duration': 0.2
            }]
        )
    ])
