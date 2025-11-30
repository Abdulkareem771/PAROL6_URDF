from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package share directory
    pkg_path = get_package_share_directory('parol6_moveit_config')
    
    # Explicit servo parameters
    servo_params = {
        'move_group_name': 'parol6_arm',  # EXPLICITLY SET
        'planning_frame': 'link_base',
        'ee_frame_name': 'link_tool0', 
        'command_out_topic': '/parol6_arm_controller/joint_trajectory',
        'publish_period': 0.01,
        'check_collisions': True,
        'collision_check_rate': 10.0,
        'scale': {
            'linear': 0.1,
            'angular': 0.3, 
            'joint': 0.5
        }
    }
    
    return LaunchDescription([
        # Servo node with explicit parameters
        Node(
            package='moveit_servo',
            executable='servo_node_main',
            name='servo_node',
            output='screen',
            parameters=[servo_params],
            remappings=[
                ('/servo_node/delta_twist_cmds', '/servo/delta_twist_cmds'),
                ('/servo_node/delta_joint_cmds', '/servo/delta_joint_cmds'),
            ]
        ),
        
        # Joy node
        Node(
            package='joy', 
            executable='joy_node',
            name='joy_node',
            parameters=[{'dev': '/dev/input/js0', 'deadzone': 0.1}]
        ),
        
        # Xbox to Servo bridge
        Node(
            package='parol6_moveit_config',
            executable='xbox_to_servo.py',
            name='xbox_to_servo',
            output='screen'
        )
    ])
