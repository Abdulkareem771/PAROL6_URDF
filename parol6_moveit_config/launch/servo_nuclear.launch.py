import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Read robot description files
    urdf_path = '/workspace/install/parol6/share/parol6/urdf/PAROL6.urdf'
    srdf_path = '/workspace/install/parol6_moveit_config/share/parol6_moveit_config/config/parol6.srdf'
    yaml_path = '/workspace/install/parol6_moveit_config/share/parol6_moveit_config/config/servo_nuclear.yaml'
    
    with open(urdf_path, 'r') as f:
        robot_description = f.read()
    with open(srdf_path, 'r') as f:
        robot_description_semantic = f.read()

    return LaunchDescription([
        Node(
            package='moveit_servo',
            executable='servo_node_main',
            name='servo_node',
            output='screen',
            parameters=[
                yaml_path,
                {'robot_description': robot_description},
                {'robot_description_semantic': robot_description_semantic},
                {'use_sim_time': True}
            ]
        ),
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{'device_id': 0, 'deadzone': 0.1}]
        ),
        Node(
            package='parol6_moveit_config',
            executable='xbox_to_servo.py',
            name='xbox_to_servo',
            output='screen'
        )
    ])
