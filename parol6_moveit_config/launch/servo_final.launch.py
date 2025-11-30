import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. Get URDF and SRDF
    # We read them directly from the file system to ensure they are passed correctly
    urdf_path = '/workspace/install/parol6/share/parol6/urdf/PAROL6.urdf'
    srdf_path = '/workspace/install/parol6_moveit_config/share/parol6_moveit_config/config/parol6.srdf'
    
    with open(urdf_path, 'r') as f:
        robot_description = f.read()
    with open(srdf_path, 'r') as f:
        robot_description_semantic = f.read()

    # 2. Servo Parameters
    # We load the YAML file we just created
    servo_yaml_path = '/workspace/install/parol6_moveit_config/share/parol6_moveit_config/config/parol6_servo_fixed.yaml'
    
    # 3. Servo Node
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        name='servo_node',
        output='screen',
        parameters=[
            servo_yaml_path,
            {'robot_description': robot_description},
            {'robot_description_semantic': robot_description_semantic},
            {'use_sim_time': True}
        ]
    )

    # 4. Joy Node
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{'device_id': 0, 'deadzone': 0.1}]
    )

    # 5. Xbox Bridge
    # We run this directly as a python script to avoid package install issues
    xbox_bridge = Node(
        package='parol6_moveit_config',
        executable='xbox_to_servo.py',
        name='xbox_to_servo',
        output='screen'
    )

    return LaunchDescription([servo_node, joy_node, xbox_bridge])
