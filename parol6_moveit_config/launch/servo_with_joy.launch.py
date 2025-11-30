import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError:
        return None

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

def generate_launch_description():
    # Get URDF and SRDF
    robot_description_config = load_file('parol6', 'urdf/PAROL6.urdf')
    robot_description = {'robot_description': robot_description_config}

    robot_description_semantic_config = load_file('parol6_moveit_config', 'config/parol6.srdf')
    robot_description_semantic = {'robot_description_semantic': robot_description_semantic_config}

    # Servo Config
    servo_yaml = load_yaml('parol6_moveit_config', 'config/parol6_servo.yaml')
    servo_params = {'moveit_servo': servo_yaml}

    # Joy Node - Reads Xbox controller input
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'device_id': 0,
            'deadzone': 0.1,
            'autorepeat_rate': 20.0,
        }],
        output='screen',
    )

    # Xbox to Servo Bridge - Converts joy messages to servo twist commands
    xbox_to_servo_script = os.path.join(
        get_package_share_directory('parol6_moveit_config'),
        'scripts',
        'xbox_to_servo.py'
    )
    
    xbox_to_servo_node = Node(
        executable='python3',
        arguments=[xbox_to_servo_script],
        name='xbox_to_servo',
        output='screen',
    )

    # Servo Node - MoveIt Servo for smooth, safe motion
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        name='servo_node',
        parameters=[
            servo_params,
            robot_description,
            robot_description_semantic,
            {'use_sim_time': True}
        ],
        output='screen',
    )

    return LaunchDescription([
        joy_node,
        xbox_to_servo_node,
        servo_node,
    ])
