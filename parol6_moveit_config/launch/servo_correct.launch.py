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

    # Load servo config - this now has ros__parameters at root
    servo_yaml = load_yaml('parol6_moveit_config', 'config/parol6_servo.yaml')
    
    # Combine all parameters - servo_yaml already has ros__parameters at root
    servo_params = {}
    servo_params.update(servo_yaml)  # This contains ros__parameters
    servo_params.update(robot_description)
    servo_params.update(robot_description_semantic)
    servo_params['use_sim_time'] = True

    # Servo Node
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        parameters=[servo_params],
        output='screen',
    )

    # Joy node
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{'dev': '/dev/input/js0', 'deadzone': 0.1}]
    )

    # Xbox to Servo bridge
    xbox_node = Node(
        package='parol6_moveit_config',
        executable='xbox_to_servo.py',
        name='xbox_to_servo',
        output='screen'
    )

    return LaunchDescription([servo_node, joy_node, xbox_node])
