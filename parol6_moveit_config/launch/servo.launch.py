import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro

def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
        return None

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
        return None

def generate_launch_description():
    # Get URDF and SRDF
    # Note: We assume the URDF is in PAROL6 package
    # We need to process xacro if it is a xacro file, but here it is a URDF
    
    # URDF
    robot_description_config = load_file('parol6', 'urdf/PAROL6.urdf')
    robot_description = {'robot_description': robot_description_config}

    # SRDF
    robot_description_semantic_config = load_file('parol6_moveit_config', 'config/parol6.srdf')
    robot_description_semantic = {'robot_description_semantic': robot_description_semantic_config}

    # Servo Config
    servo_yaml = load_yaml('parol6_moveit_config', 'config/parol6_servo.yaml')
    servo_params = {'moveit_servo': servo_yaml}

    # Start the actual move_group node/action server
    # We assume 'move_group' is already running from start_ignition.sh? 
    # Actually, servo runs standalone usually, but needs move_group for collision checking context.
    
    # Servo Node
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        parameters=[
            servo_params,
            robot_description,
            robot_description_semantic,
            {'use_sim_time': True}
        ],
        output='screen',
    )

    return LaunchDescription([servo_node])
