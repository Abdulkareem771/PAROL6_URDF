from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Package directories
    parol6_pkg = FindPackageShare('parol6')
    moveit_config_pkg = FindPackageShare('parol6_moveit_config')
    
    # URDF file
    urdf_file = os.path.join(get_package_share_directory('parol6'), 'urdf', 'PAROL6.urdf')
    with open(urdf_file, 'r') as file:
        robot_description = file.read()
    
    # SRDF file  
    srdf_file = os.path.join(get_package_share_directory('parol6_moveit_config'), 'config', 'parol6.srdf')
    with open(srdf_file, 'r') as file:
        robot_description_semantic = file.read()
    
    # Servo parameters
    servo_params = {
        'move_group_name': 'parol6_arm',
        'command_out_topic': '/parol6_arm_controller/joint_trajectory',
        'planning_frame': 'link_base',
        'ee_frame_name': 'link_tool0',
        'publish_period': 0.01,
        'check_collisions': True,
        'collision_check_rate': 10.0,
        'scale.linear': 0.1,
        'scale.angular': 0.3,
        'scale.joint': 0.5,
        'robot_description': robot_description,
        'robot_description_semantic': robot_description_semantic,
        'use_sim_time': True
    }
    
    return LaunchDescription([
        # Servo Node
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
        
        # Joy Node
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{'dev': '/dev/input/js0', 'deadzone': 0.1}]
        ),
        
        # Xbox Bridge
        Node(
            package='parol6_moveit_config', 
            executable='xbox_to_servo.py',
            name='xbox_to_servo',
            output='screen'
        )
    ])
