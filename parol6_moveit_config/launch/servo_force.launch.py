import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. Get URDF and SRDF content
    # We read them directly from the file system
    urdf_path = '/workspace/install/parol6/share/parol6/urdf/PAROL6.urdf'
    srdf_path = '/workspace/install/parol6_moveit_config/share/parol6_moveit_config/config/parol6.srdf'
    
    with open(urdf_path, 'r') as f:
        robot_description = f.read()
    with open(srdf_path, 'r') as f:
        robot_description_semantic = f.read()

    # 2. Define Servo Parameters DIRECTLY in Python
    # This avoids any YAML parsing issues
    servo_params = {
        'move_group_name': 'parol6_arm',  # <--- CRITICAL: This must be correct
        'command_out_topic': '/parol6_arm_controller/joint_trajectory',
        'planning_frame': 'link_base',
        'ee_frame_name': 'link_tool0',
        'robot_link_command_frame': 'link_base',
        
        'incoming_command_topic': '/servo/delta_twist_cmds',
        'incoming_joint_row_topic': '/servo/delta_joint_cmds',
        
        'publish_period': 0.03,
        
        # Scaling
        'scale': {
            'linear': 0.4,
            'rotational': 0.8,
            'joint': 0.5
        },
        
        # Collision checking
        'check_collisions': True,
        'collision_check_rate': 10.0,
        'collision_check_type': 'threshold_distance',
        'self_collision_proximity_threshold': 0.01,
        'scene_collision_proximity_threshold': 0.02,
        
        # Robot model
        'robot_description': robot_description,
        'robot_description_semantic': robot_description_semantic,
        'use_sim_time': True
    }

    # 3. Servo Node
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        name='servo_node',
        output='screen',
        parameters=[servo_params]
    )

    # 4. Joy Node
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{'device_id': 0, 'deadzone': 0.1}]
    )

    # 5. Xbox Bridge
    xbox_bridge = Node(
        package='parol6_moveit_config',
        executable='xbox_to_servo.py',
        name='xbox_to_servo',
        output='screen'
    )

    return LaunchDescription([servo_node, joy_node, xbox_bridge])
