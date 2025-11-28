from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get configuration directory paths
    parol6_package_path = get_package_share_directory('parol6')
    controller_config_path = os.path.join(parol6_package_path, 'config', 'ros2_controllers.yaml')
    
    # Use Gazebo-specific URDF
    urdf_file = os.path.join(parol6_package_path, 'urdf', 'PAROL6_gazebo.urdf')
    
    # Argument for setting the initial position (optional)
    initial_joint_states = DeclareLaunchArgument(
        'initial_joint_states',
        default_value='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
        description='Initial joint configuration for the robot.'
    )

    # 1. Launch Gazebo and the default empty world
    gazebo = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare("gazebo_ros"),
            "launch",
            "gazebo.launch.py",
        ]),
    )

    # 2. Start the Robot State Publisher to broadcast TF frames
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': open(urdf_file).read(),
        }]
    )

    # 3. Spawn the robot model in Gazebo using the URDF
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'parol6'],
        output='screen'
    )

    # 4. Load and start the controllers (Controller Manager)
    load_controllers = [
        # Load the joint state broadcaster
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
        ),
        # Load the MoveIt-compatible Joint Trajectory Controller
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["parol6_arm_controller", "-c", "/controller_manager"],
        ),
    ]

    return LaunchDescription([
        initial_joint_states,
        gazebo,
        robot_state_publisher_node,
        spawn_entity,
    ] + load_controllers)
