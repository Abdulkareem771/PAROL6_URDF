from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package paths
    parol6_package_path = get_package_share_directory('parol6')
    controller_config_path = os.path.join(parol6_package_path, 'config', 'ros2_controllers.yaml')
    urdf_path = os.path.join(parol6_package_path, 'urdf', 'PAROL6.urdf')
    
    # Read URDF
    with open(urdf_path, 'r') as urdf_file:
        robot_description = urdf_file.read()

    # Set IGN_GAZEBO_RESOURCE_PATH to help Ignition find meshes
    # parol6_package_path is /workspace/install/parol6/share/parol6
    # We need /workspace/install/parol6/share
    set_ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=os.path.dirname(parol6_package_path)
    )


    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }]
    )

    # Launch Ignition Gazebo
    ignition_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_ign_gazebo'),
                'launch',
                'ign_gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'ign_args': '-r -v 4 empty.sdf'
        }.items()
    )


    # Spawn robot in Ignition
    spawn_robot = Node(
        package='ros_ign_gazebo',
        executable='create',
        arguments=[
            '-name', 'parol6',
            '-topic', 'robot_description',
            '-z', '0.0',
        ],
        output='screen'
    )

    # Bridge for clock
    bridge_clock = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen'
    )

    # Load controllers
    load_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
        output="screen",
    )

    load_joint_trajectory_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["parol6_arm_controller", "-c", "/controller_manager"],
        output="screen",
    )

    return LaunchDescription([
        set_ign_resource_path,
        robot_state_publisher_node,
        ignition_gazebo,
        spawn_robot,
        bridge_clock,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_robot,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_joint_trajectory_controller],
            )
        ),
    ])

