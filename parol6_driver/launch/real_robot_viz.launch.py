import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # Load the robot configuration using MoveItConfigsBuilder
    # This automatically loads URDF, SRDF, OMPL, Kinematics, and Controllers
    moveit_config = (
        MoveItConfigsBuilder("parol6")
        .robot_description(file_path=os.path.join(
            get_package_share_directory("parol6"),
            "urdf",
            "PAROL6.urdf"
        ))
        .robot_description_semantic(file_path=os.path.join(
            get_package_share_directory("parol6_moveit_config"),
            "config",
            "parol6.srdf"
        ))
        .trajectory_execution(file_path=os.path.join(
            get_package_share_directory("parol6_moveit_config"),
            "config",
            "moveit_controllers.yaml"
        ))
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # 1. Move Group Node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # 2. RViz Node
    rviz_config_file = os.path.join(
        get_package_share_directory("parol6_moveit_config"),
        "config",
        "moveit.rviz"
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            {'use_sim_time': False},
        ],
    )

    # 3. Static TF (World -> Base Link)
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
    )

    # 4. Robot State Publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # 5. Real Robot Driver (The Custom Logic)
    driver_node = Node(
        package='parol6_driver',
        executable='real_robot_driver',
        output='screen',
    )

    return LaunchDescription([
        static_tf_node,
        robot_state_publisher,
        driver_node,
        move_group_node,
        rviz_node,
    ])
