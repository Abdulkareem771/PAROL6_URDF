import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
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

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # RViz
    rviz_config_file = os.path.join(
        get_package_share_directory("parol6_moveit_config"),
        "rviz",
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
        ],
    )

    # Static TF
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
    )

    return LaunchDescription(
        [
            move_group_node,
            rviz_node,
            static_tf_node,
            # The real-hardware parent launch already owns ros2_control,
            # controller spawners, and robot_state_publisher.
        ]
    )
