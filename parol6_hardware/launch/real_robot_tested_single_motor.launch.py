#!/usr/bin/env python3
"""
Launch file for PAROL6 with ros2_control + MoveIt + RViz
Connects to real hardware via parol6_hardware plugin.

This launch file starts EXACTLY ONE of each node:
  - ros2_control_node (PAROL6Hardware plugin)
  - robot_state_publisher
  - joint_state_broadcaster (spawner)
  - parol6_arm_controller (spawner, chained after JSB)
  - move_group (MoveIt)
  - rviz2
  - static_transform_publisher (world → base_link)

Usage:
  ros2 launch parol6_hardware real_robot.launch.py
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # =====================================================================
    # LAUNCH ARGUMENTS
    # =====================================================================
    declared_arguments = [
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0",
                              description="Serial port for STM32 Black Pill"),
        DeclareLaunchArgument("baud_rate", default_value="115200",
                              description="Baud rate for serial communication"),
    ]

    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")

    # =====================================================================
    # ROBOT DESCRIPTION (xacro → URDF with ros2_control tags)
    # =====================================================================
    # The included PAROL6 base URDF already carries a demo-only ros2_control block
    # (`parol6_ros2_control` with mock_components/GenericSystem). Strip that block
    # from the xacro output so ros2_control_node only sees the real PAROL6Hardware.
    robot_description_content = Command([
        "bash", " -lc ",
        "'",
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([FindPackageShare("parol6_hardware"), "urdf", "parol6.urdf.xacro"]),
        " use_ros2_control:=true",
        " serial_port:=", serial_port,
        " baud_rate:=", baud_rate,
        " | perl -0pe ",
        "\"s#<ros2_control name=\\\"parol6_ros2_control\\\" type=\\\"system\\\">.*?</ros2_control>##s\"",
        "'",
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # =====================================================================
    # MOVEIT CONFIG (for move_group + rviz)
    # =====================================================================
    moveit_config = (
        MoveItConfigsBuilder("parol6")
        .robot_description(file_path=os.path.join(
            get_package_share_directory("parol6"),
            "urdf", "PAROL6.urdf"
        ))
        .robot_description_semantic(file_path=os.path.join(
            get_package_share_directory("parol6_moveit_config"),
            "config", "parol6.srdf"
        ))
        .trajectory_execution(file_path=os.path.join(
            get_package_share_directory("parol6_moveit_config"),
            "config", "moveit_controllers.yaml"
        ))
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # Controller configuration
    # Method 5 is the legacy single-motor-tested path and should keep using its
    # dedicated controller tolerances rather than the generic real-hardware file.
    robot_controllers = PathJoinSubstitution([
        FindPackageShare("parol6_moveit_config"), "config", "ros2_controllers_tested_single_motor.yaml",
    ])

    # =====================================================================
    # NODE 1: ros2_control_node (ONE instance — real hardware)
    # =====================================================================
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, robot_controllers],
        output="both",
        emulate_tty=True,
    )

    # =====================================================================
    # NODE 2: robot_state_publisher (ONE instance)
    # =====================================================================
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # =====================================================================
    # NODE 3: static_transform_publisher (world → base_link)
    # =====================================================================
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
    )

    # =====================================================================
    # SPAWNER: joint_state_broadcaster (ONE spawner)
    # =====================================================================
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                    "--controller-manager", "/controller_manager"],
    )

    # =====================================================================
    # SPAWNER: parol6_arm_controller (ONE spawner, after JSB)
    # =====================================================================
    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["parol6_arm_controller",
                    "--controller-manager", "/controller_manager"],
    )

    # Chain: arm_controller starts AFTER joint_state_broadcaster exits
    delay_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[robot_controller_spawner],
        )
    )

    # =====================================================================
    # NODE 4: move_group (MoveIt motion planning)
    # =====================================================================
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # =====================================================================
    # NODE 5: rviz2
    # =====================================================================
    rviz_config_file = os.path.join(
        get_package_share_directory("parol6_moveit_config"),
        "rviz", "moveit.rviz"
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

    # =====================================================================
    # LAUNCH
    # =====================================================================
    return LaunchDescription(
        declared_arguments + [
            control_node,
            robot_state_pub_node,
            static_tf_node,
            joint_state_broadcaster_spawner,
            delay_controller,
            move_group_node,
            rviz_node,
        ]
    )
