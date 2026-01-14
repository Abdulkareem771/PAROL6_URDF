#!/usr/bin/env python3
"""
Launch file for PAROL6 with ros2_control
Day 1: SIL (Software-in-the-Loop) Validation

Purpose:
- Load robot description with ros2_control tags
- Start controller_manager
- Load and activate controllers
- Validate ROS plumbing without hardware

Usage:
  ros2 launch parol6_hardware real_robot.launch.py

Expected outcome (Day 1):
- Controllers activate successfully
- /joint_states publishes at 25Hz (with zeros)
- No crashes or errors
- Clean shutdown with Ctrl+C
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Declare launch arguments
    declared_arguments = []
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "runtime_config_package",
            default_value="parol6_hardware",
            description="Package with controller configuration",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "controllers_file",
            default_value="parol6_controllers.yaml",
            description="Controller configuration file",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="parol6_hardware",
            description="Package with robot URDF/xacro",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="parol6.urdf.xacro",
            description="URDF/xacro description file",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation time",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "use_ros2_control",
            default_value="true",
            description="Enable ros2_control",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "serial_port",
            default_value="/dev/ttyUSB0",
            description="Serial port for ESP32",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "baud_rate",
            default_value="115200",
            description="Baud rate for serial communication",
        )
    )

    # Initialize arguments
    runtime_config_package = LaunchConfiguration("runtime_config_package")
    controllers_file = LaunchConfiguration("controllers_file")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_ros2_control = LaunchConfiguration("use_ros2_control")
    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")

    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("parol6_hardware"), "urdf", "parol6.urdf.xacro"]
            ),
            " ",
            "use_ros2_control:=",
            use_ros2_control,
            " ",
            "serial_port:=",
            serial_port,
            " ",
            "baud_rate:=",
            baud_rate,
        ]
    )
    
    # Wrap in ParameterValue to avoid YAML parsing error
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # Controller configuration file path
    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare(runtime_config_package),
            "config",
            controllers_file,
        ]
    )

    # =========================================================================
    # NODE 1: Controller Manager
    # =========================================================================
    # Manages hardware interface lifecycle and controllers
    # Reads robot_description to find <ros2_control> tags
    # Loads hardware plugin (parol6_hardware/PAROL6System)
    
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, robot_controllers],
        output="both",
        emulate_tty=True,  # Better log formatting
    )

    # =========================================================================
    # NODE 2: Robot State Publisher
    # =========================================================================
    # Publishes robot transforms (/tf) based on joint states
    # Subscribes to /joint_states (from joint_state_broadcaster)
    
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )

    # =========================================================================
    # LAUNCH DESCRIPTION
    # =========================================================================
    nodes = [
        control_node,
        robot_state_pub_node,
    ]

    return LaunchDescription(declared_arguments + nodes)
