import sys
import os

try:
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, RegisterEventHandler, IncludeLaunchDescription
    from launch.conditions import IfCondition
    from launch.event_handlers import OnProcessExit
    from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
    from launch.launch_description_sources import PythonLaunchDescriptionSource

    from launch_ros.actions import Node
    from launch_ros.substitutions import FindPackageShare
    from launch_ros.parameter_descriptions import ParameterValue

    import os
    from ament_index_python.packages import get_package_share_directory
    from moveit_configs_utils import MoveItConfigsBuilder
    
    # Just checking instantiation
    moveit_config = (
        MoveItConfigsBuilder("parol6", package_name="parol6_moveit_config")
        .robot_description_semantic(file_path="config/parol6.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    d = moveit_config.to_dict()
    print("MoveItConfigsBuilder dict generated successfully")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
