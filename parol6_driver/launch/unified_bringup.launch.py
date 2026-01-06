import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # --- Arguments ---
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='real',
        description='Operation mode: [real, sim, fake]'
    )

    mode = LaunchConfiguration('mode')

    # --- Configuration ---
    # 1. MoveIt Config (Loads URDF/SRDF automatically)
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

    # --- Nodes ---

    # A. SIMULATION MODE (Ignition Gazebo)
    # Includes existing ignition.launch.py which handles RSP and Bridge
    sim_launch = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('parol6'),
            'launch',
            'ignition.launch.py'
        ]),
        condition=IfCondition(PythonExpression(["'", mode, "' == 'sim'"]))
    )

    # B. REAL/FAKE MODE COMPONENTS
    # These are NOT in ignition.launch.py, so we add them for other modes.

    # B.1. Static TF (World -> Base Link)
    # Required for Real/Fake to link the robot to the world
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
        condition=UnlessCondition(PythonExpression(["'", mode, "' == 'sim'"]))
    )

    # B.2. Robot State Publisher
    # Ign launch has its own, so we only run this for real/fake
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
        condition=UnlessCondition(PythonExpression(["'", mode, "' == 'sim'"]))
    )

    # C. REAL ROBOT DRIVER
    driver_node = Node(
        package='parol6_driver',
        executable='real_robot_driver',
        output='screen',
        condition=IfCondition(PythonExpression(["'", mode, "' == 'real'"]))
    )

    # D. MOVEIT (Planner)
    # We pass use_sim_time based on mode condition
    # Using PythonExpression to determine boolean
    
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {'use_sim_time': PythonExpression(["'", mode, "' == 'sim'"])}
        ],
    )

    # E. RVIZ
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
            {'use_sim_time': PythonExpression(["'", mode, "' == 'sim'"])}
        ],
    )

    return LaunchDescription([
        mode_arg,
        LogInfo(msg=["Starting PAROL6 Unified Bringup in mode: ", mode]),
        sim_launch,
        static_tf,
        rsp,
        driver_node,
        move_group_node,
        rviz_node
    ])
