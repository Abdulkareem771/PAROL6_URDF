import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
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

    # 1. Move Group (Allows interactive markers to work)
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # 2. RViz with Vision Config
    pkg_vision = get_package_share_directory('parol6_vision')
    rviz_config_file = os.path.join(pkg_vision, 'config', 'vision_debug.rviz')
    
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
    static_tf_world = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher_world",
        output="log",
        arguments=["--x", "0.0", "--y", "0.0", "--z", "0.0", "--yaw", "0.0", "--pitch", "0.0", "--roll", "0.0", "--frame-id", "world", "--child-frame-id", "base_link"],
    )

    # 4. Static TF (Camera -> Base Link)
    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_camera',
        arguments=['--x', '0.5', '--y', '0.0', '--z', '1.0', '--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5', '--frame-id', 'base_link', '--child-frame-id', 'kinect2_rgb_optical_frame'],
        output='screen'
    )

    # 5. Robot State Publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # 6. Joint State Publisher (Standard Fallback)
    # Since 'joint_state_publisher_gui' is missing, we use a simple script to publish zero states
    # This ensures the robot model is visible in RViz
    joint_state_publisher = Node(
        package="parol6_vision",
        executable="dummy_joint_publisher",
        name="dummy_joint_publisher",
        output="screen",
    )

    # 7. ros2_control (Optional - kept for consistency but likely crashing without hardware)
    ros2_controllers_path = os.path.join(
        get_package_share_directory("parol6_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )
    
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[moveit_config.robot_description, ros2_controllers_path],
        output="both",
    )

    load_controllers = []
    for controller in ["parol6_arm_controller", "joint_state_broadcaster"]:
        load_controllers += [
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller],
                output="screen",
            )
        ]

    return LaunchDescription([
        static_tf_world,
        static_tf_camera,
        robot_state_publisher,
        joint_state_publisher,
        move_group_node,
        ros2_control_node,
        rviz_node,
    ] + load_controllers)
