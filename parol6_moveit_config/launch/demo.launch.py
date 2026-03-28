import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    use_sim_time = LaunchConfiguration("use_sim_time")

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

    # For fake hardware execution, we must swap the hardcoded Gazebo plugin with the mock system.
    fake_robot_description = {
        "robot_description": moveit_config.robot_description["robot_description"].replace(
            "ign_ros2_control/IgnitionSystem", 
            "mock_components/GenericSystem"
        )
    }

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            {**moveit_config.to_dict(), **fake_robot_description},
            {"use_sim_time": use_sim_time},
        ],
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
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[
            fake_robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": use_sim_time},
        ],
    )

    # Static TF
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
        condition=IfCondition(use_fake_hardware),
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[fake_robot_description],
        condition=IfCondition(use_fake_hardware),
    )

    # ros2_control using FakeSystem as hardware
    ros2_controllers_path = os.path.join(
        get_package_share_directory("parol6_moveit_config"),
        "config",
        "ros2_controllers_sim.yaml",
    )
    
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[fake_robot_description, ros2_controllers_path],
        output="both",
        condition=IfCondition(use_fake_hardware),
    )

    # Load controllers
    load_controllers = []
    for controller in ["parol6_arm_controller", "joint_state_broadcaster"]:
        load_controllers += [
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller],
                output="screen",
                condition=IfCondition(use_fake_hardware),
            )
        ]

    # Camera TF — connects base_link to the root of kinect2_bridge's TF tree
    publish_camera_tf = LaunchConfiguration("publish_camera_tf")

    # Static TF (Base Link -> Kinect Root Frame)
    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_camera',
        # Connects base_link → 'kinect2' (the root of kinect2_bridge's internal TF chain).
        # kinect2_bridge with base_name_tf=kinect2 owns: kinect2→kinect2_link→optical frames.
        # Targeting 'kinect2' (not 'kinect2_link') avoids a duplicate-parent TF conflict.
        arguments=['--x', '0.646', '--y', '0.1225', '--z', '1.015',
                   '--yaw', '1.603684', '--pitch', '0.0', '--roll', '-3.14159',
                   '--frame-id', 'base_link', '--child-frame-id', 'kinect2'],
        output='screen',
        condition=IfCondition(publish_camera_tf),
    )

    # Static TF (Kinect Link -> RGB Optical Frame)
    static_tf_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_optical',
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.0',
                   '--roll', '-1.5708', '--pitch', '0.0', '--yaw', '-1.5708',
                   '--frame-id', 'kinect2_link', '--child-frame-id', 'kinect2_rgb_optical_frame'],
        output='screen',
        condition=IfCondition(publish_camera_tf),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_fake_hardware",
                default_value="true",
                description="Start internal ros2_control stack (true) or use external controllers like Gazebo (false).",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation (Gazebo) clock if true",
            ),
            DeclareLaunchArgument(
                "publish_camera_tf",
                default_value="true",
                description="Publish static TF for the camera relative to base_link",
            ),
            move_group_node,
            rviz_node,
            static_tf_node,
            robot_state_publisher,
            ros2_control_node,
            static_tf_camera,
            static_tf_optical,
        ]
        + load_controllers
    )
