import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    use_sim_time = LaunchConfiguration("use_sim_time")
    
    # ── Vision Launch Arguments ───────────────────────────────────────────────
    use_bag = LaunchConfiguration('use_bag')
    single_frame_detection = LaunchConfiguration('single_frame_detection')
    camera_frame = LaunchConfiguration('camera_frame')

    pkg_vision = get_package_share_directory('parol6_vision')

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

    # RViz (Updated to vision_debug.rviz to include vision overlays)
    rviz_config_file = os.path.join(
        pkg_vision,
        "config",
        "vision_debug.rviz"
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

    # Static TF (Kinect Root -> Kinect Link)
    static_tf_link = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_kinect_link',
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.0',
                   '--yaw', '0.0', '--pitch', '0.0', '--roll', '0.0',
                   '--frame-id', 'kinect2', '--child-frame-id', 'kinect2_link'],
        output='screen',
        condition=IfCondition(publish_camera_tf),
    )

    # Static TF (Kinect Link -> RGB Optical Frame) - Adjusted to vision_moveit positions
    static_tf_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_optical',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '-1.5708', '--pitch', '0', '--yaw', '-1.5708',
                   '--frame-id', 'kinect2_link', '--child-frame-id', camera_frame],
        output='screen',
        condition=IfCondition(publish_camera_tf),
    )

    # ── Bag Player ────────────────────────────────────
    bag_path = '/workspace/rosbag2_2026_01_26-23_26_59'
    play_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--loop',
             '--remap', '/tf_static:=/tf_static_bag_discard',
             '--remap', '/tf:=/tf_bag_discard',
             '--clock'],
        output='screen',
        condition=IfCondition(use_bag)
    )

    # ── Vision Nodes ────────────────────────────────────────────────
    path_optimizer = Node(
        package='parol6_vision',
        executable='path_optimizer',
        name='path_optimizer',
        output='screen',
        parameters=[{
            'publish_debug_images': True,
            'use_sim_time': use_sim_time,
        }]
    )

    depth_matcher = Node(
        package='parol6_vision',
        executable='depth_matcher',
        name='depth_matcher',
        output='screen',
        parameters=[{
            'sync_time_tolerance': 0.5,
            'min_depth_quality': 0.05,
            'max_depth': 5000.0,
            'min_depth': 100.0,
            'min_valid_points': 2,
            'use_sim_time': use_sim_time,
        }]
    )

    path_generator = Node(
        package='parol6_vision',
        executable='path_generator',
        name='path_generator',
        output='screen',
        parameters=[{
            'spline_degree': 3,
            'spline_smoothing': 0.005,
            'waypoint_spacing': 0.005,
            'approach_angle_deg': 45.0,
            'auto_generate': True,
            'min_points_for_path': 3,
            'use_sim_time': use_sim_time,
        }]
    )

    vision_trajectory_executor = Node(
        package='parol6_vision',
        executable='vision_trajectory_executor',
        name='vision_trajectory_executor',
        output='screen',
        parameters=[{
            'planning_group': 'parol6_arm',
            'base_frame': 'base_link',
            'end_effector_link': 'L6',
            'step_size': 0.02,
            'jump_threshold': 1.5,
            'auto_execute': True,
            'use_sim_time': use_sim_time,
        }],
    )

    point_cloud_xyzrgb = Node(
        package='depth_image_proc',
        executable='point_cloud_xyzrgb_node',
        name='point_cloud_xyzrgb',
        output='screen',
        remappings=[
            ('/rgb/camera_info',             '/kinect2/sd/camera_info'),
            ('/rgb/image_rect_color',        '/kinect2/sd/image_color_rect'),
            ('/depth_registered/image_rect', '/kinect2/sd/image_depth_rect'),
        ],
        parameters=[{'use_sim_time': use_sim_time}]
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
            DeclareLaunchArgument(
                'use_bag', default_value='true',
                description='true = replay bag; false = use live Kinect'
            ),
            DeclareLaunchArgument(
                'single_frame_detection', default_value='true',
                description='true = process one camera frame then stop detector subscription'
            ),
            DeclareLaunchArgument(
                'camera_frame', default_value='kinect2_rgb_optical_frame',
                description='The TF frame ID for the camera optical frame'
            ),
            move_group_node,
            rviz_node,
            static_tf_node,
            robot_state_publisher,
            ros2_control_node,
            static_tf_camera,
            static_tf_link,
            static_tf_optical,
            play_bag,
            path_optimizer,
            depth_matcher,
            path_generator,
            vision_trajectory_executor,
            point_cloud_xyzrgb,
        ]
        + load_controllers
    )
