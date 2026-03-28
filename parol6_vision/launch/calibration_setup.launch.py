import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # =========================================================================
    # 1. CALIBRATION RESULTS (Replace these with your actual measurements)
    # =========================================================================
    
    # Transform from ROBOT BASE to CALIBRATION CUBE (Your manual measurement)
    # Format: x y z qx qy qz qw child parent
    base_to_cube = ["0.5", "0.0", "0.0", "0.0", "0.0", "0.0", "1.0", "base_link", "cube_marker_target"]

    # Transform from ROBOT BASE to KINECT (The result from Phase 4 script)
    # This is the "Gold" value that aligns your 3D world.
    base_to_camera = ["0.5550", "0.1777", "1.0016", "0.7078", "0.7058", "0.0269", "0.0123", "base_link", "kinect2_link"]

    # =========================================================================
    # 2. NODES
    # =========================================================================

    # Aruco Detection Node
    aruco_node = Node(
        package='aruco_ros',
        executable='single',
        name='aruco_single',
        parameters=[{
            'marker_id': 6,
            'marker_size': 0.04575,  # Change to your measured size in meters
            'marker_frame': "detected_marker_frame",
            'reference_frame': "kinect2_ir_optical_frame",
            'corner_refinement': "SUBPIX",
            'marker_dict': "DICT_ARUCO_ORIGINAL"  # Based on the image you provided
        }],
        remappings=[
            ('/image', '/kinect2/sd/image_color_rect'),
            ('/camera_info', '/kinect2/sd/camera_info')
        ]
    )

    # Static Transform: Base to Camera (The result of your project)
    static_tf_base_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_kinect_publisher',
        arguments=base_to_camera
    )

    # Static Transform: Base to Cube (To verify alignment in RViz)
    static_tf_base_cube = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_cube_publisher',
        arguments=base_to_cube
    )

    return LaunchDescription([
        aruco_node,
        static_tf_base_camera,
        static_tf_base_cube
    ])