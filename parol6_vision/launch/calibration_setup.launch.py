import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # =========================================================================
    # 1. CALIBRATION RESULTS (Replace these with your actual measurements)
    # =========================================================================
    
    # Camera -> Marker calibration result (2026-03-28)
    # Translation: X=0.0193m, Y=0.0149m, Z=0.7888m  |  Euler: Roll=179.95, Pitch=-6.59, Yaw=87.63
    # Format: x y z qx qy qz qw child parent
    base_to_cube = ["0.0193", "0.0149", "0.7888", "-0.7204", "-0.6911", "-0.0418", "0.0395", "kinect2_ir_optical_frame", "detected_marker_frame"]

    # Transform from ROBOT BASE to KINECT (The result from Phase 4 script)
    # This is the "Gold" value that aligns your 3D world.
    base_to_camera = ["1.24", "-0.05", "1.42", "0.0", "0.707", "0.0", "0.707", "base_link", "kinect_link"]

    # =========================================================================
    # 2. NODES
    # =========================================================================

    # Aruco Detection Node
    aruco_node = Node(
        package='aruco_ros',
        executable='single',
        name='aruco_single',
        parameters=[{
            'marker_id': 0,
            'marker_size': 0.1,  # Change to your measured size in meters
            'marker_frame': "detected_cube_marker",
            'reference_frame': "kinect_rgb_optical_frame",
            'corner_refinement': "SUBPIX",
            'marker_dict': "DICT_4X4_50"  # Based on the image you provided
        }],
        remappings=[
            ('/image', '/kinect/rgb/image_rect_color'),
            ('/camera_info', '/kinect/rgb/camera_info')
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