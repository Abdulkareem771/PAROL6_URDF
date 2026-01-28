#!/usr/bin/env python3
"""
Convert Kinect v2 OpenCV calibration files to ROS camera_params.yaml format

This script reads the calibration files from parol6_vision/data/ and generates
a properly formatted camera_params.yaml for the vision pipeline.

Usage:
    python3 convert_kinect_calibration.py
"""

import yaml
import numpy as np
from scipy.spatial.transform import Rotation

def load_opencv_yaml(filepath):
    """Load OpenCV-style YAML file"""
    with open(filepath, 'r') as f:
        # OpenCV YAML uses custom tags, use safe_load and manually parse
        content = f.read()
        # Remove OpenCV-specific tags for basic parsing
        content = content.replace('!!opencv-matrix', '')
        content = content.replace('%YAML:1.0', '')
        data = yaml.safe_load(content)
    return data

def extract_camera_matrix(calib_color):
    """Extract intrinsic parameters from camera matrix"""
    # Camera matrix format:
    # [fx  0  cx]
    # [0  fy  cy]
    # [0   0   1]
    matrix_data = calib_color['cameraMatrix']['data']
    
    fx = matrix_data[0]    # Element [0,0]
    fy = matrix_data[4]    # Element [1,1]
    cx = matrix_data[2]    # Element [0,2]
    cy = matrix_data[5]    # Element [1,2]
    
    return fx, fy, cx, cy

def extract_distortion(calib_color):
    """Extract distortion coefficients [k1, k2, p1, p2, k3]"""
    dist_data = calib_color['distortionCoefficients']['data']
    return dist_data  # Already in correct order

def rotation_matrix_to_quaternion(rot_matrix):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
    # Reshape flat list to 3x3 matrix
    R = np.array(rot_matrix).reshape(3, 3)
    
    # Use scipy for robust conversion
    r = Rotation.from_matrix(R)
    quat = r.as_quat()  # Returns [x, y, z, w]
    
    return quat.tolist()

def extract_depth_to_color_transform(calib_pose):
    """
    Extract rotation and translation from depth→color calibration
    Note: This is INTRINSIC calibration (depth sensor relative to RGB sensor)
    NOT extrinsic (camera relative to robot base)
    """
    rot_data = calib_pose['rotation']['data']
    trans_data = calib_pose['translation']['data']
    
    # Convert rotation matrix to quaternion
    quat = rotation_matrix_to_quaternion(rot_data)
    
    # Translation is in meters
    translation = {
        'x': trans_data[0],
        'y': trans_data[1],
        'z': trans_data[2]
    }
    
    rotation = {
        'x': quat[0],
        'y': quat[1],
        'z': quat[2],
        'w': quat[3]
    }
    
    return translation, rotation

def create_camera_params_yaml(fx, fy, cx, cy, distortion, 
                                depth_to_color_trans=None, 
                                depth_to_color_rot=None,
                                camera_to_robot_trans=None,
                                camera_to_robot_rot=None):
    """
    Create ROS-compatible camera_params.yaml
    
    Args:
        fx, fy, cx, cy: Intrinsic parameters
        distortion: List of 5 distortion coefficients
        depth_to_color_trans/rot: Depth→RGB transform (optional)
        camera_to_robot_trans/rot: Camera→Robot transform (REQUIRED for vision pipeline)
    """
    
    # Default camera→robot transform if not provided
    # **USER MUST MEASURE/CALIBRATE THIS!**
    if camera_to_robot_trans is None:
        camera_to_robot_trans = {'x': 0.5, 'y': 0.0, 'z': 1.0}
        print("⚠️  WARNING: Using default camera→robot translation!")
        print("   You MUST calibrate this for accurate 3D positioning!")
    
    if camera_to_robot_rot is None:
        # Default: camera looking at robot from side (90° rotated)
        camera_to_robot_rot = {'x': -0.5, 'y': 0.5, 'z': -0.5, 'w': 0.5}
        print("⚠️  WARNING: Using default camera→robot rotation!")
        print("   You MUST calibrate this for accurate 3D positioning!")
    
    params = {
        '/**': {
            'ros__parameters': {
                'camera_intrinsics': {
                    'fx': float(fx),
                    'fy': float(fy),
                    'cx': float(cx),
                    'cy': float(cy),
                    'distortion': [float(d) for d in distortion]
                },
                'camera_frame': 'kinect2_rgb_optical_frame',
                'robot_base_frame': 'base_link',
                'camera_to_base_transform': {
                    'translation': camera_to_robot_trans,
                    'rotation': camera_to_robot_rot
                },
                'depth_range': {
                    'min': 500.0,
                    'max': 2000.0
                }
            }
        }
    }
    
    # Optionally include depth→color transform for reference
    if depth_to_color_trans and depth_to_color_rot:
        params['/**']['ros__parameters']['depth_to_color_transform'] = {
            'translation': depth_to_color_trans,
            'rotation': depth_to_color_rot,
            'note': 'Internal: depth sensor relative to RGB sensor (informational only)'
        }
    
    return params

def main():
    import os
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'parol6_vision', 'data')
    output_file = os.path.join(script_dir, 'parol6_vision', 'config', 'camera_params_calibrated.yaml')
    
    print("="*60)
    print("Kinect v2 Calibration Converter")
    print("="*60)
    
    # Load calibration files
    print("\n[1/4] Loading calibration files...")
    calib_color = load_opencv_yaml(os.path.join(data_dir, 'calib_color.yaml'))
    calib_pose = load_opencv_yaml(os.path.join(data_dir, 'calib_pose.yaml'))
    
    # Extract intrinsics
    print("[2/4] Extracting camera intrinsics...")
    fx, fy, cx, cy = extract_camera_matrix(calib_color)
    distortion = extract_distortion(calib_color)
    
    print(f"   fx: {fx:.2f} pixels")
    print(f"   fy: {fy:.2f} pixels")
    print(f"   cx: {cx:.2f} pixels")
    print(f"   cy: {cy:.2f} pixels")
    print(f"   Distortion: {[f'{d:.4f}' for d in distortion]}")
    
    # Extract depth→color transform (informational)
    print("\n[3/4] Extracting depth→color transform (internal calibration)...")
    depth_to_color_trans, depth_to_color_rot = extract_depth_to_color_transform(calib_pose)
    print(f"   Translation: {depth_to_color_trans}")
    print(f"   Rotation (quat): {depth_to_color_rot}")
    
    # Create params (without camera→robot transform - user must provide)
    print("\n[4/4] Generating camera_params.yaml...")
    params = create_camera_params_yaml(
        fx, fy, cx, cy, distortion,
        depth_to_color_trans, depth_to_color_rot,
        # camera_to_robot_trans and camera_to_robot_rot are None (use defaults)
    )
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("### Camera Parameters Configuration (Auto-Generated)\n")
        f.write("# Generated from Kinect v2 calibration files\n")
        f.write("# Intrinsic parameters: Calibrated ✅\n")
        f.write("# Extrinsic (camera→robot): DEFAULT ⚠️  CALIBRATE THIS!\n\n")
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Generated: {output_file}")
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. ✅ Intrinsic calibration loaded from your files")
    print("2. ⚠️  Camera→Robot transform uses DEFAULT values")
    print("3. 📏 MEASURE camera position relative to robot base:")
    print("      - Translation: (x, y, z) in meters")
    print("      - Rotation: Quaternion or Euler angles")
    print("4. ✏️  Edit camera_params_calibrated.yaml:")
    print("      - Update 'camera_to_base_transform' section")
    print("5. ✅ Copy to camera_params.yaml to activate")
    print("="*60)

if __name__ == '__main__':
    main()
