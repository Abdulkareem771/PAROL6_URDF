from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'parol6_vision'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        # Include config files (.yaml and .rviz)
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='PAROL6 Team',
    maintainer_email='your.email@example.com',
    description='Vision-guided welding path detection for PAROL6 robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'red_line_detector = parol6_vision.red_line_detector:main',
            'depth_matcher = parol6_vision.depth_matcher:main',
            'path_generator = parol6_vision.path_generator:main',
            'moveit_controller = parol6_vision.moveit_controller:main',
            'dummy_joint_publisher = parol6_vision.dummy_joint_publisher:main',
            'hsv_inspector = parol6_vision.hsv_inspector_node:main',
            'capture_images = parol6_vision.capture_images_node:main',
            'read_image = parol6_vision.read_image_node:main',
            'yolo_segment = parol6_vision.yolo_segment:main',
            'color_mode = parol6_vision.color_mode:main',
            'path_optimizer = parol6_vision.path_optimizer:main',
            'crop_image = parol6_vision.crop_image_node:main',
            'manual_line = parol6_vision.manual_line_node:main',
            'manual_line_aligner = parol6_vision.manual_line_aligner_node:main',
            'vision_trajectory_executor = parol6_vision.vision_trajectory_executor:main',
            'eye_to_hand_calibrator = parol6_vision.eye_to_hand_calibrator:main',
        ],
    },
    scripts=['test/mock_camera_publisher.py', 'test/check_path.py'],
)
