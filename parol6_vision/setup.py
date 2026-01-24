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
        ],
    },
    scripts=['test/mock_camera_publisher.py', 'test/check_path.py'],
)
