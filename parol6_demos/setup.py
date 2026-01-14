from setuptools import setup
import os
from glob import glob

package_name = 'parol6_demos'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kareem',
    maintainer_email='your.email@example.com',
    description='Demo applications for PAROL6 vision-guided welding system',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'demo_cartesian_path = parol6_demos.demo_cartesian_path:main',
        ],
    },
)
