from setuptools import setup

package_name = 'mobile_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    py_modules=['mobile_bridge'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kareem',
    maintainer_email='kareem@example.com',
    description='Mobile ROS interface for PAROL6 robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mobile_bridge = mobile_bridge:main',
        ],
    },
)
