from setuptools import setup

package_name = 'serial_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Publish serial lines from ESP',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'serial_node = serial_publisher.serial_node:main',
        ],
    },
)
