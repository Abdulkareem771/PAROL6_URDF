from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="moveit_servo",
            executable="servo_node",
            name="servo_node", 
            output="screen",
            parameters=[os.path.join("/workspace/parol6_moveit_config/config", "parol6_servo.yaml")],
            remappings=[
                ("/servo_node/delta_twist_cmds", "/servo/delta_twist_cmds"),
                ("/servo_node/delta_joint_cmds", "/servo/delta_joint_cmds"),
            ]
        ),
        Node(
            package="joy", 
            executable="joy_node",
            name="joy_node",
            parameters=[{"dev": "/dev/input/js0", "deadzone": 0.1}]
        ),
        Node(
            package="parol6_control",
            executable="xbox_to_servo.py",
            name="xbox_to_servo",
            output="screen"
        )
    ])
