#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import sys

class PathChecker(Node):
    def __init__(self):
        super().__init__('path_checker')
        self.sub = self.create_subscription(Path, '/vision/welding_path', self.callback, 10)
        self.get_logger().info('Path Checker Waiting for /vision/welding_path...')

    def callback(self, msg):
        if len(msg.poses) > 0:
            self.get_logger().info('SUCCESS: Verified Path Generator Output!')
            self.get_logger().info(f'Path has {len(msg.poses)} waypoints.')
            sys.exit(0)

def main():
    rclpy.init()
    node = PathChecker()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
