#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CmdPublisher(Node):
    def __init__(self):
        super().__init__('cmd_publisher')
        self.pub = self.create_publisher(String, 'esp_command', 10)

    def publish(self, text):
        msg = String()
        msg.data = text
        self.pub.publish(msg)
        self.get_logger().info(f"Published: {text}")

def main(args=None):
    rclpy.init(args=args)
    node = CmdPublisher()
    try:
        while rclpy.ok():
            s = input("Enter command (ON/OFF/BLINK or q to quit): ").strip()
            if not s:
                continue
            if s.lower() in ('q','quit','exit'):
                break
            node.publish(s.upper())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
