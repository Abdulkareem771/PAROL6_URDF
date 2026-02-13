#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from parol6_msgs.msg import WeldLineArray
import time

class SyncChecker(Node):
    def __init__(self):
        super().__init__('sync_checker')
        self.latest_stamps = {}
        
        # Subscribe to the same topics as depth_matcher
        self.create_subscription(WeldLineArray, '/vision/weld_lines_2d', lambda m: self.cb(m, 'lines'), 10)
        self.create_subscription(Image, '/kinect2/qhd/image_depth_rect', lambda m: self.cb(m, 'depth'), 10)
        self.create_subscription(CameraInfo, '/kinect2/qhd/camera_info', lambda m: self.cb(m, 'info'), 10)
        
        self.timer = self.create_timer(1.0, self.check_sync)
        self.get_logger().info("Listening for topics...")

    def cb(self, msg, topic_key):
        # Store the timestamp of the message header
        self.latest_stamps[topic_key] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def check_sync(self):
        required = ['lines', 'depth', 'info']
        missing = [key for key in required if key not in self.latest_stamps]
        
        if missing:
            self.get_logger().info(f"Waiting for topics... Missing: {missing}")
            return

        times = [self.latest_stamps[k] for k in required]
        max_diff = max(times) - min(times)
        
        self.get_logger().info(f"Max timestamp skew: {max_diff:.4f}s")
        
        if max_diff > 0.1:
            self.get_logger().warn(f"⚠️  SYNC ISSUE: Skew ({max_diff:.4f}s) > 0.1s tolerance! Depth Matcher will drop messages.")
        else:
            self.get_logger().info(f"✅ Sync OK (Skew {max_diff:.4f}s < 0.1s)")

def main(args=None):
    rclpy.init(args=args)
    node = SyncChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
