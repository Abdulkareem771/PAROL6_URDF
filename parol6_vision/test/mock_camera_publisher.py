#!/usr/bin/env python3
"""
Mock Camera Publisher for Integration Testing.
Publishes synthetic Red Line images and Depth maps to simulate Kinect v2.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class MockCameraPublisher(Node):
    def __init__(self):
        super().__init__('mock_camera_publisher')
        
        self.color_pub = self.create_publisher(Image, '/kinect2/qhd/image_color_rect', 10)
        self.depth_pub = self.create_publisher(Image, '/kinect2/qhd/image_depth_rect', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/kinect2/qhd/camera_info', 10)
        
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0, self.publish_data) # 1 Hz
        
        # Create synthetic data once
        self.color_img, self.depth_img = self.create_synthetic_data()
        
    def create_synthetic_data(self):
        # 960x540 (qhd)
        width, height = 960, 540
        
        # Color: Black background with Red Line
        color = np.zeros((height, width, 3), dtype=np.uint8)
        # Red line from (200, 200) to (600, 400). Thickness 20.
        cv2.line(color, (200, 200), (600, 400), (0, 0, 255), 20)
        
        # Depth: Plane at 1.0m (1000mm)
        # 16UC1 format
        depth = np.full((height, width), 1000, dtype=np.uint16)
        
        return color, depth
        
    def publish_data(self):
        now = self.get_clock().now().to_msg()
        
        # 1. Publish Color
        color_msg = self.bridge.cv2_to_imgmsg(self.color_img, encoding='bgr8')
        color_msg.header.stamp = now
        color_msg.header.frame_id = 'kinect2_rgb_optical_frame'
        self.color_pub.publish(color_msg)
        
        # 2. Publish Depth
        depth_msg = self.bridge.cv2_to_imgmsg(self.depth_img, encoding='passthrough') # 16UC1
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = 'kinect2_rgb_optical_frame'
        self.depth_pub.publish(depth_msg)
        
        # 3. Publish CameraInfo
        info = CameraInfo()
        info.header.stamp = now
        info.header.frame_id = 'kinect2_rgb_optical_frame'
        info.width = 960
        info.height = 540
        # Simple pinhole: fx=500, fy=500, cx=480, cy=270
        info.k = [500.0, 0.0, 480.0, 0.0, 500.0, 270.0, 0.0, 0.0, 1.0]
        info.p = [500.0, 0.0, 480.0, 0.0,  0.0, 500.0, 270.0, 0.0,  0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)
        
        self.get_logger().info('Published mock camera data')

def main(args=None):
    rclpy.init(args=args)
    node = MockCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
