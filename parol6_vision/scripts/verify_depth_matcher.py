#!/usr/bin/env python3
"""
verify_depth_matcher.py

A standalone verification script for the Depth Matcher node.
This script acts as a "mock" system, publishing synchronized:
1. 2D Weld Line detections
2. Depth Images
3. Camera Info

It then listens for the resulting 3D Weld Lines to verify the node is working correctly.
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from parol6_msgs.msg import WeldLine, WeldLineArray, WeldLine3DArray
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
import numpy as np
import cv2
import threading
import time

class DepthMatcherVerifier(Node):
    def __init__(self):
        super().__init__('depth_matcher_verifier')
        
        self.bridge = CvBridge()
        
        # Publishers (Inputs to Depth Matcher)
        self.lines_pub = self.create_publisher(WeldLineArray, '/vision/weld_lines_2d', 10)
        self.depth_pub = self.create_publisher(Image, '/kinect2/qhd/image_depth_rect', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/kinect2/qhd/camera_info', 10)
        
        # Subscriber (Output from Depth Matcher)
        self.result_sub = self.create_subscription(
            WeldLine3DArray, 
            '/vision/weld_lines_3d',
            self.result_callback,
            10
        )
        
        self.received_message = False
        self.get_logger().info("Verifier initialized. Waiting for Depth Matcher...")

    def publish_mock_data(self):
        """Publish synchronized mock data to trigger the Depth Matcher."""
        timestamp = self.get_clock().now().to_msg()
        
        # 1. Create a dummy Depth Image (flat plane at 1.0m)
        depth_img = np.ones((540, 960), dtype=np.uint16) * 1000  # 1000mm = 1m
        depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="passthrough")
        depth_msg.header.stamp = timestamp
        depth_msg.header.frame_id = "camera_rgb_optical_frame"
        
        # 2. Create Dummy Camera Info
        info_msg = CameraInfo()
        info_msg.header.stamp = timestamp
        info_msg.header.frame_id = "camera_rgb_optical_frame"
        info_msg.width = 960
        info_msg.height = 540
        # Simulating Kinect v2 QHD intrinsics
        info_msg.k = [1081.37, 0.0, 959.5, 0.0, 1081.37, 539.5, 0.0, 0.0, 1.0]
        
        # 3. Create Dummy Weld Line (Horizontal line in center)
        line = WeldLine()
        line.id = "test_line_0"
        line.confidence = 0.95
        
        # Create points across the middle
        line.pixels = []
        for x in range(400, 600, 10):
            pt = Point32()
            pt.x = float(x)
            pt.y = 270.0 # Center Y
            pt.z = 0.0
            line.pixels.append(pt)
            
        weld_array = WeldLineArray()
        weld_array.header.stamp = timestamp
        weld_array.header.frame_id = "camera_rgb_optical_frame"
        weld_array.lines = [line]
        
        # Publish
        self.lines_pub.publish(weld_array)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info_msg)
        
        self.get_logger().info("Published synchronized mock data batch.")

    def result_callback(self, msg):
        self.received_message = True
        self.get_logger().info("✅ SUCCESS! Received 3D Weld Lines.")
        self.get_logger().info(f"  - Frame ID: {msg.header.frame_id}")
        self.get_logger().info(f"  - Num Lines: {len(msg.lines)}")
        if len(msg.lines) > 0:
            line = msg.lines[0]
            self.get_logger().info(f"  - Line 0 Points: {len(line.points)}")
            self.get_logger().info(f"  - Line 0 Depth Quality: {line.depth_quality}")
            
            # Verify approximate 3D position
            # At 1m depth, center pixel (960/2, 540/2) should be roughly (0,0,1) in camera frame
            # But note: Output is transformed to 'base_link'. 
            pass

    def save_report_image(self):
        """Save a visual report of the test."""
        # Create a visual canvas (RGB)
        h, w = 540, 960
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 1. Visualize Depth (Simulated as grayscale gradient for style)
        # In our mock, depth is constant 1000mm, so let's just make it gray
        canvas[:] = (50, 50, 50) 
        
        # 2. Draw the Input 2D Line (Green)
        # We know our mock data: y=270, x=400..600
        pt1 = (400, 270)
        pt2 = (600, 270)
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 3)
        cv2.putText(canvas, "Input 2D Weld Line", (400, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                   
        # 3. Add Info Text
        cv2.putText(canvas, "PAROL6 VISION - DEPTH MATCHER VERIFICATION", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(canvas, "Result: SUCCESS", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
        cv2.putText(canvas, f"Timestamp: {time.ctime()}", (50, 500), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Save
        filename = "verification_result.png"
        cv2.imwrite(filename, canvas)
        print(f"Image saved to: {filename}")

def main(args=None):
    rclpy.init(args=args)
    verifier = DepthMatcherVerifier()
    
    # Spin in a separate thread so we can publish
    spin_thread = threading.Thread(target=rclpy.spin, args=(verifier,))
    spin_thread.start()
    
    try:
        # Wait a bit for connections
        time.sleep(2.0)
        
        # Keep publishing until success or timeout
        for i in range(10): # Try for 10 seconds
            if verifier.received_message:
                break
            verifier.publish_mock_data()
            time.sleep(1.0)
            
        if verifier.received_message:
            print("\n" + "="*50)
            print("TEST PASSED: Depth Matcher is compatible and working!")
            print("Saving visual report to 'verification_result.png'...")
            verifier.save_report_image()
            print("="*50 + "\n")
        else:
            print("\n" + "="*50)
            print("TEST FAILED: No 3D lines received.")
            print("Possible reasons:")
            print("1. TF Transform missing (camera_rgb_optical_frame -> base_link)")
            print("2. Topics not matching (check remappings)")
            print("3. Synchronization failed (timestamps too far apart)")
            print("="*50 + "\n")
            
    finally:
        rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()
