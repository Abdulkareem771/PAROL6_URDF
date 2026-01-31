#!/usr/bin/env python3
"""
Depth Accuracy Test Tool
Tests if camera depth measurements match real-world distances
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class DepthTester(Node):
    def __init__(self):
        super().__init__('depth_tester')
        self.bridge = CvBridge()
        
        # Subscribe to depth image
        self.depth_sub = self.create_subscription(
            Image,
            '/kinect2/sd/image_depth_rect',  # registered depth image
            self.depth_callback,
            10
        )
        
        self.get_logger().info('Depth Tester Started - SNAPSHOT MODE')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Instructions:')
        self.get_logger().info('1. Position your object at a known distance')
        self.get_logger().info('2. Press SPACE to capture a frame')
        self.get_logger().info('3. Measure with tape measure and compare')
        self.get_logger().info('4. Click on frozen image to check different points')
        self.get_logger().info('5. Press SPACE again for next measurement')
        self.get_logger().info('6. Press Q to quit')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Waiting for first frame...\n')
        
        self.mouse_x = None
        self.mouse_y = None
        self.frozen_frame = None
        self.frozen_depth = None
        self.capture_requested = False
        self.shot_number = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select point"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x = x
            self.mouse_y = y
            # Redraw the frozen frame with new measurement point
            if self.frozen_frame is not None and self.frozen_depth is not None:
                self.display_measurement(self.frozen_depth, update_only=True)
    
    def display_measurement(self, depth_image, update_only=False):
        """Display depth measurement on image"""
        # Get image dimensions
        height, width = depth_image.shape
        center_x, center_y = width // 2, height // 2
        
        # Create visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=255.0/5000.0), 
            cv2.COLORMAP_JET
        )
        
        # Measure depth at center (default)
        measure_x = center_x
        measure_y = center_y
        
        # If user clicked, measure at that point
        if self.mouse_x is not None and self.mouse_y is not None:
            measure_x = self.mouse_x
            measure_y = self.mouse_y
        
        # Calculate average in a 20x20 region for stability
        region_size = 20
        y1 = max(0, measure_y - region_size//2)
        y2 = min(height, measure_y + region_size//2)
        x1 = max(0, measure_x - region_size//2)
        x2 = min(width, measure_x + region_size//2)
        
        region = depth_image[y1:y2, x1:x2]
        # Filter out invalid values (0 or NaN)
        valid_depths = region[(region > 0) & ~np.isnan(region)]
        
        if len(valid_depths) > 0:
            avg_depth_mm = np.mean(valid_depths)
            avg_depth_m = avg_depth_mm / 1000.0
            std_depth_mm = np.std(valid_depths)
            std_depth_m = std_depth_mm / 1000.0
        else:
            avg_depth_m = 0
            std_depth_m = 0
        
        # Draw crosshairs at measurement point
        cv2.drawMarker(depth_colormap, (measure_x, measure_y), 
                      (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        
        # Draw measurement box
        cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Show shot number
        shot_text = f"Shot #{self.shot_number} [FROZEN]"
        cv2.putText(depth_colormap, shot_text, (10, height-60), font, 0.8, (0, 255, 255), 2)
        
        if avg_depth_m > 0:
            text1 = f"Distance: {avg_depth_m:.3f} m ({avg_depth_mm:.1f} mm)"
            text2 = f"Std Dev: +/- {std_depth_mm:.1f} mm"
            text3 = f"Position: ({measure_x}, {measure_y})"
            text4 = "Press SPACE for next shot | Click to remeasure | Q to quit"
            
            cv2.putText(depth_colormap, text1, (10, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(depth_colormap, text2, (10, 65), font, 0.7, (0, 255, 0), 2)
            cv2.putText(depth_colormap, text3, (10, 100), font, 0.6, (255, 255, 255), 2)
            cv2.putText(depth_colormap, text4, (10, height-20), font, 0.6, (255, 255, 255), 2)
            
            # Print to console (only on new capture, not on click updates)
            if not update_only:
                self.get_logger().info('=' * 60)
                self.get_logger().info(f'📸 SHOT #{self.shot_number} CAPTURED')
                self.get_logger().info(
                    f"📏 Distance: {avg_depth_m:.3f} m ± {std_depth_m:.4f} m "
                    f"({avg_depth_mm:.1f} ± {std_depth_mm:.1f} mm)"
                )
                self.get_logger().info(f"📍 Position: ({measure_x}, {measure_y})")
                self.get_logger().info('=' * 60)
                self.get_logger().info('Press SPACE to capture next shot, or Q to quit\n')
        else:
            cv2.putText(depth_colormap, "No valid depth data", (10, 30), 
                       font, 0.7, (0, 0, 255), 2)
            text4 = "Press SPACE for next shot | Q to quit"
            cv2.putText(depth_colormap, text4, (10, height-20), font, 0.6, (255, 255, 255), 2)
        
        # Show the image
        cv2.imshow('Depth Measurement Tool', depth_colormap)
        cv2.setMouseCallback('Depth Measurement Tool', self.mouse_callback)
    
    def depth_callback(self, msg):
        """Process depth image"""
        try:
            # If we have a frozen frame, just handle key presses
            if self.frozen_frame is not None:
                key = cv2.waitKey(1)
                if key == ord(' '):  # Space bar
                    self.capture_requested = True
                    self.frozen_frame = None
                    self.frozen_depth = None
                    self.mouse_x = None
                    self.mouse_y = None
                    self.get_logger().info('Ready for next shot...')
                elif key == ord('q') or key == ord('Q'):
                    self.get_logger().info('Shutting down...')
                    cv2.destroyAllWindows()
                    rclpy.shutdown()
                return
            
            # Convert ROS Image message to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            
            # Show live preview
            height, width = depth_image.shape
            preview = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=255.0/5000.0), 
                cv2.COLORMAP_JET
            )
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(preview, "LIVE VIEW - Press SPACE to capture", 
                       (10, 30), font, 0.8, (255, 255, 0), 2)
            cv2.putText(preview, "Q to quit", 
                       (10, height-20), font, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Depth Measurement Tool', preview)
            
            key = cv2.waitKey(1)
            if key == ord(' '):  # Space bar - capture frame
                self.shot_number += 1
                self.frozen_frame = preview.copy()
                self.frozen_depth = depth_image.copy()
                self.display_measurement(depth_image)
            elif key == ord('q') or key == ord('Q'):
                self.get_logger().info('Shutting down...')
                cv2.destroyAllWindows()
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = DepthTester()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
