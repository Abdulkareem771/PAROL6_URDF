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
        
        # Subscribe to depth image (HD resolution - highest quality)
        self.depth_sub = self.create_subscription(
            Image,
            '/kinect2/hd/image_depth_rect',  # registered depth image
            self.depth_callback,
            10
        )
        
        # Subscribe to color image (HD resolution)
        self.color_sub = self.create_subscription(
            Image,
            '/kinect2/hd/image_color_rect',  # registered color image
            self.color_callback,
            10
        )
        
        self.get_logger().info('Depth Tester Started - SNAPSHOT & MEASUREMENT MODE')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Instructions:')
        self.get_logger().info('1. Position your object at a known distance')
        self.get_logger().info('2. Press SPACE to capture a frame')
        self.get_logger().info('3. Press M to toggle MEASUREMENT mode')
        self.get_logger().info('4. In measurement mode: Click 2 points to measure distance')
        self.get_logger().info('5. Press C to clear measurements')
        self.get_logger().info('6. Press SPACE for next shot | Q to quit')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Waiting for first frame...\n')
        
        self.mouse_x = None
        self.mouse_y = None
        self.measurement_mode = False
        self.measurement_points = []  # Store (x, y, depth) tuples
        
        # Camera intrinsics for HD resolution (1920x1080)
        # From your verified calibration (camera_info topic)
        self.fx = 1185.6124290477237  # focal length x
        self.fy = 1189.1865298780951  # focal length y
        self.cx = 1014.1114412036075  # principal point x
        self.cy = 590.9925793653344   # principal point y
        self.frozen_depth_vis = None
        self.frozen_color = None
        self.frozen_depth = None
        self.latest_color = None
        self.capture_requested = False
        self.shot_number = 0
        
    def color_callback(self, msg):
        """Store latest color image"""
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing color image: {str(e)}')
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select point"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.measurement_mode and self.frozen_depth is not None:
                # In measurement mode, collect points for distance measurement
                # Need to adjust x coordinate since we're showing combined image
                # Points on left side (color) are 0 to width
                # Points on right side (depth) are width to 2*width
                height, width = self.frozen_depth.shape
                
                # Normalize x to image width (handle both color and depth sides)
                if x < width:
                    # Clicked on color side
                    actual_x = x
                else:
                    # Clicked on depth side
                    actual_x = x - width
                
                actual_y = y
                
                # Get depth at this point
                if 0 <= actual_x < width and 0 <= actual_y < height:
                    depth_value = self.frozen_depth[actual_y, actual_x]
                    
                    if depth_value > 0 and not np.isnan(depth_value):
                        self.measurement_points.append((actual_x, actual_y, depth_value))
                        self.get_logger().info(f'Point {len(self.measurement_points)}: ({actual_x}, {actual_y}) depth={depth_value:.1f}mm')
                        
                        if len(self.measurement_points) >= 2:
                            distance = self.calculate_3d_distance(self.measurement_points[-2], self.measurement_points[-1])
                            self.get_logger().info(f'📏 Distance: {distance:.1f} mm ({distance/10:.1f} cm)')
                    else:
                        self.get_logger().warn(f'No valid depth at ({actual_x}, {actual_y})')
                
                # Redraw with measurement overlay
                self.display_measurement(self.frozen_depth, self.frozen_color, update_only=True)
            else:
                # Normal mode - just update measurement point
                self.mouse_x = x
                self.mouse_y = y
                # Redraw the frozen frame with new measurement point
                if self.frozen_depth_vis is not None and self.frozen_depth is not None:
                    self.display_measurement(self.frozen_depth, self.frozen_color, update_only=True)
    

    
    def calculate_3d_distance(self, point1, point2):
        """Calculate 3D Euclidean distance between two points using depth data"""
        x1, y1, z1 = point1  # z is depth in mm
        x2, y2, z2 = point2
        
        # Convert pixel coordinates to 3D coordinates using camera intrinsics
        # X = (x - cx) * Z / fx
        # Y = (y - cy) * Z / fy
        # Z = depth
        
        X1 = (x1 - self.cx) * z1 / self.fx
        Y1 = (y1 - self.cy) * z1 / self.fy
        Z1 = z1
        
        X2 = (x2 - self.cx) * z2 / self.fx
        Y2 = (y2 - self.cy) * z2 / self.fy
        Z2 = z2
        
        # Calculate Euclidean distance
        distance = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)
        
        return distance  # in mm
    
    def display_measurement(self, depth_image, color_image, update_only=False):
        """Display depth measurement on image"""
        # Get image dimensions
        height, width = depth_image.shape
        center_x, center_y = width // 2, height // 2
        
        # Create depth visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=255.0/5000.0), 
            cv2.COLORMAP_JET
        )
        
        # Prepare color image (copy so we don't modify original)
        if color_image is not None:
            color_display = color_image.copy()
        else:
            # If no color image, create a blank one
            color_display = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(color_display, "No color image", (width//4, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
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
        
        # Draw crosshairs and box on BOTH images
        for img in [depth_colormap, color_display]:
            cv2.drawMarker(img, (measure_x, measure_y), 
                          (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Create measurement info table
        font = cv2.FONT_HERSHEY_SIMPLEX
        table_color = (0, 255, 0) if avg_depth_m > 0 else (0, 0, 255)
        
        # Draw semi-transparent info panel on color image
        info_height = 150
        overlay = color_display.copy()
        cv2.rectangle(overlay, (0, 0), (width, info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, color_display, 0.4, 0, color_display)
        
        # Table header
        cv2.putText(color_display, f"Shot #{self.shot_number} - Measurement Data", 
                   (10, 25), font, 0.7, (0, 255, 255), 2)
        cv2.line(color_display, (10, 35), (width-10, 35), (255, 255, 255), 1)
        
        # Table rows
        y_offset = 60
        line_height = 30
        
        if avg_depth_m > 0:
            # Distance row
            cv2.putText(color_display, "Distance:", (20, y_offset), font, 0.6, (255, 255, 255), 1)
            cv2.putText(color_display, f"{avg_depth_m:.3f} m", (width//2, y_offset), font, 0.6, table_color, 2)
            
            # Millimeters row
            y_offset += line_height
            cv2.putText(color_display, "(mm):", (20, y_offset), font, 0.6, (255, 255, 255), 1)
            cv2.putText(color_display, f"{avg_depth_mm:.1f} ± {std_depth_mm:.1f}", (width//2, y_offset), font, 0.6, table_color, 2)
            
            # Position row
            y_offset += line_height
            cv2.putText(color_display, "Position:", (20, y_offset), font, 0.6, (255, 255, 255), 1)
            cv2.putText(color_display, f"({measure_x}, {measure_y})", (width//2, y_offset), font, 0.6, (200, 200, 200), 1)
            
            # Console log (only on new capture)
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
            cv2.putText(color_display, "No valid depth data at this point", 
                       (20, y_offset), font, 0.6, (0, 0, 255), 2)
        
        # Add resolution indicator
        cv2.putText(depth_colormap, "HD (1920x1080)", (10, height-25), font, 0.5, (255, 255, 255), 1)
        cv2.putText(depth_colormap, "DEPTH", (10, height-10), font, 0.5, (255, 255, 255), 1)
        cv2.putText(color_display, "COLOR", (10, height-10), font, 0.5, (255, 255, 255), 1)
        
        # Draw measurement points and lines if in measurement mode
        if self.measurement_mode and len(self.measurement_points) > 0:
            for i, (px, py, _) in enumerate(self.measurement_points):
                # Draw on both images
                cv2.circle(color_display, (px, py), 8, (255, 0, 255), -1)
                cv2.circle(depth_colormap, (px, py), 8, (255, 0, 255), -1)
                # Label the point
                cv2.putText(color_display, f"P{i+1}", (px+10, py-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(depth_colormap, f"P{i+1}", (px+10, py-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Draw line between consecutive pairs of points
            for i in range(0, len(self.measurement_points) - 1, 2):
                p1 = self.measurement_points[i]
                p2 = self.measurement_points[i + 1]
                
                cv2.line(color_display, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 255), 2)
                cv2.line(depth_colormap, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 255), 2)
                
                # Calculate and display distance
                distance = self.calculate_3d_distance(p1, p2)
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                
                dist_text = f"{distance:.1f}mm ({distance/10:.1f}cm)"
                cv2.putText(color_display, dist_text, (mid_x, mid_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(depth_colormap, dist_text, (mid_x, mid_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Combine images side by side
        combined = np.hstack([color_display, depth_colormap])
        
        # Show measurement mode indicator
        if self.measurement_mode:
            mode_text = "[MEASUREMENT MODE] Click 2 points to measure"
            cv2.putText(combined, mode_text, (width//2 - 250, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Add instructions at bottom
        if self.measurement_mode:
            instructions = "M: toggle mode | C: clear | Click 2 points | SPACE: next | Q: quit"
        else:
            instructions = "M: measure mode | SPACE: next shot | Click: point | Q: quit"
        cv2.putText(combined, instructions, (width//2 - 300, height*2-10), 
                   font, 0.6, (255, 255, 255), 2)
        
        # Show the combined image
        cv2.imshow('Depth Measurement Tool - Color & Depth', combined)
        cv2.setMouseCallback('Depth Measurement Tool - Color & Depth', self.mouse_callback)
    
    def depth_callback(self, msg):
        """Process depth image"""
        try:
            # If we have a frozen frame, just handle key presses
            if self.frozen_depth_vis is not None:
                key = cv2.waitKey(1)
                if key == ord(' '):  # Space bar
                    self.capture_requested = True
                    self.frozen_depth_vis = None
                    self.frozen_color = None
                    self.frozen_depth = None
                    self.mouse_x = None
                    self.mouse_y = None
                    self.measurement_points.clear()
                    self.measurement_mode = False
                    self.get_logger().info('Ready for next shot...')
                elif key == ord('m') or key == ord('M'):  # Toggle measurement mode
                    self.measurement_mode = not self.measurement_mode
                    status = "ON" if self.measurement_mode else "OFF"
                    self.get_logger().info(f'Measurement mode: {status}')
                    self.display_measurement(self.frozen_depth, self.frozen_color, update_only=True)
                elif key == ord('c') or key == ord('C'):  # Clear measurements
                    self.measurement_points.clear()
                    self.get_logger().info('Measurements cleared')
                    self.display_measurement(self.frozen_depth, self.frozen_color, update_only=True)
                elif key == ord('q') or key == ord('Q'):
                    self.get_logger().info('Shutting down...')
                    cv2.destroyAllWindows()
                    rclpy.shutdown()
                return
            
            # Convert ROS Image message to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            
            # Show live preview
            height, width = depth_image.shape
            depth_preview = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=255.0/5000.0), 
                cv2.COLORMAP_JET
            )
            
            # Get color preview
            if self.latest_color is not None:
                color_preview = self.latest_color.copy()
            else:
                color_preview = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(color_preview, "Waiting for color...", (width//4, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add text overlays
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(depth_preview, "LIVE DEPTH - Press SPACE to capture", 
                       (10, 30), font, 0.6, (255, 255, 0), 2)
            cv2.putText(color_preview, "LIVE COLOR - Press SPACE to capture", 
                       (10, 30), font, 0.6, (255, 255, 0), 2)
            
            # Combine side by side
            combined_preview = np.hstack([color_preview, depth_preview])
            
            cv2.putText(combined_preview, "Q to quit", 
                       (width-100, height-20), font, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Depth Measurement Tool - Color & Depth', combined_preview)
            
            key = cv2.waitKey(1)
            if key == ord(' '):  # Space bar - capture frame
                self.shot_number += 1
                self.frozen_depth_vis = depth_preview.copy()
                self.frozen_depth = depth_image.copy()
                self.frozen_color = self.latest_color.copy() if self.latest_color is not None else None
                self.display_measurement(depth_image, self.frozen_color)
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
