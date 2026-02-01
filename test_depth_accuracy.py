#!/usr/bin/env python3
"""
Depth Accuracy Test Tool - Professional UI with Sidebar
Tests camera depth measurements with independent RGB/Depth interaction
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
            '/kinect2/sd/image_depth_rect',
            self.depth_callback,
            10
        )
        
        # Subscribe to color image
        self.color_sub = self.create_subscription(
            Image,
            '/kinect2/sd/image_color_rect',
            self.color_callback,
            10
        )
        
        self.get_logger().info('Depth Tester Started - Professional UI Mode')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Instructions:')
        self.get_logger().info('1. Press SPACE to capture a snapshot')
        self.get_logger().info('2. Click on RGB view for RGB pixel info')
        self.get_logger().info('3. Click on DEPTH view for depth measurements')
        self.get_logger().info('4. Press M to toggle 3D measurement mode')
        self.get_logger().info('5. Press C to clear measurements')
        self.get_logger().info('=' * 60)
        
        # Separate tracking for RGB and Depth clicks
        self.rgb_click = None  # (x, y, r, g, b, depth, std_dev)
        self.depth_click = None  # (x, y, depth, std_dev)
        
        self.measurement_mode = False
        self.measurement_points = []  # For 3D distance measurements
        
        # Camera intrinsics (from calibration)
        self.fx = 365.4126892089844
        self.fy = 365.4126892089844
        self.cx = 260.4054870605469
        self.cy = 206.64410400390625
        
        self.frozen_depth = None
        self.frozen_color = None
        self.latest_color = None
        self.shot_number = 0
        
        # UI dimensions
        self.sidebar_width = 300
        self.image_height = 424
        self.image_width = 512
        
    def color_callback(self, msg):
        """Store latest color image"""
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing color image: {str(e)}')
    
    def create_sidebar(self):
        """Create sidebar with measurement table - two-column layout"""
        sidebar = np.ones((self.image_height, self.sidebar_width, 3), dtype=np.uint8) * 40
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Column positions
        col1_x = 10
        col2_x = 155
        
        # Title
        y = 30
        cv2.putText(sidebar, "MEASUREMENTS", (10, y), 
                   font, 0.7, (255, 255, 255), 2)
        cv2.line(sidebar, (10, y + 5), (self.sidebar_width - 10, y + 5), 
                (100, 100, 100), 1)
        
        # Shot number
        cv2.putText(sidebar, f"Shot #{self.shot_number}", (10, y + 30), 
                   font, 0.5, (100, 200, 255), 1)
        
        # RGB Section
        y = 70
        cv2.putText(sidebar, "RGB VIEW DATA:", (col1_x, y), 
                   font, 0.5, (100, 255, 100), 1)
        y += 22
        
        if self.rgb_click:
            x_pos, y_pos, r, g, b, depth_mm, std_mm = self.rgb_click
            
            # Left column: Position and RGB
            cv2.putText(sidebar, f"Pos: ({x_pos},{y_pos})", (col1_x, y), 
                       font, 0.35, (200, 200, 200), 1)
            cv2.putText(sidebar, f"RGB: {r},{g},{b}", (col1_x, y + 16), 
                       font, 0.35, (200, 200, 200), 1)
            
            # Color swatch
            cv2.rectangle(sidebar, (col1_x, y + 20), (col1_x + 45, y + 38), 
                         (int(b), int(g), int(r)), -1)
            cv2.rectangle(sidebar, (col1_x, y + 20), (col1_x + 45, y + 38), 
                         (180, 180, 180), 1)
            
            # Right column: Distance data
            if depth_mm > 0:
                depth_m = depth_mm / 1000.0
                cv2.putText(sidebar, f"{depth_m:.3f}m", (col2_x, y), 
                           font, 0.38, (100, 255, 100), 1)
                cv2.putText(sidebar, f"{depth_mm:.1f}mm", (col2_x, y + 16), 
                           font, 0.35, (100, 255, 100), 1)
                cv2.putText(sidebar, f"+/-{std_mm:.1f}mm", (col2_x, y + 32), 
                           font, 0.32, (150, 150, 150), 1)
            
            y += 55
        else:
            cv2.putText(sidebar, "Click RGB view", (col1_x, y), 
                       font, 0.35, (150, 150, 150), 1)
            y += 35
        
        # Divider
        cv2.line(sidebar, (10, y), (self.sidebar_width - 10, y), 
                (100, 100, 100), 1)
        y += 15
        
        # Depth Section
        cv2.putText(sidebar, "DEPTH VIEW DATA:", (col1_x, y), 
                   font, 0.5, (255, 200, 100), 1)
        y += 22
        
        if self.depth_click:
            x_pos, y_pos, depth_mm, std_mm = self.depth_click
            depth_m = depth_mm / 1000.0
            
            # Left column: Position
            cv2.putText(sidebar, f"Pos: ({x_pos},{y_pos})", (col1_x, y), 
                       font, 0.35, (200, 200, 200), 1)
            
            # Right column: Distance
            cv2.putText(sidebar, f"{depth_m:.3f}m", (col2_x, y), 
                       font, 0.38, (100, 255, 255), 1)
            cv2.putText(sidebar, f"{depth_mm:.1f}mm", (col2_x, y + 16), 
                       font, 0.35, (100, 255, 255), 1)
            cv2.putText(sidebar, f"+/-{std_mm:.1f}mm", (col2_x, y + 32), 
                       font, 0.32, (150, 150, 150), 1)
            
            y += 50
        else:
            cv2.putText(sidebar, "Click depth view", (col1_x, y), 
                       font, 0.35, (150, 150, 150), 1)
            y += 35
        
        # Divider
        cv2.line(sidebar, (10, y), (self.sidebar_width - 10, y), 
                (100, 100, 100), 1)
        y += 15
        
        # 3D Measurements Section
        cv2.putText(sidebar, "3D MEASUREMENTS:", (col1_x, y), 
                   font, 0.45, (255, 100, 255), 1)
        y += 20
        
        if self.measurement_mode:
            # Left: status, Right: point count
            cv2.putText(sidebar, "[ACTIVE]", (col1_x, y), 
                       font, 0.35, (255, 100, 255), 1)
            cv2.putText(sidebar, f"Pts: {len(self.measurement_points)}", (col2_x, y), 
                       font, 0.35, (200, 200, 200), 1)
            
            # Show last distance if available
            if len(self.measurement_points) >= 2:
                dist = self.calculate_3d_distance(
                    self.measurement_points[-2], 
                    self.measurement_points[-1]
                )
                y += 18
                cv2.putText(sidebar, f"{dist:.1f}mm", (col1_x, y), 
                           font, 0.37, (255, 100, 255), 1)
                cv2.putText(sidebar, f"{dist/10:.1f}cm", (col2_x, y), 
                           font, 0.37, (255, 100, 255), 1)
            
            y += 25
        else:
            cv2.putText(sidebar, "Press M to enable", (col1_x, y), 
                       font, 0.35, (150, 150, 150), 1)
            y += 30
        
        # Divider
        cv2.line(sidebar, (10, y), (self.sidebar_width - 10, y), 
                (100, 100, 100), 1)
        y += 15
        
        # Controls
        cv2.putText(sidebar, "CONTROLS:", (col1_x, y), 
                   font, 0.45, (255, 255, 100), 1)
        y += 18
        
        # Two columns for controls
        cv2.putText(sidebar, "SPACE - Shot", (col1_x, y), 
                   font, 0.32, (200, 200, 200), 1)
        cv2.putText(sidebar, "M - Measure", (col2_x, y), 
                   font, 0.32, (200, 200, 200), 1)
        y += 16
        cv2.putText(sidebar, "C - Clear", (col1_x, y), 
                   font, 0.32, (200, 200, 200), 1)
        cv2.putText(sidebar, "Q - Quit", (col2_x, y), 
                   font, 0.32, (200, 200, 200), 1)
        
        return sidebar
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks - separate logic for RGB vs Depth"""
        if event == cv2.EVENT_LBUTTONDOWN and self.frozen_depth is not None:
            # Account for sidebar offset
            x_adjusted = x - self.sidebar_width
            
            if x_adjusted < 0:
                # Clicked on sidebar, ignore
                return
            
            height, width = self.frozen_depth.shape
            
            # Determine which view was clicked
            if x_adjusted < width:
                # RGB view clicked
                self.handle_rgb_click(x_adjusted, y)
            elif x_adjusted < width * 2:
                # Depth view clicked
                x_depth = x_adjusted - width
                self.handle_depth_click(x_depth, y)
            
            # Redraw
            self.display_measurement(update_only=True)
    
    def handle_rgb_click(self, x, y):
        """Handle click on RGB view - now with depth lookup and 3D measurement support"""
        if self.frozen_color is None or self.frozen_depth is None:
            return
        
        height, width = self.frozen_color.shape[:2]
        if 0 <= x < width and 0 <= y < height:
            # Get RGB values
            b, g, r = self.frozen_color[y, x]
            
            # Look up corresponding depth (images are registered/aligned)
            depth_value = self.frozen_depth[y, x]
            
            # Calculate depth statistics
            region_size = 20
            y1 = max(0, y - region_size//2)
            y2 = min(height, y + region_size//2)
            x1 = max(0, x - region_size//2)
            x2 = min(width, x + region_size//2)
            
            region = self.frozen_depth[y1:y2, x1:x2]
            valid_depths = region[(region > 0) & ~np.isnan(region)]
            
            if len(valid_depths) > 0:
                avg_depth_mm = np.mean(valid_depths)
                std_depth_mm = np.std(valid_depths)
            else:
                avg_depth_mm = 0
                std_depth_mm = 0
            
            # Store RGB click with depth info
            self.rgb_click = (x, y, int(r), int(g), int(b), avg_depth_mm, std_depth_mm)
            
            # If in measurement mode, add point for 3D measurement
            if self.measurement_mode and depth_value > 0 and not np.isnan(depth_value):
                self.measurement_points.append((x, y, depth_value))
                self.get_logger().info(
                    f'3D Measurement Point {len(self.measurement_points)} (from RGB): '
                    f'({x}, {y}) depth={depth_value:.1f}mm'
                )
                
                if len(self.measurement_points) >= 2:
                    dist = self.calculate_3d_distance(
                        self.measurement_points[-2],
                        self.measurement_points[-1]
                    )
                    self.get_logger().info(f'📏 Distance: {dist:.1f} mm ({dist/10:.1f} cm)')
            
            if avg_depth_mm > 0:
                self.get_logger().info(
                    f'RGB Click: ({x}, {y}) RGB=({r}, {g}, {b}) '
                    f'Depth={avg_depth_mm:.1f}mm ± {std_depth_mm:.1f}mm'
                )
            else:
                self.get_logger().info(f'RGB Click: ({x}, {y}) RGB=({r}, {g}, {b}) [No depth]')
    
    def handle_depth_click(self, x, y):
        """Handle click on Depth view"""
        height, width = self.frozen_depth.shape
        
        if 0 <= x < width and 0 <= y < height:
            # If in measurement mode, add point for 3D measurement
            if self.measurement_mode:
                depth_value = self.frozen_depth[y, x]
                if depth_value > 0 and not np.isnan(depth_value):
                    self.measurement_points.append((x, y, depth_value))
                    self.get_logger().info(
                        f'3D Measurement Point {len(self.measurement_points)}: '
                        f'({x}, {y}) depth={depth_value:.1f}mm'
                    )
                    
                    if len(self.measurement_points) >= 2:
                        dist = self.calculate_3d_distance(
                            self.measurement_points[-2],
                            self.measurement_points[-1]
                        )
                        self.get_logger().info(f'📏 Distance: {dist:.1f} mm ({dist/10:.1f} cm)')
            
            # Calculate depth stats at this point
            region_size = 20
            y1 = max(0, y - region_size//2)
            y2 = min(height, y + region_size//2)
            x1 = max(0, x - region_size//2)
            x2 = min(width, x + region_size//2)
            
            region = self.frozen_depth[y1:y2, x1:x2]
            valid_depths = region[(region > 0) & ~np.isnan(region)]
            
            if len(valid_depths) > 0:
                avg_depth_mm = np.mean(valid_depths)
                std_depth_mm = np.std(valid_depths)
                self.depth_click = (x, y, avg_depth_mm, std_depth_mm)
                self.get_logger().info(
                    f'Depth Click: ({x}, {y}) '
                    f'{avg_depth_mm:.1f}mm ± {std_depth_mm:.1f}mm'
                )
    
    def calculate_3d_distance(self, point1, point2):
        """Calculate 3D Euclidean distance between two points"""
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        
        # Convert to 3D coordinates using camera intrinsics
        X1 = (x1 - self.cx) * z1 / self.fx
        Y1 = (y1 - self.cy) * z1 / self.fy
        Z1 = z1
        
        X2 = (x2 - self.cx) * z2 / self.fx
        Y2 = (y2 - self.cy) * z2 / self.fy
        Z2 = z2
        
        distance = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)
        return distance
    
    def display_measurement(self, update_only=False):
        """Display with sidebar and clean video feeds"""
        if self.frozen_depth is None or self.frozen_color is None:
            return
        
        height, width = self.frozen_depth.shape
        
        # Create clean depth visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(self.frozen_depth, alpha=255.0/5000.0),
            cv2.COLORMAP_JET
        )
        
        color_display = self.frozen_color.copy()
        
        # Draw markers on RGB view (green for RGB clicks)
        if self.rgb_click:
            x, y = self.rgb_click[0], self.rgb_click[1]
            cv2.circle(color_display, (x, y), 10, (0, 255, 0), 2)
            cv2.drawMarker(color_display, (x, y), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            
            # Draw measurement box if depth available
            depth_mm = self.rgb_click[5]
            if depth_mm > 0:
                region_size = 20
                y1 = max(0, y - region_size//2)
                y2 = min(height, y + region_size//2)
                x1 = max(0, x - region_size//2)
                x2 = min(width, x + region_size//2)
                cv2.rectangle(color_display, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Draw markers on Depth view (cyan for depth clicks)
        if self.depth_click:
            x, y = self.depth_click[0], self.depth_click[1]
            cv2.circle(depth_colormap, (x, y), 10, (255, 255, 0), 2)
            cv2.drawMarker(depth_colormap, (x, y), (255, 255, 0),
                          cv2.MARKER_CROSS, 20, 2)
            
            # Draw measurement box
            region_size = 20
            y1 = max(0, y - region_size//2)
            y2 = min(height, y + region_size//2)
            x1 = max(0, x - region_size//2)
            x2 = min(width, x + region_size//2)
            cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (255, 255, 0), 1)
        
        # Draw 3D measurement points (magenta) on BOTH views
        if self.measurement_mode and len(self.measurement_points) > 0:
            for i, (px, py, _) in enumerate(self.measurement_points):
                # Draw on both RGB and depth views
                cv2.circle(color_display, (px, py), 6, (255, 0, 255), -1)
                cv2.circle(depth_colormap, (px, py), 6, (255, 0, 255), -1)
                cv2.putText(color_display, f"P{i+1}", (px+8, py-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                cv2.putText(depth_colormap, f"P{i+1}", (px+8, py-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # Draw lines between pairs on both views
            for i in range(0, len(self.measurement_points) - 1, 2):
                p1 = self.measurement_points[i]
                p2 = self.measurement_points[i + 1]
                cv2.line(color_display, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 255), 2)
                cv2.line(depth_colormap, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 255), 2)
        
        # Add minimal labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(color_display, "RGB VIEW", (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(depth_colormap, "DEPTH VIEW", (10, 25), font, 0.6, (255, 255, 255), 2)
        
        # Create sidebar
        sidebar = self.create_sidebar()
        
        # Combine: [Sidebar | RGB | Depth]
        video_feeds = np.hstack([color_display, depth_colormap])
        combined = np.hstack([sidebar, video_feeds])
        
        cv2.imshow('Depth Measurement Tool - Professional UI', combined)
        cv2.setMouseCallback('Depth Measurement Tool - Professional UI', 
                            self.mouse_callback)
        
        if not update_only:
            self.get_logger().info(f'📸 Shot #{self.shot_number} captured')
    
    def depth_callback(self, msg):
        """Process depth image"""
        try:
            # Handle frozen frame
            if self.frozen_depth is not None:
                key = cv2.waitKey(1)
                if key == ord(' '):
                    self.frozen_depth = None
                    self.frozen_color = None
                    self.rgb_click = None
                    self.depth_click = None
                    self.measurement_points.clear()
                    self.measurement_mode = False
                    self.get_logger().info('Ready for next shot...')
                elif key == ord('m') or key == ord('M'):
                    self.measurement_mode = not self.measurement_mode
                    self.get_logger().info(f'Measurement mode: {"ON" if self.measurement_mode else "OFF"}')
                    self.display_measurement(update_only=True)
                elif key == ord('c') or key == ord('C'):
                    self.measurement_points.clear()
                    self.rgb_click = None
                    self.depth_click = None
                    self.get_logger().info('Cleared all measurements')
                    self.display_measurement(update_only=True)
                elif key == ord('q') or key == ord('Q'):
                    self.get_logger().info('Shutting down...')
                    cv2.destroyAllWindows()
                    rclpy.shutdown()
                return
            
            # Live preview
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            height, width = depth_image.shape
            
            depth_preview = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=255.0/5000.0),
                cv2.COLORMAP_JET
            )
            
            color_preview = self.latest_color.copy() if self.latest_color is not None else \
                           np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(color_preview, "LIVE RGB - Press SPACE", (10, 30), 
                       font, 0.6, (255, 255, 0), 2)
            cv2.putText(depth_preview, "LIVE DEPTH - Press SPACE", (10, 30),
                       font, 0.6, (255, 255, 0), 2)
            
            # Create live sidebar
            sidebar = np.ones((height, self.sidebar_width, 3), dtype=np.uint8) * 40
            cv2.putText(sidebar, "LIVE MODE", (50, height//2), 
                       font, 0.7, (255, 255, 100), 2)
            cv2.putText(sidebar, "Press SPACE", (40, height//2 + 40),
                       font, 0.5, (200, 200, 200), 1)
            cv2.putText(sidebar, "to capture", (45, height//2 + 65),
                       font, 0.5, (200, 200, 200), 1)
            
            combined = np.hstack([sidebar, color_preview, depth_preview])
            cv2.imshow('Depth Measurement Tool - Professional UI', combined)
            
            key = cv2.waitKey(1)
            if key == ord(' '):
                self.shot_number += 1
                self.frozen_depth = depth_image.copy()
                self.frozen_color = self.latest_color.copy() if self.latest_color is not None else None
                self.display_measurement()
            elif key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')


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
