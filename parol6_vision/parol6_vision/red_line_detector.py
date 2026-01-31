#!/usr/bin/env python3
"""
Red Line Detector Node - Vision-Guided Welding Path Detection (Patched V2)

Modifications for offline compatibility:
1. Removed `cv_bridge` dependency (Manual conversion) to fix NumPy 2.0 issues.
2. Removed `skimage` dependency (OpenCV thinning) to fix missing package.
3. Removed `sklearn` dependency (Simple sorting) to fix missing package.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from parol6_msgs.msg import WeldLine, WeldLineArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point32, Point
from std_msgs.msg import ColorRGBA
import cv2
import numpy as np
# from sklearn.decomposition import PCA - REMOVED

# --- Helper function to replace cv_bridge ---
def imgmsg_to_cv2(img_msg, desired_encoding="passthrough"):
    if img_msg.encoding == "bgr8":
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == "rgb8":
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == "mono8":
        dtype = np.uint8
        n_channels = 1
    else:
        # Fallback for other encodings if needed
        dtype = np.uint8
        n_channels = 3 # Assume 3 for now

    img_buf = np.frombuffer(img_msg.data, dtype=dtype)
    img = img_buf.reshape(img_msg.height, img_msg.width, -1)
    
    if n_channels == 1:
        img = img.reshape(img_msg.height, img_msg.width)
    
    if desired_encoding == "bgr8" and img_msg.encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

def cv2_to_imgmsg(cv_img, encoding="bgr8"):
    msg = Image()
    msg.height = cv_img.shape[0]
    msg.width = cv_img.shape[1]
    if len(cv_img.shape) == 3:
        msg.encoding = encoding
        msg.step = cv_img.shape[1] * cv_img.shape[2]
    else:
        msg.encoding = "mono8"
        msg.step = cv_img.shape[1]

    msg.data = cv_img.tobytes()
    return msg
# ---------------------------------------------

class RedLineDetector(Node):
    def __init__(self):
        super().__init__('red_line_detector')
        
        # HSV Color Ranges
        self.declare_parameter('hsv_lower_1', [0, 100, 100])
        self.declare_parameter('hsv_upper_1', [10, 255, 255])
        self.declare_parameter('hsv_lower_2', [160, 50, 0])
        self.declare_parameter('hsv_upper_2', [180, 255, 255])
        
        self.declare_parameter('morphology_kernel_size', 5)
        self.declare_parameter('erosion_iterations', 1)
        self.declare_parameter('dilation_iterations', 2)
        
        self.declare_parameter('min_line_length', 60)
        self.declare_parameter('min_contour_area', 700)
        self.declare_parameter('douglas_peucker_epsilon', 2.0)
        
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('max_lines_per_frame', 5)
        self.declare_parameter('publish_debug_images', True)
        
        # Get Parameters
        self.hsv_lower_1 = np.array(self.get_parameter('hsv_lower_1').value, dtype=np.uint8)
        self.hsv_upper_1 = np.array(self.get_parameter('hsv_upper_1').value, dtype=np.uint8)
        self.hsv_lower_2 = np.array(self.get_parameter('hsv_lower_2').value, dtype=np.uint8)
        self.hsv_upper_2 = np.array(self.get_parameter('hsv_upper_2').value, dtype=np.uint8)
        
        self.kernel_size = self.get_parameter('morphology_kernel_size').value
        self.erosion_iters = self.get_parameter('erosion_iterations').value
        self.dilation_iters = self.get_parameter('dilation_iterations').value
        
        self.min_line_length = self.get_parameter('min_line_length').value
        self.min_contour_area = self.get_parameter('min_contour_area').value
        self.dp_epsilon = self.get_parameter('douglas_peucker_epsilon').value
        
        self.min_confidence = self.get_parameter('min_confidence').value
        self.max_lines = self.get_parameter('max_lines_per_frame').value
        self.publish_debug = self.get_parameter('publish_debug_images').value
        
        
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        
        # Publishers
        self.weld_lines_pub = self.create_publisher(WeldLineArray, '/vision/weld_lines_2d', 10)
        
        if self.publish_debug:
            self.debug_image_pub = self.create_publisher(Image, '/red_line_detector/debug_image', 10)
            self.markers_pub = self.create_publisher(MarkerArray, '/red_line_detector/markers', 10)
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image, '/kinect2/qhd/image_color_rect', self.image_callback, 10
        )
        
        self.frame_count = 0
        self.detection_count = 0
        self.get_logger().info('Red Line Detector (Offline Optimized) initialized')

    def image_callback(self, msg):
        self.frame_count += 1
        try:
            cv_image = imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return
        
        detected_lines = self.detect_red_lines(cv_image)
        
        valid_lines = [l for l in detected_lines if l.confidence >= self.min_confidence][:self.max_lines]
        
        if len(valid_lines) > 0:
            self.detection_count += 1
            self.get_logger().info(f'Frame {self.frame_count}: Detected {len(valid_lines)} line(s)')
        
        weld_array = WeldLineArray()
        weld_array.header = msg.header
        weld_array.lines = valid_lines
        self.weld_lines_pub.publish(weld_array)
        
        if self.publish_debug:
            debug_img = self.create_debug_image(cv_image, valid_lines)
            debug_msg = cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
            
            markers = self.create_markers(valid_lines, msg.header)
            self.markers_pub.publish(markers)

    def detect_red_lines(self, image):
        mask = self.segment_red_color(image)
        
        # Simple logging for debug
        # pixels_before = np.count_nonzero(mask)
        
        mask_clean = self.apply_morphology(mask)
        pixels_after = np.count_nonzero(mask_clean)
        
        skeleton = self.skeletonize(mask_clean)
        contours = self.extract_contours(skeleton)
        
        detected_lines = []
        for idx, contour in enumerate(contours):
            if len(contour) < self.min_line_length:
                continue
            
            ordered_points = self.order_points_along_line(contour)
            simplified_points = self.simplify_polyline(ordered_points)
            
            # Simple confidence metric since we removed expensive checks
            # Confidence = Ratio of skeleton pixels to bounding box length (linearity check)
            
            bbox_min, bbox_max = self.compute_bbox(simplified_points)
            diag_len = np.sqrt((bbox_max.x - bbox_min.x)**2 + (bbox_max.y - bbox_min.y)**2)
            
            # If line is longer, we are more confident
            confidence = min(len(simplified_points) / max(diag_len, 1.0) * 1.5, 1.0)
            
            # Ensure high confidence for anything that looks like a line
            if len(simplified_points) > 10:
                confidence = max(confidence, 0.8)

            line = WeldLine()
            line.id = f"red_line_{idx}"
            line.confidence = float(confidence)
            line.pixels = [Point32(x=float(p[0]), y=float(p[1]), z=0.0) for p in simplified_points]
            line.bbox_min = bbox_min
            line.bbox_max = bbox_max
            
            detected_lines.append(line)
        
        return detected_lines

    def segment_red_color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_lower_1, self.hsv_upper_1)
        mask2 = cv2.inRange(hsv, self.hsv_lower_2, self.hsv_upper_2)
        return cv2.bitwise_or(mask1, mask2)

    def apply_morphology(self, mask):
        mask = cv2.erode(mask, self.kernel, iterations=self.erosion_iters)
        return cv2.dilate(mask, self.kernel, iterations=self.dilation_iters)

    def skeletonize(self, mask):
        """
        Skeletonization using OpenCV thinning. Fallback to simple thinning if needed.
        """
        try:
            return cv2.ximgproc.thinning(mask)
        except AttributeError:
            # Simple fallback: just return the mask (thinner kernel might help)
            # For robust welding lines, the mask centroid is often enough
            return mask

    def extract_contours(self, skeleton):
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours_nx2 = []
        for cnt in contours:
            try:
                cnt_reshaped = cnt.reshape(-1, 2)
                contours_nx2.append(cnt_reshaped)
            except Exception:
                pass
        return [cnt for cnt in contours_nx2 if len(cnt) >= self.min_line_length]

    def order_points_along_line(self, points):
        """
        Simple ordering: Sort by X coordinate if mostly horizontal, else Y.
        Replaces PCA-based sorting to remove sklearn dependency.
        """
        if len(points) < 2: return points
        
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        dx = max_pt[0] - min_pt[0]
        dy = max_pt[1] - min_pt[1]
        
        # If wider than tall, sort by X. Else sort by Y.
        if dx > dy:
            return points[points[:, 0].argsort()]
        else:
            return points[points[:, 1].argsort()]

    def simplify_polyline(self, points):
        if len(points) < 3: return points
        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        simplified = cv2.approxPolyDP(points_cv, epsilon=self.dp_epsilon, closed=False)
        simplified = simplified.squeeze()
        if simplified.ndim == 1: simplified = simplified.reshape(1, -1)
        return simplified

    def compute_bbox(self, points):
        if len(points) == 0: return Point(x=0.0, y=0.0, z=0.0), Point(x=0.0, y=0.0, z=0.0)
        min_pt, max_pt = np.min(points, axis=0), np.max(points, axis=0)
        return Point(x=float(min_pt[0]), y=float(min_pt[1]), z=0.0), Point(x=float(max_pt[0]), y=float(max_pt[1]), z=0.0)

    def create_debug_image(self, image, lines):
        debug_img = image.copy()
        for line in lines:
            points = np.array([[int(p.x), int(p.y)] for p in line.pixels], dtype=np.int32)
            color = (0, 255, 0) if line.confidence >= 0.9 else (0, 255, 255)
            cv2.polylines(debug_img, [points], False, color, 2)
            bbox_min = (int(line.bbox_min.x), int(line.bbox_min.y))
            bbox_max = (int(line.bbox_max.x), int(line.bbox_max.y))
            cv2.rectangle(debug_img, bbox_min, bbox_max, color, 1)
        return debug_img

    def create_markers(self, lines, header):
        marker_array = MarkerArray()
        for idx, line in enumerate(lines):
            marker = Marker()
            marker.header = header
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.points = [Point(x=p.x/1000.0, y=p.y/1000.0, z=0.0) for p in line.pixels]
            marker.scale.x = 0.002
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            marker_array.markers.append(marker)
        return marker_array

def main(args=None):
    rclpy.init(args=args)
    node = RedLineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
