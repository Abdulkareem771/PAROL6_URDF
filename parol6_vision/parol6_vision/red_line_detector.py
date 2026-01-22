#!/usr/bin/env python3
"""
Red Line Detector Node - Vision-Guided Welding Path Detection

This node detects red marker lines in camera images using computer vision techniques.
It processes RGB images from the Kinect v2 camera and publishes detected weld seams
as WeldLineArray messages.

================================================================================
ALGORITHM OVERVIEW
================================================================================

The detection pipeline consists of 5 main stages:

1. COLOR SEGMENTATION (HSV-based)
   - Convert RGB → HSV color space
   - Apply dual red color masks (handles HSV wraparound at 0°/180°)
   - Combine masks with bitwise OR

2. MORPHOLOGICAL PROCESSING
   - Erosion: Remove small noise pixels
   - Dilation: Fill gaps and connect nearby regions
   - Result: Clean binary mask of red regions

3. SKELETONIZATION
   - Extract centerline of thick red markers
   - Produces 1-pixel-wide line representation
   - Preserves line topology

4. CONTOUR DETECTION & ORDERING
   - Find connected components in skeleton
   - Sort points along principal direction (PCA)
   - Order points from start to end of line

5. LINE SIMPLIFICATION
   - Douglas-Peucker algorithm for polyline approximation
   - Reduces point count while preserving shape
   - Configurable epsilon parameter

================================================================================
CONFIDENCE COMPUTATION
================================================================================

Confidence score quantifies detection quality:

    confidence = (N_valid / N_total) × continuity_score

Where:
    N_valid = Valid pixels after morphological filtering
    N_total = Initial detected pixels (raw color mask)
    continuity_score = Line smoothness metric ∈ [0, 1]
    
continuity_score is computed from:
    - Douglas-Peucker compression ratio (fewer points = smoother)
    - Absence of sharp direction changes (low angle variance)

================================================================================
THESIS-READY STATEMENTS
================================================================================

For thesis documentation:

> "Red weld line detection employs HSV color space segmentation combined with
> morphological operations for robust marker extraction under varying lighting
> conditions. Skeletonization ensures sub-pixel precision in line localization."

> "Detection confidence quantifies both spatial coverage (retention of valid
> pixels after filtering) and geometric quality (line continuity), providing
> a unified metric for downstream quality assurance."

================================================================================
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from parol6_msgs.msg import WeldLine, WeldLineArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point32, Point
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
import cv2
import numpy as np
from skimage import morphology
from sklearn.decomposition import PCA


class RedLineDetector(Node):
    """
    Red Line Detector Node
    
    Detects red marker lines indicating welding seams using computer vision.
    
    Subscribed Topics:
        /kinect2/qhd/image_color_rect (sensor_msgs/Image): Rectified RGB image
        
    Published Topics:
        /vision/weld_lines_2d (parol6_msgs/WeldLineArray): Detected weld lines
        /red_line_detector/debug_image (sensor_msgs/Image): Visualization overlay
        /red_line_detector/markers (visualization_msgs/MarkerArray): RViz markers
        
    Parameters:
        hsv_lower_1, hsv_upper_1: Lower red HSV range (0-10°)
        hsv_lower_2, hsv_upper_2: Upper red HSV range (170-180°)
        morphology_kernel_size: Kernel size for morphological operations
        min_line_length: Minimum line length in pixels
        douglas_peucker_epsilon: Polyline simplification tolerance
        min_confidence: Minimum confidence threshold to publish
        publish_debug_images: Enable debug visualization
    """
    
    def __init__(self):
        super().__init__('red_line_detector')
        
        # ============================================================
        # DECLARE PARAMETERS
        # ============================================================
        
        # HSV Color Ranges - Red wraps around in HSV, so we use two ranges
        # Range 1: Low red (0-10 Hue degrees)
        #self.declare_parameter('hsv_lower_1', [0, 100, 100])
        #self.declare_parameter('hsv_upper_1', [10, 255, 255])
        
        # Range 2: High red (170-180 Hue degrees)
        self.declare_parameter('hsv_lower_2', [160, 160, 140])
        self.declare_parameter('hsv_upper_2', [180, 255, 180])
        
        # Morphological Operations
        self.declare_parameter('morphology_kernel_size', 5)
        self.declare_parameter('erosion_iterations', 1)
        self.declare_parameter('dilation_iterations', 2)
        
        # Line Extraction
        self.declare_parameter('min_line_length', 50)
        self.declare_parameter('min_contour_area', 400)
        self.declare_parameter('douglas_peucker_epsilon', 2.0)
        
        # Quality Thresholds
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('max_lines_per_frame', 5)
        
        # Performance
        self.declare_parameter('processing_rate', 10.0)
        self.declare_parameter('publish_debug_images', True)
        
        # ============================================================
        # GET PARAMETERS
        # ============================================================
        
        self.hsv_lower_1 = np.array(
            self.get_parameter('hsv_lower_1').value, dtype=np.uint8)
        self.hsv_upper_1 = np.array(
            self.get_parameter('hsv_upper_1').value, dtype=np.uint8)
        self.hsv_lower_2 = np.array(
            self.get_parameter('hsv_lower_2').value, dtype=np.uint8)
        self.hsv_upper_2 = np.array(
            self.get_parameter('hsv_upper_2').value, dtype=np.uint8)
        
        self.kernel_size = self.get_parameter('morphology_kernel_size').value
        self.erosion_iters = self.get_parameter('erosion_iterations').value
        self.dilation_iters = self.get_parameter('dilation_iterations').value
        
        self.min_line_length = self.get_parameter('min_line_length').value
        self.min_contour_area = self.get_parameter('min_contour_area').value
        self.dp_epsilon = self.get_parameter('douglas_peucker_epsilon').value
        
        self.min_confidence = self.get_parameter('min_confidence').value
        self.max_lines = self.get_parameter('max_lines_per_frame').value
        
        self.publish_debug = self.get_parameter('publish_debug_images').value
        
        # ============================================================
        # INITIALIZE CV BRIDGE
        # ============================================================
        
        self.bridge = CvBridge()
        
        # Create morphological kernel once (reused for performance)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.kernel_size, self.kernel_size)
        )
        
        # ============================================================
        # CREATE PUBLISHERS
        # ============================================================
        
        # Semantic output (used by downstream pipeline)
        self.weld_lines_pub = self.create_publisher(
            WeldLineArray,
            '/vision/weld_lines_2d',
            10
        )
        
        # Debug visualization (development only)
        if self.publish_debug:
            self.debug_image_pub = self.create_publisher(
                Image,
                '/red_line_detector/debug_image',
                10
            )
            
            self.markers_pub = self.create_publisher(
                MarkerArray,
                '/red_line_detector/markers',
                10
            )
        
        # ============================================================
        # CREATE SUBSCRIBER
        # ============================================================
        
        self.image_sub = self.create_subscription(
            Image,
            '/kinect2/qhd/image_color_rect',
            self.image_callback,
            10
        )
        
        # ============================================================
        # STATISTICS
        # ============================================================
        
        self.frame_count = 0
        self.detection_count = 0
        
        self.get_logger().info('Red Line Detector initialized')
        self.get_logger().info(f'HSV Range 1: {self.hsv_lower_1} to {self.hsv_upper_1}')
        self.get_logger().info(f'HSV Range 2: {self.hsv_lower_2} to {self.hsv_upper_2}')
        self.get_logger().info(f'Min confidence: {self.min_confidence}')
    
    # ================================================================
    # MAIN PROCESSING CALLBACK
    # ================================================================
    
    def image_callback(self, msg):
        """
        Main image processing callback.
        
        Processes incoming RGB images through the complete detection pipeline:
        1. Convert ROS Image → OpenCV
        2. Color segmentation (HSV)
        3. Morphological processing
        4. Skeletonization
        5. Contour detection and ordering
        6. Confidence computation
        7. Publish results
        
        Args:
            msg (sensor_msgs/Image): Input RGB image from camera
        """
        self.frame_count += 1
        
        try:
            # Convert ROS Image message to OpenCV format (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # DEBUG
            # self.get_logger().info(f"Received image: {cv_image.shape}")
        except Exception as e:
            self.get_logger().error(f'CV Bridge conversion failed: {e}')
            return
        
        # Run detection pipeline
        detected_lines = self.detect_red_lines(cv_image)
        
        # Filter by confidence
        valid_lines = [
            line for line in detected_lines 
            if line.confidence >= self.min_confidence
        ]
        
        # Limit number of detections
        valid_lines = valid_lines[:self.max_lines]
        
        if len(valid_lines) > 0:
            self.detection_count += 1
            self.get_logger().info(
                f'Frame {self.frame_count}: Detected {len(valid_lines)} line(s), '
                f'confidences: {[f"{l.confidence:.2f}" for l in valid_lines]}'
            )
        
        # Publish detected lines
        weld_array = WeldLineArray()
        weld_array.header = msg.header
        weld_array.lines = valid_lines
        self.weld_lines_pub.publish(weld_array)
        
        # Publish debug visualizations
        if self.publish_debug:
            debug_img = self.create_debug_image(cv_image, valid_lines)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
            
            markers = self.create_markers(valid_lines, msg.header)
            self.markers_pub.publish(markers)
    
    # ================================================================
    # DETECTION PIPELINE
    # ================================================================
    
    def detect_red_lines(self, image):
        """
        Complete red line detection pipeline.
        
        Args:
            image (np.ndarray): Input BGR image
            
        Returns:
            list[WeldLine]: List of detected weld lines with confidence scores
        """
        # Step 1: Color Segmentation
        mask = self.segment_red_color(image)
        pixels_before = np.count_nonzero(mask)
        # DEBUG LOGGING
        if pixels_before == 0:
            self.get_logger().warn("Segmentation returned 0 pixels!")
        else:
            self.get_logger().info(f"Segmentation found {pixels_before} pixels")
        
        # Step 2: Morphological Processing
        mask_clean = self.apply_morphology(mask)
        pixels_after = np.count_nonzero(mask_clean)
        # DEBUG
        self.get_logger().info(f"Morphology output: {pixels_after} pixels")
        
        # Step 3: Skeletonization
        skeleton = self.skeletonize(mask_clean)
        
        # Step 4: Extract Contours
        contours = self.extract_contours(skeleton)
        # DEBUG
        self.get_logger().info(f"Extracted {len(contours)} contours")
        
        # Step 5: Process each contour into a WeldLine
        detected_lines = []
        for idx, contour in enumerate(contours):
            # Filter by minimum length
            if len(contour) < self.min_line_length:
                # DEBUG
                # self.get_logger().info(f"Contour {idx} rejected: length {len(contour)} < {self.min_line_length}")
                continue
            
            # Order points along line
            ordered_points = self.order_points_along_line(contour)
            
            # Simplify with Douglas-Peucker
            simplified_points = self.simplify_polyline(ordered_points)
            
            # Compute confidence
            continuity = self.compute_continuity(simplified_points)
            retention = pixels_after / max(pixels_before, 1)
            confidence = retention * continuity
            # DEBUG
            self.get_logger().info(f"Line {idx}: Len={len(contour)}, Continuity={continuity:.2f}, Retention={retention:.2f}, Conf={confidence:.2f}")
            
            # Create WeldLine message
            line = WeldLine()
            line.id = f"red_line_{idx}"
            line.confidence = float(confidence)
            
            # Convert to Point32 (ROS message type)
            line.pixels = [
                Point32(x=float(pt[0]), y=float(pt[1]), z=0.0)
                for pt in simplified_points
            ]
            
            # Compute bounding box
            line.bbox_min, line.bbox_max = self.compute_bbox(simplified_points)
            
            detected_lines.append(line)
        
        return detected_lines
    
    def segment_red_color(self, image):
        """
        Segment red color using HSV color space.
        
        Red color wraps around in HSV (hue goes from 0° to 180° in OpenCV).
        We use two ranges to catch both low red (0-10°) and high red (170-180°).
        
        Args:
            image (np.ndarray): Input BGR image
            
        Returns:
            np.ndarray: Binary mask (uint8) where white = red pixels
        """
        # Convert BGR → HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, self.hsv_lower_1, self.hsv_upper_1)
        mask2 = cv2.inRange(hsv, self.hsv_lower_2, self.hsv_upper_2)
        
        # Combine masks (bitwise OR)
        mask = cv2.bitwise_or(mask1, mask2)
        
        return mask
    
    def apply_morphology(self, mask):
        """
        Apply morphological operations for noise reduction.
        
        Morphology operations:
        1. Erosion: Removes small noise pixels and thin connections
        2. Dilation: Fills gaps and connects nearby regions
        
        This is crucial for robust detection under non-ideal lighting.
        
        Args:
            mask (np.ndarray): Binary mask
            
        Returns:
            np.ndarray: Cleaned binary mask
        """
        # Erosion: Remove noise
        mask = cv2.erode(mask, self.kernel, iterations=self.erosion_iters)
        
        # Dilation: Fill gaps
        mask = cv2.dilate(mask, self.kernel, iterations=self.dilation_iters)
        
        return mask
    
    def skeletonize(self, mask):
        """
        Extract 1-pixel-wide skeleton from binary mask.
        
        Skeletonization reduces thick red markers to their centerline,
        providing sub-pixel precision for line localization.
        
        Uses scikit-image's morphological skeletonization algorithm.
        
        Args:
            mask (np.ndarray): Binary mask (uint8)
            
        Returns:
            np.ndarray: Binary skeleton (uint8)
        """
        # Convert to boolean for skimage
        mask_bool = mask > 0
        
        # Skeletonize
        skeleton_bool = morphology.skeletonize(mask_bool)
        
        # DEBUG
        skel_pixels = np.count_nonzero(skeleton_bool)
        self.get_logger().info(f"Skeleton pixels: {skel_pixels}")
        
        # Convert back to uint8
        skeleton = (skeleton_bool * 255).astype(np.uint8)
        
        return skeleton
    
    def extract_contours(self, skeleton):
        """
        Extract connected components (contours) from skeleton.
        
        Each contour represents a candidate weld line.
        
        Args:
            skeleton (np.ndarray): Binary skeleton image
            
        Returns:
            list[np.ndarray]: List of contours, each is Nx2 array of (x,y) points
        """
        contours, _ = cv2.findContours(
            skeleton,
            cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_NONE
        )
        
        # DEBUG
        self.get_logger().info(f"Raw contours from findContours: {len(contours)}")
        
        # Convert contours to simpler format (Nx2)
        contours_nx2 = []
        for cnt in contours:
            # Reshape from (N, 1, 2) to (N, 2)
            try:
                cnt_reshaped = cnt.reshape(-1, 2)
                contours_nx2.append(cnt_reshaped)
            except Exception as e:
                self.get_logger().warn(f"Failed to reshape contour: {e}")
        
        # Filter by length (number of points) instead of area
        # Skeleton has 0 area, so contourArea check was killing it!
        contours = [
            cnt for cnt in contours_nx2 
            if len(cnt) >= self.min_line_length
        ]
        
        self.get_logger().info(f"Contours after length filter: {len(contours)}")
        
        return contours
    
    def order_points_along_line(self, points):
        """
        Order points along the principal direction of the line.
        
        Uses PCA to find line direction, then sorts points along that direction.
        This ensures points go from start → end, not random order.
        
        Args:
            points (np.ndarray): Nx2 array of (x,y) points
            
        Returns:
            np.ndarray: Nx2 array of ordered points
        """
        if len(points) < 2:
            return points
        
        # Use PCA to find principal direction
        pca = PCA(n_components=1)
        pca.fit(points)
        
        # Project points onto principal component
        projections = pca.transform(points).flatten()
        
        # Sort by projection value
        sorted_indices = np.argsort(projections)
        ordered_points = points[sorted_indices]
        
        return ordered_points
    
    def simplify_polyline(self, points):
        """
        Simplify polyline using Douglas-Peucker algorithm.
        
        Reduces number of points while preserving overall shape.
        Epsilon parameter controls approximation tolerance.
        
        Args:
            points (np.ndarray): Nx2 array of points
            
        Returns:
            np.ndarray: Mx2 array of simplified points (M < N)
        """
        if len(points) < 3:
            return points
        
        # Reshape for OpenCV (needs Nx1x2 format)
        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply Douglas-Peucker
        simplified = cv2.approxPolyDP(
            points_cv,
            epsilon=self.dp_epsilon,
            closed=False
        )
        
        # Reshape back to Nx2
        simplified = simplified.squeeze()
        
        # Handle edge case where simplification returns single point
        if simplified.ndim == 1:
            simplified = simplified.reshape(1, -1)
        
        return simplified
    
    def compute_continuity(self, points):
        """
        Compute line continuity score ∈ [0, 1].
        
        Continuity measures line smoothness based on:
        1. Compression ratio: (simplified_points / original_points)
           - Fewer points after Douglas-Peucker = smoother line
        2. Angle variance: Standard deviation of direction changes
           - Low variance = straight/smooth, high variance = jagged
        
        Args:
            points (np.ndarray): Nx2 array of points
            
        Returns:
            float: Continuity score ∈ [0, 1]
        """
        if len(points) < 3:
            return 1.0
        
        # Compute angles between consecutive segments
        vectors = np.diff(points, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        
        # Compute angle changes
        angle_diffs = np.abs(np.diff(angles))
        
        # Normalize by wrapping to [-π, π]
        angle_diffs = np.where(angle_diffs > np.pi, 
                              2*np.pi - angle_diffs, 
                              angle_diffs)
        
        # Low angle variance = high continuity
        angle_variance = np.var(angle_diffs)
        
        # Map variance to [0, 1] using exponential decay
        # variance = 0 → score = 1, variance = large → score → 0
        continuity_score = np.exp(-angle_variance * 5.0)
        
        return float(np.clip(continuity_score, 0.0, 1.0))
    
    def compute_bbox(self, points):
        """
        Compute bounding box from points.
        
        Args:
            points (np.ndarray): Nx2 array of (x,y) points
            
        Returns:
            tuple: (bbox_min, bbox_max) as geometry_msgs/Point
        """
        if len(points) == 0:
            return Point(x=0.0, y=0.0, z=0.0), Point(x=0.0, y=0.0, z=0.0)
        
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        
        bbox_min = Point(x=float(min_pt[0]), y=float(min_pt[1]), z=0.0)
        bbox_max = Point(x=float(max_pt[0]), y=float(max_pt[1]), z=0.0)
        
        return bbox_min, bbox_max
    
    # ================================================================
    # VISUALIZATION
    # ================================================================
    
    def create_debug_image(self, image, lines):
        """
        Create visualization overlay showing detected lines.
        
        Args:
            image (np.ndarray): Original BGR image
            lines (list[WeldLine]): Detected lines
            
        Returns:
            np.ndarray: BGR image with overlay
        """
        debug_img = image.copy()
        
        # Draw each detected line
        for line in lines:
            # Extract points
            points = np.array(
                [[int(p.x), int(p.y)] for p in line.pixels],
                dtype=np.int32
            )
            
            # Choose color based on confidence
            if line.confidence >= 0.9:
                color = (0, 255, 0)  # Green = excellent
            elif line.confidence >= 0.7:
                color = (0, 255, 255)  # Yellow = good
            else:
                color = (0, 165, 255)  # Orange = acceptable
            
            # Draw polyline
            cv2.polylines(debug_img, [points], False, color, 2)
            
            # Draw bounding box
            bbox_min = (int(line.bbox_min.x), int(line.bbox_min.y))
            bbox_max = (int(line.bbox_max.x), int(line.bbox_max.y))
            cv2.rectangle(debug_img, bbox_min, bbox_max, color, 1)
            
            # Draw label
            label = f"{line.id}: {line.confidence:.2f}"
            cv2.putText(
                debug_img, label, 
                (bbox_min[0], bbox_min[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        
        return debug_img
    
    def create_markers(self, lines, header):
        """
        Create RViz MarkerArray for visualization.
        
        Args:
            lines (list[WeldLine]): Detected lines
            header (std_msgs/Header): Header for markers
            
        Returns:
            visualization_msgs/MarkerArray: RViz markers
        """
        marker_array = MarkerArray()
        
        for idx, line in enumerate(lines):
            # Line strip marker
            marker = Marker()
            marker.header = header
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Set points (convert Point32 → Point for marker)
            marker.points = [
                Point(x=p.x / 1000.0, y=p.y / 1000.0, z=0.0)  # Convert pixels → meters
                for p in line.pixels
            ]
            
            # Set appearance based on confidence
            marker.scale.x = 0.002  # 2mm line width
            
            if line.confidence >= 0.9:
                color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
            elif line.confidence >= 0.7:
                color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)  # Yellow
            else:
                color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0) # Orange
            
            marker.color = color
            marker_array.markers.append(marker)
        
        return marker_array


def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)
    
    node = RedLineDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Print final statistics
        node.get_logger().info(
            f'Shutting down. Processed {node.frame_count} frames, '
            f'detected lines in {node.detection_count} frames '
            f'({100.0*node.detection_count/max(node.frame_count,1):.1f}%)'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
