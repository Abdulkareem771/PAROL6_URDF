#!/usr/bin/env python3
"""
Path Optimizer Node - Vision-Guided Welding Path Detection

This node detects red marker lines in camera images using computer vision
techniques and publishes the best detected weld line per frame.

Unlike the red_line_detector node, path_optimizer:
  - Accepts lines of ALL lengths (no minimum length / contour-area filter)
  - Publishes exactly ONE line per frame (the highest-confidence contour)
  - Uses /path_optimizer/ prefix for debug topics

================================================================================
ALGORITHM OVERVIEW
================================================================================

1. COLOR SEGMENTATION (HSV-based)
   - Convert RGB → HSV color space
   - Apply dual red color masks (handles HSV wraparound at 0°/180°)
   - Combine masks with bitwise OR

2. MORPHOLOGICAL PROCESSING
   - Erosion:  Remove salt-pepper noise and small artifacts
   - Dilation: Fill gaps and connect fragmented line segments
   - Result:   Clean binary mask of red regions

3. SKELETONIZATION
   - Reduce thick mask to a 1-pixel-wide centerline
   - Preserves line topology for downstream usage
   - Uses scikit-image morphological skeletonization

4. CONTOUR DETECTION (no length filter)
   - Find connected components in the skeleton
   - ALL contours accepted regardless of point count

5. PCA-based POINT ORDERING
   - Sort points along the principal direction of each contour
   - Ensures points are ordered start → end

6. LINE SIMPLIFICATION (Douglas-Peucker)
   - Reduce point count while preserving overall shape
   - Configurable epsilon parameter

7. CONFIDENCE SCORING
   - confidence = retention × continuity_score ∈ [0, 1]
   - Best-confidence line selected and published

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


class PathOptimizer(Node):
    """
    Path Optimizer Node

    Detects red marker lines indicating welding seams. Designed for variant-
    length lines: no minimum-length filter is applied. Publishes the single
    highest-confidence detected line per frame.

    Subscribed Topics:
        /vision/processing_mode/annotated_image (sensor_msgs/Image)

    Published Topics:
        /vision/weld_lines_2d         (parol6_msgs/WeldLineArray)
        /path_optimizer/debug_image   (sensor_msgs/Image)
        /path_optimizer/markers       (visualization_msgs/MarkerArray)

    Parameters:
        hsv_lower_1 / hsv_upper_1: Lower red HSV range  (0–10°)
        hsv_lower_2 / hsv_upper_2: Upper red HSV range  (160–180°)
        morphology_kernel_size:    Kernel size for morphological operations
        erosion_iterations:        Number of erosion passes
        dilation_iterations:       Number of dilation passes
        douglas_peucker_epsilon:   Polyline simplification tolerance
        min_confidence:            Minimum confidence threshold to publish
        publish_debug_images:      Enable debug visualization topics
    """

    def __init__(self):
        super().__init__('path_optimizer')

        # ============================================================
        # DECLARE PARAMETERS
        # ============================================================

        # HSV Color Ranges — red wraps around in HSV, so two ranges are used
        # Range 1: Low red (0–10 Hue degrees)
        self.declare_parameter('hsv_lower_1', [0, 100, 100])
        self.declare_parameter('hsv_upper_1', [10, 255, 255])

        # Range 2: High red (160–180 Hue degrees)
        self.declare_parameter('hsv_lower_2', [160, 50, 0])
        self.declare_parameter('hsv_upper_2', [180, 255, 255])

        # Morphological Operations
        self.declare_parameter('morphology_kernel_size', 3)
        self.declare_parameter('erosion_iterations', 0)
        self.declare_parameter('dilation_iterations', 2)

        # Line Extraction
        self.declare_parameter('douglas_peucker_epsilon', 2.0)

        # Quality Thresholds
        self.declare_parameter('min_confidence', 0.5)

        # Performance / Output
        self.declare_parameter('publish_debug_images', True)

        # ============================================================
        # LOAD PARAMETERS
        # ============================================================

        self.hsv_lower_1 = np.array(
            self.get_parameter('hsv_lower_1').value, dtype=np.uint8)
        self.hsv_upper_1 = np.array(
            self.get_parameter('hsv_upper_1').value, dtype=np.uint8)
        self.hsv_lower_2 = np.array(
            self.get_parameter('hsv_lower_2').value, dtype=np.uint8)
        self.hsv_upper_2 = np.array(
            self.get_parameter('hsv_upper_2').value, dtype=np.uint8)

        self.kernel_size    = self.get_parameter('morphology_kernel_size').value
        self.erosion_iters  = self.get_parameter('erosion_iterations').value
        self.dilation_iters = self.get_parameter('dilation_iterations').value

        self.dp_epsilon     = self.get_parameter('douglas_peucker_epsilon').value
        self.min_confidence = self.get_parameter('min_confidence').value
        self.publish_debug  = self.get_parameter('publish_debug_images').value

        # ============================================================
        # CV BRIDGE & MORPHOLOGY KERNEL
        # ============================================================

        self.bridge = CvBridge()

        # Pre-build kernel (reused every frame for performance)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.kernel_size, self.kernel_size)
        )

        # ============================================================
        # PUBLISHERS
        # ============================================================

        # Primary semantic output
        self.weld_lines_pub = self.create_publisher(
            WeldLineArray,
            '/vision/weld_lines_2d',
            10
        )

        # Debug visualizations (optional)
        if self.publish_debug:
            self.debug_image_pub = self.create_publisher(
                Image,
                '/path_optimizer/debug_image',
                10
            )
            self.markers_pub = self.create_publisher(
                MarkerArray,
                '/path_optimizer/markers',
                10
            )

        # ============================================================
        # SUBSCRIBER
        # ============================================================

        self.image_sub = self.create_subscription(
            Image,
            '/vision/processing_mode/annotated_image',
            self.image_callback,
            10
        )

        # ============================================================
        # STATISTICS
        # ============================================================

        self.frame_count     = 0
        self.detection_count = 0

        self.get_logger().info('Path Optimizer initialized')
        self.get_logger().info(
            f'HSV Range 1: {self.hsv_lower_1} → {self.hsv_upper_1}')
        self.get_logger().info(
            f'HSV Range 2: {self.hsv_lower_2} → {self.hsv_upper_2}')
        self.get_logger().info(
            f'Morphology kernel: {self.kernel_size}px, '
            f'erosion={self.erosion_iters}, dilation={self.dilation_iters}')
        self.get_logger().info(
            f'Min confidence: {self.min_confidence}  |  '
            f'Debug images: {self.publish_debug}')

    # ================================================================
    # MAIN PROCESSING CALLBACK
    # ================================================================

    def image_callback(self, msg):
        """
        Main image processing callback.

        Pipeline per frame:
            1. ROS Image → OpenCV (BGR)
            2. HSV color segmentation
            3. Morphological cleanup
            4. Skeletonization
            5. Contour detection (no length filter)
            6. PCA ordering + Douglas-Peucker + confidence
            7. Select best line (highest confidence ≥ min_confidence)
            8. Publish WeldLineArray (0 or 1 line) + debug output

        Args:
            msg (sensor_msgs/Image): Annotated image from processing mode
        """
        self.frame_count += 1

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge conversion failed: {e}')
            return

        # ---- Run full detection pipeline ----
        best_line = self.detect_best_line(cv_image)

        # ---- Build WeldLineArray (0 or 1 entry) ----
        weld_array = WeldLineArray()
        weld_array.header = msg.header

        if best_line is not None:
            weld_array.lines = [best_line]
            self.detection_count += 1
            self.get_logger().info(
                f'Frame {self.frame_count}: line detected  '
                f'confidence={best_line.confidence:.2f}  '
                f'points={len(best_line.pixels)}'
            )
        else:
            weld_array.lines = []
            self.get_logger().info(
                f'Frame {self.frame_count}: no line above '
                f'min_confidence={self.min_confidence:.2f}'
            )

        self.weld_lines_pub.publish(weld_array)

        # ---- Publish debug visualizations ----
        if self.publish_debug:
            debug_img = self.create_debug_image(cv_image, weld_array.lines)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)

            markers = self.create_markers(weld_array.lines, msg.header)
            self.markers_pub.publish(markers)

    # ================================================================
    # DETECTION PIPELINE
    # ================================================================

    def detect_best_line(self, image):
        """
        Run the full pipeline and return the single best WeldLine, or None.

        Steps:
            1. Color segmentation
            2. Morphology
            3. Skeletonization
            4. Contour extraction (all sizes)
            5. Per-contour: PCA ordering → Douglas-Peucker → confidence
            6. Return the contour with the highest confidence ≥ min_confidence

        Args:
            image (np.ndarray): Input BGR image

        Returns:
            WeldLine | None
        """
        # Step 1: Color segmentation
        mask = self.segment_red_color(image)
        pixels_before = int(np.count_nonzero(mask))

        if pixels_before == 0:
            self.get_logger().warn('Segmentation returned 0 red pixels')
            return None

        self.get_logger().info(f'Segmentation: {pixels_before} red pixels')

        # Step 2: Morphological cleanup
        mask_clean = self.apply_morphology(mask)
        pixels_after = int(np.count_nonzero(mask_clean))
        self.get_logger().info(f'After morphology: {pixels_after} pixels')

        if pixels_after == 0:
            return None

        # Step 3: Skeletonization
        skeleton = self.skeletonize(mask_clean)
        skel_pixels = int(np.count_nonzero(skeleton))
        self.get_logger().info(f'Skeleton pixels: {skel_pixels}')

        if skel_pixels == 0:
            return None

        # Step 4: Contour extraction (ALL contours, no length gate)
        contours = self.extract_contours(skeleton)
        self.get_logger().info(f'Contours found: {len(contours)}')

        if not contours:
            return None

        # Step 5 & 6: Score each contour, keep the best one
        retention = pixels_after / max(pixels_before, 1)
        best_line      = None
        best_confidence = -1.0

        for idx, contour in enumerate(contours):
            ordered_pts  = self.order_points_along_line(contour)
            simplified   = self.simplify_polyline(ordered_pts)
            continuity   = self.compute_continuity(simplified)
            confidence   = float(retention * continuity)

            self.get_logger().info(
                f'  Contour {idx}: pts={len(contour)}  '
                f'continuity={continuity:.2f}  conf={confidence:.2f}'
            )

            if confidence >= self.min_confidence and confidence > best_confidence:
                best_confidence = confidence

                line = WeldLine()
                line.id         = 'path_optimizer_line'
                line.confidence = confidence

                # Dense skeleton points for downstream 3-D reconstruction
                line.pixels = [
                    Point32(x=float(pt[0]), y=float(pt[1]), z=0.0)
                    for pt in ordered_pts
                ]

                line.bbox_min, line.bbox_max = self.compute_bbox(ordered_pts)
                best_line = line

        return best_line

    # ================================================================
    # STAGE 1 — COLOR SEGMENTATION
    # ================================================================

    def segment_red_color(self, image):
        """
        Segment red pixels using HSV dual-range masking.

        Red hue wraps around 0°/180° in OpenCV HSV (H ∈ [0, 180]).
        Two inRange calls are combined with bitwise OR.

        Args:
            image (np.ndarray): BGR input image

        Returns:
            np.ndarray: uint8 binary mask — 255 = red pixel, 0 = other
        """
        hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_lower_1, self.hsv_upper_1)
        mask2 = cv2.inRange(hsv, self.hsv_lower_2, self.hsv_upper_2)
        return cv2.bitwise_or(mask1, mask2)

    # ================================================================
    # STAGE 2 — MORPHOLOGICAL PROCESSING
    # ================================================================

    def apply_morphology(self, mask):
        """
        Apply erosion then dilation to clean up the binary mask.

        - Erosion:  removes isolated noise pixels and thin salt artifacts
        - Dilation: reconnects nearby fragments and fills small holes

        Args:
            mask (np.ndarray): Binary uint8 mask

        Returns:
            np.ndarray: Cleaned binary mask
        """
        mask = cv2.erode(mask,  self.kernel, iterations=self.erosion_iters)
        mask = cv2.dilate(mask, self.kernel, iterations=self.dilation_iters)
        return mask

    # ================================================================
    # STAGE 3 — SKELETONIZATION
    # ================================================================

    def skeletonize(self, mask):
        """
        Reduce the thick binary mask to a 1-pixel-wide centerline.

        Uses scikit-image morphological skeletonization. The centerline
        preserves topology and is suitable for sub-pixel line localization.

        Args:
            mask (np.ndarray): uint8 binary mask

        Returns:
            np.ndarray: uint8 skeleton (255 = centerline pixel)
        """
        skeleton_bool = morphology.skeletonize(mask > 0)
        return (skeleton_bool * 255).astype(np.uint8)

    # ================================================================
    # STAGE 4 — CONTOUR EXTRACTION (no length filter)
    # ================================================================

    def extract_contours(self, skeleton):
        """
        Extract connected components from the skeleton image.

        Unlike red_line_detector, NO minimum length / area filter is applied,
        so lines of any size are returned.

        Args:
            skeleton (np.ndarray): uint8 binary skeleton

        Returns:
            list[np.ndarray]: Each element is an (N, 2) array of (x, y) points
        """
        raw_contours, _ = cv2.findContours(
            skeleton,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE
        )

        contours = []
        for cnt in raw_contours:
            try:
                contours.append(cnt.reshape(-1, 2))
            except Exception as e:
                self.get_logger().warn(f'Contour reshape error: {e}')

        return contours

    # ================================================================
    # STAGE 5 — PCA-BASED POINT ORDERING
    # ================================================================

    def order_points_along_line(self, points):
        """
        Sort points along the principal direction of the line using PCA.

        Projects each point onto the first principal component and sorts
        by that projection value, giving a start-to-end ordering.

        Args:
            points (np.ndarray): (N, 2) array of unordered (x, y) points

        Returns:
            np.ndarray: (N, 2) array ordered from start → end
        """
        if len(points) < 2:
            return points

        pca = PCA(n_components=1)
        pca.fit(points)
        projections    = pca.transform(points).flatten()
        sorted_indices = np.argsort(projections)
        return points[sorted_indices]

    # ================================================================
    # STAGE 6 — DOUGLAS-PEUCKER SIMPLIFICATION
    # ================================================================

    def simplify_polyline(self, points):
        """
        Simplify the polyline using the Douglas-Peucker algorithm.

        Reduces point count while preserving overall geometry.

        Args:
            points (np.ndarray): (N, 2) ordered point array

        Returns:
            np.ndarray: (M, 2) simplified point array, M ≤ N
        """
        if len(points) < 3:
            return points

        points_cv  = points.reshape(-1, 1, 2).astype(np.float32)
        simplified = cv2.approxPolyDP(
            points_cv,
            epsilon=self.dp_epsilon,
            closed=False
        ).squeeze()

        if simplified.ndim == 1:
            simplified = simplified.reshape(1, -1)

        return simplified

    # ================================================================
    # STAGE 7 — CONFIDENCE SCORING
    # ================================================================

    def compute_continuity(self, points):
        """
        Compute line continuity score ∈ [0, 1].

        Based on angle variance between consecutive segments:
        - Low variance  → smooth, continuous line  → score near 1
        - High variance → jagged, fragmented line  → score near 0

        Formula:
            continuity = exp(−variance × 5)

        Args:
            points (np.ndarray): (N, 2) simplified point array

        Returns:
            float: Continuity score ∈ [0, 1]
        """
        if len(points) < 3:
            return 1.0

        vectors     = np.diff(points, axis=0)
        angles      = np.arctan2(vectors[:, 1], vectors[:, 0])
        angle_diffs = np.abs(np.diff(angles))

        # Wrap to [0, π]
        angle_diffs = np.where(
            angle_diffs > np.pi,
            2 * np.pi - angle_diffs,
            angle_diffs
        )

        angle_variance   = np.var(angle_diffs)
        continuity_score = np.exp(-angle_variance * 5.0)
        return float(np.clip(continuity_score, 0.0, 1.0))

    # ================================================================
    # UTILITY — BOUNDING BOX
    # ================================================================

    def compute_bbox(self, points):
        """
        Compute axis-aligned bounding box from a set of points.

        Args:
            points (np.ndarray): (N, 2) point array

        Returns:
            tuple[Point, Point]: (bbox_min, bbox_max)
        """
        if len(points) == 0:
            origin = Point(x=0.0, y=0.0, z=0.0)
            return origin, origin

        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)

        return (
            Point(x=float(min_pt[0]), y=float(min_pt[1]), z=0.0),
            Point(x=float(max_pt[0]), y=float(max_pt[1]), z=0.0)
        )

    # ================================================================
    # VISUALIZATION — DEBUG IMAGE
    # ================================================================

    def create_debug_image(self, image, lines):
        """
        Draw detected lines over the original image.

        Color coding by confidence:
            Green  (≥ 0.9): excellent
            Yellow (≥ 0.7): good
            Orange (< 0.7): acceptable

        Args:
            image (np.ndarray): Original BGR image
            lines (list[WeldLine]): Detected lines (0 or 1 item)

        Returns:
            np.ndarray: BGR debug image
        """
        debug_img = image.copy()

        for line in lines:
            pts = np.array(
                [[int(p.x), int(p.y)] for p in line.pixels],
                dtype=np.int32
            )

            if line.confidence >= 0.9:
                color = (0, 255, 0)    # Green
            elif line.confidence >= 0.7:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange

            cv2.polylines(debug_img, [pts], False, color, 2)

            bbox_min = (int(line.bbox_min.x), int(line.bbox_min.y))
            bbox_max = (int(line.bbox_max.x), int(line.bbox_max.y))
            cv2.rectangle(debug_img, bbox_min, bbox_max, color, 1)

            label = f'{line.id}: {line.confidence:.2f}'
            cv2.putText(
                debug_img, label,
                (bbox_min[0], max(bbox_min[1] - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        return debug_img

    # ================================================================
    # VISUALIZATION — RVIZ MARKERS
    # ================================================================

    def create_markers(self, lines, header):
        """
        Build a MarkerArray for RViz visualization.

        Each line is represented as a LINE_STRIP marker.
        Marker colour matches the confidence colour-coding of the debug image.

        Args:
            lines (list[WeldLine]): Detected lines (0 or 1 item)
            header: ROS message header

        Returns:
            visualization_msgs/MarkerArray
        """
        marker_array = MarkerArray()

        for idx, line in enumerate(lines):
            marker             = Marker()
            marker.header      = header
            marker.ns          = 'path_optimizer'
            marker.id          = idx
            marker.type        = Marker.LINE_STRIP
            marker.action      = Marker.ADD

            # Normalize pixel coordinates to approximate 3-D space
            marker.points = [
                Point(
                    x=(p.x / 1000.0) * 0.45,
                    y=(p.y / 1000.0) * 0.45,
                    z=0.45
                )
                for p in line.pixels
            ]

            marker.scale.x = 0.002  # 2 mm line width

            if line.confidence >= 0.9:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            elif line.confidence >= 0.7:
                marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            else:
                marker.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0)

            marker_array.markers.append(marker)

        return marker_array


# ====================================================================
# ENTRY POINT
# ====================================================================

def main(args=None):
    """ROS 2 node entry point."""
    rclpy.init(args=args)
    node = PathOptimizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Shutting down.  Frames processed: {node.frame_count}  |  '
            f'Lines detected: {node.detection_count}  |  '
            f'Detection rate: '
            f'{100.0 * node.detection_count / max(node.frame_count, 1):.1f}%'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
