#!/usr/bin/env python3
"""
Yolo_segment Node - YOLO Instance Segmentation for Seam Intersection Detection

This node is a ROS 2 port of ``phase_2_first_mode.py``.
It subscribes to an input camera image topic, runs YOLO instance segmentation
to identify two workpiece objects, computes their expanded-mask intersection
(the weld seam region), and publishes:

  * A **debug image** overlaying all intermediate contours on the raw frame.
  * An **annotated image** showing only the filled intersection contour (bgr8).
  * The **centroid** of the intersection region as a PointStamped message.

================================================================================
ALGORITHM OVERVIEW
================================================================================

1. YOLO INFERENCE
   - Load ``ultralytics`` YOLO segmentation model from ``model_path`` parameter.
   - Run inference on every incoming camera frame.

2. MASK EXTRACTION
   - Pull the two largest segmentation masks from the result.
   - Resize each mask to the original image dimensions.
   - Binarise (threshold > 0.5 → 0 or 255).

3. MORPHOLOGICAL CLEANING
   - Apply morphological OPEN (5×5 kernel) to each binary mask to remove noise.

4. MASK DILATION (seam expansion)
   - Dilate each cleaned mask outward by ``expand_px`` pixels using an
     elliptical structuring element.  This widens the seam search region.

5. INTERSECTION
   - Bitwise-AND the two dilated masks → intersection region.
   - Find the largest external contour of the intersection.

6. PUBLISH
   - Debug image  : raw frame + original contours (blue) + expanded contours
                    (green) + filled intersection (red).
   - Annotated image: clean copy of raw frame + filled intersection only (red).
   - Seam centroid : geometry_msgs/PointStamped with pixel-space (x, y, z=0).

================================================================================
SUBSCRIBED TOPICS
================================================================================
  /vision/captured_image_color  (sensor_msgs/Image)  – input camera frame

================================================================================
PUBLISHED TOPICS
================================================================================
  /yolo_segment/debug_image      (sensor_msgs/Image)         – full debug overlay
  /yolo_segment/annotated_image  (sensor_msgs/Image, bgr8)   – intersection only
  /yolo_segment/seam_centroid    (geometry_msgs/PointStamped) – seam centroid px

================================================================================
PARAMETERS
================================================================================
  model_path    (string)  – Absolute path to YOLO best.pt weights file.
  image_topic   (string)  – Input image topic to subscribe to.
  expand_px     (int)     – Pixels to dilate each object mask outward (default 8).
  publish_debug     (bool)    – Whether to publish the debug overlay image.
  mask_conf         (float)   – Minimum confidence threshold for YOLO detections (default 0.5).
  print_detections  (bool)    – Print the number of detected objects per frame in real time (default True).

================================================================================
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

# ultralytics is installed system-wide inside the Docker container
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Default model path inside the Docker container
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = (
    '/workspace/venvs/vision_venvs/ultralytics_cpu_env'
    '/yolo_segmentation_models_results/experiment_2/weights/best.pt'
)

_DEFAULT_IMAGE_TOPIC = '/vision/captured_image_color'


class YoloSegmentNode(Node):
    """
    ROS 2 node that performs YOLO segmentation and seam-intersection detection.

    Subscribed Topics:
        <image_topic>  (sensor_msgs/Image): Input colour image from camera.

    Published Topics:
        /yolo_segment/debug_image      (sensor_msgs/Image): Full debug overlay.
        /yolo_segment/annotated_image  (sensor_msgs/Image): Intersection-only overlay (bgr8).
        /yolo_segment/seam_centroid    (geometry_msgs/PointStamped): Seam centroid in pixels.

    Parameters:
        model_path    (str):   Absolute path to YOLO weights (best.pt).
        image_topic   (str):   Camera topic to subscribe to.
        expand_px         (int):   Dilation radius for mask expansion (default 8).
        publish_debug     (bool):  Publish the debug overlay image (default True).
        mask_conf         (float): Minimum confidence threshold for YOLO detections (default 0.5).
        print_detections  (bool):  Print detected-object count every frame in real time (default True).
    """

    def __init__(self):
        super().__init__('Yolo_segment')

        # ------------------------------------------------------------------ #
        # DECLARE PARAMETERS                                                   #
        # ------------------------------------------------------------------ #
        self.declare_parameter('model_path', _DEFAULT_MODEL_PATH)
        self.declare_parameter('image_topic', _DEFAULT_IMAGE_TOPIC)
        self.declare_parameter('expand_px', 8)
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('mask_conf', 0.5)
        self.declare_parameter('print_detections', True)

        # ------------------------------------------------------------------ #
        # READ PARAMETERS                                                      #
        # ------------------------------------------------------------------ #
        self.model_path = self.get_parameter('model_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.expand_px = int(self.get_parameter('expand_px').value)
        self.publish_debug = self.get_parameter('publish_debug').value
        self.mask_conf = float(self.get_parameter('mask_conf').value)
        self.print_detections = self.get_parameter('print_detections').value

        # ------------------------------------------------------------------ #
        # LOAD YOLO MODEL                                                      #
        # ------------------------------------------------------------------ #
        self.get_logger().info(f'Loading YOLO model from: {self.model_path}')
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info('YOLO model loaded successfully.')
        except Exception as exc:
            self.get_logger().error(f'Failed to load YOLO model: {exc}')
            raise

        # ------------------------------------------------------------------ #
        # CV BRIDGE                                                            #
        # ------------------------------------------------------------------ #
        self.bridge = CvBridge()

        # Pre-build structuring elements (reused every frame)
        self._morph_kernel = np.ones((5, 5), np.uint8)
        self._dil_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * self.expand_px + 1, 2 * self.expand_px + 1),
        )

        # ------------------------------------------------------------------ #
        # PUBLISHERS                                                           #
        # ------------------------------------------------------------------ #
        self.annotated_pub = self.create_publisher(
            Image, '/yolo_segment/annotated_image', 10
        )
        self.seam_centroid_pub = self.create_publisher(
            PointStamped, '/yolo_segment/seam_centroid', 10
        )
        if self.publish_debug:
            self.debug_pub = self.create_publisher(
                Image, '/yolo_segment/debug_image', 10
            )

        # ------------------------------------------------------------------ #
        # SUBSCRIBER                                                           #
        # ------------------------------------------------------------------ #
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            10,
        )

        # ------------------------------------------------------------------ #
        # COUNTERS                                                             #
        # ------------------------------------------------------------------ #
        self._frame_count = 0
        self._detection_count = 0

        self.get_logger().info(
            f'Yolo_segment node initialized.\n'
            f'  Subscribed to    : {self.image_topic}\n'
            f'  Expand px        : {self.expand_px}\n'
            f'  Mask conf        : {self.mask_conf}\n'
            f'  Debug images     : {self.publish_debug}\n'
            f'  Print detections : {self.print_detections}'
        )

    # ---------------------------------------------------------------------- #
    # MAIN CALLBACK                                                            #
    # ---------------------------------------------------------------------- #

    def _image_callback(self, msg: Image) -> None:
        """Process one incoming camera frame through the full pipeline."""
        self._frame_count += 1

        # Convert ROS Image → OpenCV BGR
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'CvBridge conversion failed: {exc}')
            return

        # Run the segmentation pipeline
        annotated_img, debug_img, centroid_px, num_detections = self._run_pipeline(img)

        # ── Real-time detection count ──
        if self.print_detections:
            self.get_logger().info(
                f'Frame {self._frame_count}: YOLO detected {num_detections} object(s)'
            )

        
        # Publish annotated image (always)
        ann_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        """
        # Publish annotated image (always) – convert BGR→RGB before encoding
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        ann_msg = self.bridge.cv2_to_imgmsg(annotated_rgb, encoding='rgb8')
        """
        ann_msg.header = msg.header
        self.annotated_pub.publish(ann_msg)

        # Publish debug image (optional)
        if self.publish_debug and debug_img is not None:
            dbg_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            dbg_msg.header = msg.header
            self.debug_pub.publish(dbg_msg)

        # Publish centroid
        if centroid_px is not None:
            self._detection_count += 1
            cx, cy = centroid_px
            pt_msg = PointStamped()
            pt_msg.header = msg.header
            pt_msg.point.x = float(cx)
            pt_msg.point.y = float(cy)
            pt_msg.point.z = 0.0
            self.seam_centroid_pub.publish(pt_msg)
            self.get_logger().info(
                f'Frame {self._frame_count}: seam centroid at '
                f'({cx:.1f}, {cy:.1f}) px'
            )
        else:
            self.get_logger().warn(
                f'Frame {self._frame_count}: no intersection found '
                f'(need ≥2 detected objects).'
            )

    # ---------------------------------------------------------------------- #
    # PIPELINE                                                                 #
    # ---------------------------------------------------------------------- #

    def _run_pipeline(self, img: np.ndarray):
        """
        Execute the full segmentation and intersection pipeline.

        Args:
            img: OpenCV BGR image (H×W×3, uint8).

        Returns:
            annotated_img  – BGR image with filled intersection contour only.
            debug_img      – BGR image with all intermediate overlays, or None
                             if publish_debug is False.
            centroid_px    – (cx, cy) tuple in pixel coords, or None if not found.
            num_detections – int, number of objects detected by YOLO this frame.
        """
        h, w = img.shape[:2]
        annotated_img = img.copy()
        debug_img = img.copy() if self.publish_debug else None

        # 1 ─ YOLO Inference ------------------------------------------------
        results = self.model(img, verbose=False, conf=self.mask_conf)
        result = results[0]

        # Count detections above the confidence threshold
        num_detections = len(result.boxes) if result.boxes is not None else 0

        if result.masks is None or len(result.masks.data) < 2:
            # Not enough objects detected – return unmodified frames
            return annotated_img, debug_img, None, num_detections

        # 2 ─ Extract & Binarise Masks (take the first two objects) ----------
        raw_masks = result.masks.data
        obj_matrices = []
        for mask_tensor in raw_masks[:2]:          # only need first two
            mask_np = mask_tensor.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h))
            mask_binary = (mask_resized > self.mask_conf).astype(np.uint8) * 255
            obj_matrices.append(mask_binary)

        obj_1, obj_2 = obj_matrices

        # 3 ─ Morphological OPEN (noise removal) -----------------------------
        obj_1 = cv2.morphologyEx(obj_1, cv2.MORPH_OPEN, self._morph_kernel)
        obj_2 = cv2.morphologyEx(obj_2, cv2.MORPH_OPEN, self._morph_kernel)

        # 4 ─ Find original contours (for debug overlay) ---------------------
        contour_obj1 = self._find_largest_contour(obj_1)
        contour_obj2 = self._find_largest_contour(obj_2)

        # 5 ─ Dilation (seam expansion) --------------------------------------
        obj_1_exp = cv2.dilate(obj_1, self._dil_kernel)
        obj_2_exp = cv2.dilate(obj_2, self._dil_kernel)

        contour_obj1_exp = self._find_largest_contour(obj_1_exp)
        contour_obj2_exp = self._find_largest_contour(obj_2_exp)

        # 6 ─ Intersection ---------------------------------------------------
        intersection_mask = cv2.bitwise_and(obj_1_exp, obj_2_exp)
        contour_I = self._find_largest_contour(intersection_mask)

        # 7 ─ Draw on annotated_img (intersection only, filled red) ----------
        centroid_px = None
        if contour_I is not None:
            cv2.drawContours(annotated_img, [contour_I], -1, (0, 0, 255), -1)

            # Compute centroid using image moments
            M = cv2.moments(contour_I)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centroid_px = (cx, cy)
                # Mark centroid with a white crosshair on annotated image
                """                
                self._draw_crosshair(
                    annotated_img, int(cx), int(cy), (255, 255, 255)
                )"""
                """
                # Draw a small white filled circle at the centroid
                radius = 3
                cv2.circle(annotated_img, (int(cx), int(cy)), radius, (255, 255, 255), -1)
                """


        # 8 ─ Draw on debug_img (all layers) ---------------------------------
        if self.publish_debug and debug_img is not None:
            # Original object contours – blue
            if contour_obj1 is not None:
                cv2.drawContours(
                    debug_img, [contour_obj1], -1, (255, 0, 0), 2
                )
            if contour_obj2 is not None:
                cv2.drawContours(
                    debug_img, [contour_obj2], -1, (255, 0, 0), 2
                )
            # Expanded contours – green
            if contour_obj1_exp is not None:
                cv2.drawContours(
                    debug_img, [contour_obj1_exp], -1, (0, 255, 0), 2
                )
            if contour_obj2_exp is not None:
                cv2.drawContours(
                    debug_img, [contour_obj2_exp], -1, (0, 255, 0), 2
                )
            # Intersection – filled red
            if contour_I is not None:
                cv2.drawContours(
                    debug_img, [contour_I], -1, (0, 0, 255), -1
                )
                """
                if centroid_px is not None:
                    self._draw_crosshair(
                        debug_img, int(centroid_px[0]), int(centroid_px[1]),
                        (255, 255, 255)
                    )"""
                """
                # Draw a small white filled circle at the centroid
                radius = 3
                cv2.circle(debug_img, (int(cx), int(cy)), radius, (255, 255, 255), -1)
                """

        return annotated_img, debug_img, centroid_px, num_detections

    # ---------------------------------------------------------------------- #
    # HELPERS                                                                  #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _find_largest_contour(mask: np.ndarray):
        """Return the largest external contour of a binary mask, or None."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def _draw_crosshair(
        img: np.ndarray, cx: int, cy: int, color: tuple, size: int = 10
    ) -> None:
        """Draw a small crosshair at (cx, cy) on img."""
        cv2.line(img, (cx - size, cy), (cx + size, cy), color, 2)
        cv2.line(img, (cx, cy - size), (cx, cy + size), color, 2)


# --------------------------------------------------------------------------- #
# ENTRY POINT                                                                  #
# --------------------------------------------------------------------------- #

def main(args=None):
    """Main entry point for the Yolo_segment ROS 2 node."""
    rclpy.init(args=args)
    node = YoloSegmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Shutting down Yolo_segment. '
            f'Processed {node._frame_count} frames, '
            f'found intersections in {node._detection_count} frames '
            f'({100.0 * node._detection_count / max(node._frame_count, 1):.1f}%).'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
