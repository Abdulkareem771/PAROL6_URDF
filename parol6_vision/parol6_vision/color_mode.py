#!/usr/bin/env python3
"""
color_mode Node – HSV Colour-Based Seam Intersection Detection

This node is a ROS 2 port of ``color_mode.py``.
It subscribes to an input camera image topic, applies HSV colour thresholding
to isolate two coloured objects (green and blue workpieces), expands their
masks outward by ``expand_px`` pixels, computes the intersection of the two
expanded regions (the weld seam area), and publishes:

  * An **annotated image** showing the filled intersection contour in red (bgr8).
  * A **debug image** showing all intermediate contours overlaid on the raw frame.
  * The **centroid** of the intersection region as a PointStamped message.

================================================================================
ALGORITHM OVERVIEW
================================================================================

1. HSV CONVERSION
   - Convert incoming BGR frame to HSV colour space.

2. MASK CREATION
   - Green mask (G): pixels whose HSV values fall within [35,50,50]–[100,255,255].
   - Blue  mask (B): pixels whose HSV values fall within [100,50,50]–[140,255,255].

3. MORPHOLOGICAL CLEANING
   - Apply morphological OPEN (5×5 kernel) to each mask to suppress noise.

4. MASK DILATION (seam expansion)
   - Dilate each cleaned mask outward by ``expand_px`` pixels using an
     elliptical structuring element.

5. INTERSECTION
   - Bitwise-AND the two dilated masks → intersection region.
   - Find the largest external contour of the intersection.

6. PUBLISH
   - Annotated image : clean copy of raw frame + filled intersection (red).
   - Debug image     : raw frame + original contours (blue) + expanded contours
                       (green) + filled intersection (red).
   - Seam centroid   : geometry_msgs/PointStamped with pixel-space (x, y, z=0).

================================================================================
SUBSCRIBED TOPICS
================================================================================
  /vision/captured_image_color  (sensor_msgs/Image)  – input camera frame

================================================================================
PUBLISHED TOPICS
================================================================================
  /color_mode/annotated_image  (sensor_msgs/Image, bgr8)   – intersection only
  /color_mode/debug_image      (sensor_msgs/Image, bgr8)   – full debug overlay
  /color_mode/seam_centroid    (geometry_msgs/PointStamped) – seam centroid (px)

================================================================================
PARAMETERS
================================================================================
  image_topic   (string)  – Input image topic to subscribe to.
  expand_px     (int)     – Pixels to dilate each colour mask outward (default 2).
  publish_debug (bool)    – Whether to publish the debug overlay image (default True).

================================================================================
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default topic
# ---------------------------------------------------------------------------
_DEFAULT_IMAGE_TOPIC = '/vision/captured_image_color'


class ColorModeNode(Node):
    """
    ROS 2 node that detects the seam between two coloured workpieces using
    HSV colour thresholding and morphological mask operations.

    Subscribed Topics:
        <image_topic>  (sensor_msgs/Image): Input colour image from camera.

    Published Topics:
        /color_mode/annotated_image  (sensor_msgs/Image): Intersection-only overlay (bgr8).
        /color_mode/debug_image      (sensor_msgs/Image): Full debug overlay (bgr8).
        /color_mode/seam_centroid    (geometry_msgs/PointStamped): Seam centroid in pixels.

    Parameters:
        image_topic   (str):  Camera topic to subscribe to.
        expand_px     (int):  Dilation radius for mask expansion (default 2).
        publish_debug (bool): Publish the debug overlay image (default True).
    """

    # ---------------------------------------------------------------------- #
    # HSV colour ranges (OpenCV convention: H 0-180, S 0-255, V 0-255)        #
    # ---------------------------------------------------------------------- #
    _LOWER_GREEN = np.array([35,  50,  50])
    _UPPER_GREEN = np.array([100, 255, 255])

    _LOWER_BLUE  = np.array([100,  50,  50])
    _UPPER_BLUE  = np.array([140, 255, 255])

    def __init__(self):
        super().__init__('color_mode')

        # ------------------------------------------------------------------ #
        # DECLARE PARAMETERS                                                   #
        # ------------------------------------------------------------------ #
        self.declare_parameter('image_topic',   _DEFAULT_IMAGE_TOPIC)
        self.declare_parameter('expand_px',     2)
        self.declare_parameter('publish_debug', True)

        # ------------------------------------------------------------------ #
        # READ PARAMETERS                                                      #
        # ------------------------------------------------------------------ #
        self.image_topic   = self.get_parameter('image_topic').value
        self.expand_px     = int(self.get_parameter('expand_px').value)
        self.publish_debug = self.get_parameter('publish_debug').value

        # ------------------------------------------------------------------ #
        # CV BRIDGE                                                            #
        # ------------------------------------------------------------------ #
        self.bridge = CvBridge()

        # Pre-build structuring elements (reused every frame)
        self._morph_kernel = np.ones((5, 5), np.uint8)
        self._dil_kernel   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * self.expand_px + 1, 2 * self.expand_px + 1),
        )

        # ------------------------------------------------------------------ #
        # PUBLISHERS                                                           #
        # ------------------------------------------------------------------ #
        self.annotated_pub = self.create_publisher(
            Image, '/color_mode/annotated_image', 10
        )
        self.seam_centroid_pub = self.create_publisher(
            PointStamped, '/color_mode/seam_centroid', 10
        )
        if self.publish_debug:
            self.debug_pub = self.create_publisher(
                Image, '/color_mode/debug_image', 10
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
        self._frame_count     = 0
        self._detection_count = 0

        self.get_logger().info(
            f'color_mode node initialized.\n'
            f'  Subscribed to : {self.image_topic}\n'
            f'  Expand px     : {self.expand_px}\n'
            f'  Debug images  : {self.publish_debug}'
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

        # Run the colour-segmentation pipeline
        annotated_img, debug_img, centroid_px = self._run_pipeline(img)

        # Publish annotated image (always)
        ann_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        ann_msg.header = msg.header
        self.annotated_pub.publish(ann_msg)

        # Publish debug image (optional)
        if self.publish_debug and debug_img is not None:
            dbg_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            dbg_msg.header = msg.header
            self.debug_pub.publish(dbg_msg)

        # Publish seam centroid
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
                f'Frame {self._frame_count}: no green/blue intersection found.'
            )

    # ---------------------------------------------------------------------- #
    # PIPELINE                                                                 #
    # ---------------------------------------------------------------------- #

    def _run_pipeline(self, img: np.ndarray):
        """
        Execute the full HSV-based colour segmentation and intersection pipeline.

        Args:
            img: OpenCV BGR image (H×W×3, uint8).

        Returns:
            annotated_img  – BGR image with filled intersection contour only (red).
            debug_img      – BGR image with all intermediate overlays, or None
                             if publish_debug is False.
            centroid_px    – (cx, cy) tuple in pixel coords, or None if not found.
        """
        annotated_img = img.copy()
        debug_img     = img.copy() if self.publish_debug else None

        # 1 ─ Convert to HSV -------------------------------------------------
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 2 ─ Create colour masks --------------------------------------------
        G = cv2.inRange(hsv, self._LOWER_GREEN, self._UPPER_GREEN)
        B = cv2.inRange(hsv, self._LOWER_BLUE,  self._UPPER_BLUE)

        # 3 ─ Morphological OPEN (noise removal) -----------------------------
        G = cv2.morphologyEx(G, cv2.MORPH_OPEN, self._morph_kernel)
        B = cv2.morphologyEx(B, cv2.MORPH_OPEN, self._morph_kernel)

        # 4 ─ Find original contours (for debug overlay) ---------------------
        contour_G = self._find_largest_contour(G)
        contour_B = self._find_largest_contour(B)

        # 5 ─ Dilation (seam expansion) --------------------------------------
        G_exp = cv2.dilate(G, self._dil_kernel)
        B_exp = cv2.dilate(B, self._dil_kernel)

        contour_G_exp = self._find_largest_contour(G_exp)
        contour_B_exp = self._find_largest_contour(B_exp)

        # 6 ─ Intersection ---------------------------------------------------
        intersection_mask = cv2.bitwise_and(G_exp, B_exp)
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
                # White crosshair on annotated image
                #self._draw_crosshair(annotated_img, int(cx), int(cy), (255, 255, 255))

        # 8 ─ Draw on debug_img (all layers) ---------------------------------
        if self.publish_debug and debug_img is not None:
            # Original colour-object contours – blue
            if contour_G is not None:
                cv2.drawContours(debug_img, [contour_G], -1, (255, 0, 0), 2)
            if contour_B is not None:
                cv2.drawContours(debug_img, [contour_B], -1, (255, 0, 0), 2)
            # Expanded contours – green
            if contour_G_exp is not None:
                cv2.drawContours(debug_img, [contour_G_exp], -1, (0, 255, 0), 2)
            if contour_B_exp is not None:
                cv2.drawContours(debug_img, [contour_B_exp], -1, (0, 255, 0), 2)
            # Intersection – filled red
            if contour_I is not None:
                cv2.drawContours(debug_img, [contour_I], -1, (0, 0, 255), -1)
                """
                if centroid_px is not None:
                    self._draw_crosshair(
                        debug_img, int(centroid_px[0]), int(centroid_px[1]),
                        (255, 255, 255)
                    )"""

        return annotated_img, debug_img, centroid_px

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
    """Main entry point for the color_mode ROS 2 node."""
    rclpy.init(args=args)
    node = ColorModeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Shutting down color_mode. '
            f'Processed {node._frame_count} frames, '
            f'found intersections in {node._detection_count} frames '
            f'({100.0 * node._detection_count / max(node._frame_count, 1):.1f}%).'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
