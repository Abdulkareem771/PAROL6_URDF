#!/usr/bin/env python3
"""
manual_line Node — Manual Stroke Weld Seam Node
================================================

A ROS 2 node that overlays manually drawn strokes (red lines) on every incoming
image frame, making it consistent with ``color_mode`` and ``yolo_segment``.

KEY DESIGN GOAL
---------------
Once the operator draws the seam line for a given fixture / part position, the
node saves the strokes to disk (``~/.parol6/manual_line_config.json``) and
replays them automatically on every subsequent startup.  No re-drawing needed
for repeat jobs at the same position.

WORKFLOW
--------
1. Operator draws lines in the GUI's Manual Red Line panel.
2. GUI serialises stroke data as JSON and calls ``~/set_strokes`` service.
3. Node paints the strokes on every new frame; saves config to disk.
4. On **next startup** the node auto-loads the saved config and starts
   painting immediately — no action required from the operator.
5. To start fresh: GUI calls ``~/reset_strokes`` → saved config is cleared.

PUBLISHED TOPICS
----------------
  /vision/processing_mode/annotated_image  (sensor_msgs/Image, bgr8)
  /vision/processing_mode/debug_image      (sensor_msgs/Image, bgr8)
  /vision/processing_mode/seam_centroid    (geometry_msgs/PointStamped)

SUBSCRIBED TOPICS
-----------------
  /vision/captured_image_color  (sensor_msgs/Image)

SERVICES
--------
  ~/set_strokes    (std_srvs/Trigger) — GUI sends JSON in ``strokes_json`` param
  ~/reset_strokes  (std_srvs/Trigger) — clear all strokes + delete saved config

PARAMETERS
----------
  image_topic   (str)    — input topic (default /vision/captured_image_color)
  stroke_color  (int[3]) — BGR paint colour (default [0, 0, 255] = red in BGR)
  stroke_width  (int)    — line thickness in pixels (default 5)
  strokes_json  (str)    — JSON-encoded stroke list (updated by set_strokes svc)
  publish_debug (bool)   — publish debug overlay (default True)
"""

import json
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np

# ---------------------------------------------------------------------------
_DEFAULT_IMAGE_TOPIC = '/vision/captured_image_color'
_DEFAULT_CONFIG_PATH = os.path.expanduser('~/.parol6/manual_line_config.json')


class ManualLineNode(Node):
    """
    Overlays user-defined strokes on every captured frame and publishes the
    result to the standard processing_mode topics.
    """

    def __init__(self):
        super().__init__('manual_line')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('image_topic',   _DEFAULT_IMAGE_TOPIC)
        self.declare_parameter('stroke_color',  [0, 0, 255])   # BGR red
        self.declare_parameter('stroke_width',  5)
        self.declare_parameter('strokes_json',  '')            # set by GUI
        self.declare_parameter('publish_debug', True)

        self._in_topic     = self.get_parameter('image_topic').value
        self._stroke_color = list(self.get_parameter('stroke_color').value)
        self._stroke_width = int(self.get_parameter('stroke_width').value)
        self._publish_debug = self.get_parameter('publish_debug').value

        # Loaded stroke list: [  [ [x1,y1], [x2,y2], ... ], ... ]
        # Each inner list is a polyline (one brush stroke).
        self._strokes: list[list[list[int]]] = []

        # ── CV bridge ────────────────────────────────────────────────────
        self._bridge = CvBridge()

        # ── Publishers ───────────────────────────────────────────────────
        self._ann_pub = self.create_publisher(
            Image, '/vision/processing_mode/annotated_image', 10
        )
        self._centroid_pub = self.create_publisher(
            PointStamped, '/vision/processing_mode/seam_centroid', 10
        )
        if self._publish_debug:
            self._debug_pub = self.create_publisher(
                Image, '/vision/processing_mode/debug_image', 10
            )

        # ── Subscriber ───────────────────────────────────────────────────
        self._sub = self.create_subscription(
            Image, self._in_topic, self._image_callback, 10
        )

        # ── Services ─────────────────────────────────────────────────────
        self.create_service(Trigger, '~/set_strokes',   self._svc_set_strokes)
        self.create_service(Trigger, '~/reset_strokes', self._svc_reset_strokes)

        # ── Load saved config on startup ─────────────────────────────────
        self._config_path = _DEFAULT_CONFIG_PATH
        self._load_config()

        self.get_logger().info(
            f'manual_line node initialized.\n'
            f'  Subscribed to : {self._in_topic}\n'
            f'  Stroke color  : {self._stroke_color} (BGR)\n'
            f'  Stroke width  : {self._stroke_width} px\n'
            f'  Saved strokes : {len(self._strokes)}\n'
            f'  Config path   : {self._config_path}'
        )

    # ── Config persistence ───────────────────────────────────────────────────

    def _load_config(self) -> None:
        """Load stroke config from disk if it exists."""
        if not os.path.isfile(self._config_path):
            self.get_logger().info('[manual_line] No saved config found — starting fresh.')
            return
        try:
            with open(self._config_path, 'r') as f:
                cfg = json.load(f)
            self._stroke_color = cfg.get('color', self._stroke_color)
            self._stroke_width = int(cfg.get('width', self._stroke_width))
            self._strokes      = cfg.get('strokes', [])
            self.get_logger().info(
                f'[manual_line] Loaded {len(self._strokes)} stroke(s) from {self._config_path}.'
            )
        except Exception as exc:
            self.get_logger().error(f'[manual_line] Failed to load config: {exc}')

    def _save_config(self) -> None:
        """Save current strokes + style to disk."""
        try:
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with open(self._config_path, 'w') as f:
                json.dump({
                    'color':   self._stroke_color,
                    'width':   self._stroke_width,
                    'strokes': self._strokes,
                }, f, indent=2)
            self.get_logger().info(
                f'[manual_line] Config saved: {len(self._strokes)} stroke(s).'
            )
        except Exception as exc:
            self.get_logger().error(f'[manual_line] Failed to save config: {exc}')

    # ── Services ─────────────────────────────────────────────────────────────

    def _svc_set_strokes(self, _req, response):
        """
        Reload strokes from the ``strokes_json`` ROS parameter.
        The GUI should call:
            ros2 param set /manual_line strokes_json '<json>'
            ros2 service call /manual_line/set_strokes std_srvs/srv/Trigger {}
        """
        try:
            raw = self.get_parameter('strokes_json').value
            if not raw:
                response.success = False
                response.message = 'strokes_json parameter is empty.'
                return response

            data = json.loads(raw)
            # Accept both a plain list-of-strokes or a full config dict
            if isinstance(data, list):
                self._strokes = data
            elif isinstance(data, dict):
                self._strokes = data.get('strokes', [])
                self._stroke_color = data.get('color', self._stroke_color)
                self._stroke_width = int(data.get('width', self._stroke_width))

            self._save_config()
            response.success = True
            response.message = f'Loaded {len(self._strokes)} stroke(s).'
        except Exception as exc:
            response.success = False
            response.message = f'Error: {exc}'
        return response

    def _svc_reset_strokes(self, _req, response):
        """Clear all strokes from memory and delete saved config."""
        self._strokes = []
        try:
            if os.path.isfile(self._config_path):
                os.remove(self._config_path)
                self.get_logger().info('[manual_line] Saved config deleted.')
        except Exception as exc:
            self.get_logger().warn(f'[manual_line] Could not delete config: {exc}')
        response.success = True
        response.message = 'All strokes cleared.'
        return response

    # ── Image callback ───────────────────────────────────────────────────────

    def _image_callback(self, msg: Image) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'[manual_line] CvBridge failed: {exc}')
            return

        annotated, debug, centroid_px = self._apply_strokes(img)

        # Publish annotated image (always)
        ann_msg = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        ann_msg.header = msg.header
        self._ann_pub.publish(ann_msg)

        # Publish debug image
        if self._publish_debug and debug is not None:
            dbg_msg = self._bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            dbg_msg.header = msg.header
            self._debug_pub.publish(dbg_msg)

        # Publish centroid
        if centroid_px is not None:
            cx, cy = centroid_px
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x = float(cx)
            pt.point.y = float(cy)
            pt.point.z = 0.0
            self._centroid_pub.publish(pt)

    # ── Core rendering ───────────────────────────────────────────────────────

    def _apply_strokes(self, img: np.ndarray):
        """
        Paint stored strokes onto the image.
        Returns:
            annotated – original + strokes
            debug     – same + bounding-box and centroid overlay
            centroid  – (cx, cy) tuple, or None if no strokes
        """
        annotated = img.copy()
        debug = img.copy() if self._publish_debug else None
        bgr_color = tuple(int(c) for c in self._stroke_color)  # ensure int tuple

        if not self._strokes:
            # No strokes: pass-through with a badge
            if debug is not None:
                cv2.rectangle(debug, (0, 0), (280, 26), (30, 30, 30), -1)
                cv2.putText(debug, 'Manual Line — no strokes saved',
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 180, 255), 1, cv2.LINE_AA)
            return annotated, debug, None

        # Stroke mask for centroid computation
        stroke_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for stroke_pts in self._strokes:
            if len(stroke_pts) < 2:
                continue
            pts = np.array(stroke_pts, dtype=np.int32)
            cv2.polylines(annotated, [pts], isClosed=False,
                          color=bgr_color, thickness=self._stroke_width,
                          lineType=cv2.LINE_AA)
            cv2.polylines(stroke_mask, [pts], isClosed=False,
                          color=255, thickness=self._stroke_width)

        # Compute centroid of all painted pixels
        ys, xs = np.where(stroke_mask > 0)
        centroid_px = None
        if len(xs) > 0:
            cx = float(xs.mean())
            cy = float(ys.mean())
            centroid_px = (cx, cy)

        # Debug image — same strokes + centroid crosshair
        if debug is not None:
            for stroke_pts in self._strokes:
                if len(stroke_pts) < 2:
                    continue
                pts = np.array(stroke_pts, dtype=np.int32)
                cv2.polylines(debug, [pts], isClosed=False,
                              color=bgr_color, thickness=self._stroke_width,
                              lineType=cv2.LINE_AA)
            if centroid_px is not None:
                icx, icy = int(centroid_px[0]), int(centroid_px[1])
                cv2.drawMarker(debug, (icx, icy), (255, 255, 255),
                               cv2.MARKER_CROSS, 20, 2)
            badge = f'{len(self._strokes)} stroke(s) loaded | centroid {centroid_px}'
            cv2.rectangle(debug, (0, 0), (480, 26), (30, 30, 30), -1)
            cv2.putText(debug, badge, (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 140), 1, cv2.LINE_AA)

        return annotated, debug, centroid_px


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ManualLineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
