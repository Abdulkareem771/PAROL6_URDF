#!/usr/bin/env python3
"""
Read_image Node — PAROL6 Vision Pipeline (Stage 2)

Watches ``parol6_vision/data/images_captured/`` for **new** colour + depth
PNG pairs written by the ``capture_images`` node and publishes each pair
exactly once.

=============================================================================
BEHAVIOUR
=============================================================================

* On startup the node scans the folder and records every file already present.
  Those files are **not** published — only files that appear *after* the node
  starts are published.
* When both ``color_<ts>.png`` and ``depth_<ts>.png`` for the same timestamp
  token are detected, the pair is published together.
* Colour images are published on ``/vision/captured_image_color``.
* Depth images are published on ``/vision/captured_image_depth``.
* Each message carries a fresh ROS timestamp and
  ``frame_id = kinect2_rgb_optical_frame`` so that downstream nodes
  (``red_line_detector``, ``depth_matcher``) receive headers consistent with
  live Kinect2 data.

=============================================================================
INTEGRATION
=============================================================================

Use the ``capture_and_replay.launch.py`` launch file which remaps:

    red_line_detector:  /kinect2/sd/image_color_rect → /vision/captured_image_color
    depth_matcher:      /kinect2/sd/image_depth_rect → /vision/captured_image_depth

=============================================================================
PARAMETERS
=============================================================================

    save_dir    string   parol6_vision/data/images_captured
    poll_rate   float    1.0   Hz — how often to check for new files
    frame_id    string   kinect2_rgb_optical_frame
"""

import os
import re

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2


# Matches: color_YYYYMMDD_HHMMSS_ffffff.png  →  group(1) = timestamp token
_COLOR_RE = re.compile(r'^color_(.+)\.png$')
_DEPTH_RE = re.compile(r'^depth_(.+)\.png$')


class ReadImageNode(Node):
    """
    Read_image — publishes new colour + depth pairs found in images_captured/.

    Subscribed Topics
    -----------------
    /kinect2/sd/camera_info : sensor_msgs/CameraInfo   (cached, re-stamped)

    Published Topics
    ----------------
    /vision/captured_image_color  : sensor_msgs/Image      (bgr8)
    /vision/captured_image_depth  : sensor_msgs/Image      (16UC1)
    /vision/captured_camera_info  : sensor_msgs/CameraInfo (re-stamped)
    """

    def __init__(self):
        super().__init__('read_image')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('save_dir', 'parol6_vision/data/images_captured')
        self.declare_parameter('poll_rate', 1.0)
        self.declare_parameter('frame_id', 'kinect2_rgb_optical_frame')

        raw_dir = self.get_parameter('save_dir').value
        self._save_dir = os.path.expanduser(raw_dir)
        self._poll_rate = self.get_parameter('poll_rate').value
        self._frame_id = self.get_parameter('frame_id').value

        # ── CV Bridge ─────────────────────────────────────────────────
        self._bridge = CvBridge()

        # ── Publishers ───────────────────────────────────────────────
        self._color_pub = self.create_publisher(
            Image, '/vision/captured_image_color', 10
        )
        self._depth_pub = self.create_publisher(
            Image, '/vision/captured_image_depth', 10
        )
        self._info_pub = self.create_publisher(
            CameraInfo, '/vision/captured_camera_info', 10
        )

        # ── Camera info cache ─────────────────────────────────────────
        # We subscribe to the live /kinect2/sd/camera_info, cache the
        # latest message, and re-publish it with a fresh timestamp that
        # matches the replayed image pair — this is what lets
        # depth_matcher's ApproximateTimeSynchronizer fire reliably.
        self._latest_camera_info: CameraInfo | None = None
        self._info_sub = self.create_subscription(
            CameraInfo,
            '/kinect2/sd/camera_info',
            self._camera_info_callback,
            10
        )

        # ── Seed the "seen" set with files already on disk ────────────
        self._seen_tokens: set[str] = set()
        self._seen_tokens = self._scan_existing_tokens()
        self.get_logger().info(
            f'Watching: {self._save_dir}  '
            f'(ignoring {len(self._seen_tokens)} pre-existing token(s))'
        )

        # ── Polling timer ─────────────────────────────────────────────
        self.create_timer(1.0 / self._poll_rate, self._poll_callback)
        self.get_logger().info(
            f'Read_image node ready — polling every {1.0/self._poll_rate:.1f} s.'
        )

    # ─────────────────────────────────────────────────────────────────
    # Startup scan — record tokens already present so we skip them
    # ─────────────────────────────────────────────────────────────────

    def _scan_existing_tokens(self) -> set[str]:
        """Return all timestamp tokens already present in save_dir."""
        tokens: set[str] = set()
        if not os.path.isdir(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)
            return tokens
        for fname in os.listdir(self._save_dir):
            m = _COLOR_RE.match(fname) or _DEPTH_RE.match(fname)
            if m:
                tokens.add(m.group(1))
        return tokens

    # ─────────────────────────────────────────────────────────────────
    # Camera info cache callback
    # ─────────────────────────────────────────────────────────────────

    def _camera_info_callback(self, msg: CameraInfo):
        """Cache the latest camera_info for re-stamping at publish time."""
        self._latest_camera_info = msg

    # ─────────────────────────────────────────────────────────────────
    # Polling callback
    # ─────────────────────────────────────────────────────────────────

    def _poll_callback(self):
        """Detect new matched pairs and publish them."""
        if not os.path.isdir(self._save_dir):
            return

        color_tokens: set[str] = set()
        depth_tokens: set[str] = set()

        try:
            entries = os.listdir(self._save_dir)
        except OSError as exc:
            self.get_logger().warn(f'Cannot list {self._save_dir}: {exc}')
            return

        for fname in entries:
            mc = _COLOR_RE.match(fname)
            if mc:
                color_tokens.add(mc.group(1))
                continue
            md = _DEPTH_RE.match(fname)
            if md:
                depth_tokens.add(md.group(1))

        # Tokens that have BOTH colour and depth and are NEW
        complete_new = (color_tokens & depth_tokens) - self._seen_tokens

        for token in sorted(complete_new):
            color_path = os.path.join(self._save_dir, f'color_{token}.png')
            depth_path = os.path.join(self._save_dir, f'depth_{token}.png')
            self._publish_pair(token, color_path, depth_path)
            self._seen_tokens.add(token)

    # ─────────────────────────────────────────────────────────────────
    # Publish one matched pair
    # ─────────────────────────────────────────────────────────────────

    def _publish_pair(self, token: str, color_path: str, depth_path: str):
        """Load both PNGs and publish as ROS Image messages."""
        now = self.get_clock().now().to_msg()

        # ── Colour ───────────────────────────────────────────────────
        try:
            cv_color = cv2.imread(color_path, cv2.IMREAD_COLOR)
            if cv_color is None:
                raise ValueError('cv2.imread returned None')
            color_msg = self._bridge.cv2_to_imgmsg(cv_color, encoding='bgr8')
            color_msg.header.stamp = now
            color_msg.header.frame_id = self._frame_id
            self._color_pub.publish(color_msg)
        except Exception as exc:
            self.get_logger().error(
                f'Failed to publish colour [{token}]: {exc}'
            )
            return

        # ── Depth ─────────────────────────────────────────────────────
        try:
            # Load as 16-bit (IMREAD_UNCHANGED preserves the bit depth)
            cv_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if cv_depth is None:
                raise ValueError('cv2.imread returned None')
            if cv_depth.dtype.kind != 'u' or cv_depth.itemsize != 2:
                # Saved file was not 16-bit — attempt conversion gracefully
                cv_depth = cv_depth.astype('uint16')
            depth_msg = self._bridge.cv2_to_imgmsg(cv_depth, encoding='16UC1')
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = self._frame_id
            self._depth_pub.publish(depth_msg)
        except Exception as exc:
            self.get_logger().error(
                f'Failed to publish depth [{token}]: {exc}'
            )
            return

        # ── Camera info (re-stamped) ──────────────────────────────────
        if self._latest_camera_info is not None:
            info_msg = CameraInfo()
            info_msg.header.stamp = now
            info_msg.header.frame_id = self._frame_id
            # Copy intrinsics from cached live message
            info_msg.width    = self._latest_camera_info.width
            info_msg.height   = self._latest_camera_info.height
            info_msg.k        = self._latest_camera_info.k
            info_msg.d        = self._latest_camera_info.d
            info_msg.r        = self._latest_camera_info.r
            info_msg.p        = self._latest_camera_info.p
            info_msg.distortion_model = self._latest_camera_info.distortion_model
            self._info_pub.publish(info_msg)
        else:
            self.get_logger().warn(
                f'[{token}] No camera_info received yet — '
                'depth_matcher sync may not fire. '
                'Ensure /kinect2/sd/camera_info is publishing.'
            )

        self.get_logger().info(
            f'[PUBLISHED] token={token}\n'
            f'  color  → /vision/captured_image_color\n'
            f'  depth  → /vision/captured_image_depth\n'
            f'  info   → /vision/captured_camera_info'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ReadImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
