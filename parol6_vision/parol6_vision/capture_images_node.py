#!/usr/bin/env python3
"""
Capture_images Node — PAROL6 Vision Pipeline (Stage 1)

Captures matched colour + depth image pairs from the Kinect v2 and saves them
to ``parol6_vision/data/images_captured/`` (configurable via ``save_dir``).

=============================================================================
CAPTURE MODES
=============================================================================

keyboard  (default)
    A background thread reads stdin.  Press ``s`` followed by Enter to save
    one colour/depth pair immediately.  Any other key is ignored.

timed
    Saves one pair automatically every ``frame_time`` seconds (default 60 s).

=============================================================================
SAVED FILES
=============================================================================

Each capture produces two PNG files with the same timestamp token::

    color_<YYYYMMDD_HHMMSS_ffffff>.png   — 8-bit BGR
    depth_<YYYYMMDD_HHMMSS_ffffff>.png   — 16-bit unsigned (millimetres)

=============================================================================
TOPICS
=============================================================================

Subscribed
    /kinect2/qhd/image_color_rect  (sensor_msgs/Image)  — rectified colour
    /kinect2/qhd/image_depth_rect  (sensor_msgs/Image)  — aligned depth

=============================================================================
PARAMETERS
=============================================================================

    save_dir        string   parol6_vision/data/images_captured
    capture_mode    string   keyboard   (keyboard | timed)
    frame_time      float    60.0       seconds between auto-saves (timed mode)
    image_encoding  string   bgr8       cv_bridge encoding for colour
"""

import os
import sys
import threading
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters


class CaptureImagesNode(Node):
    """
    Capture_images — saves matched Kinect2 colour + depth PNG pairs to disk.

    Subscribed Topics
    -----------------
    /kinect2/qhd/image_color_rect : sensor_msgs/Image
    /kinect2/qhd/image_depth_rect : sensor_msgs/Image
    """

    def __init__(self):
        super().__init__('capture_images')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('save_dir', 'parol6_vision/data/images_captured')
        self.declare_parameter('capture_mode', 'keyboard')   # keyboard | timed
        self.declare_parameter('frame_time', 60.0)
        self.declare_parameter('image_encoding', 'bgr8')

        raw_dir = self.get_parameter('save_dir').value
        self._save_dir = os.path.expanduser(raw_dir)
        self._capture_mode = self.get_parameter('capture_mode').value
        self._frame_time = self.get_parameter('frame_time').value
        self._encoding = self.get_parameter('image_encoding').value

        # Validate mode
        if self._capture_mode not in ('keyboard', 'timed'):
            self.get_logger().error(
                f"Unknown capture_mode '{self._capture_mode}'. "
                "Use 'keyboard' or 'timed'. Defaulting to 'keyboard'."
            )
            self._capture_mode = 'keyboard'

        # ── Create output directory ───────────────────────────────────
        os.makedirs(self._save_dir, exist_ok=True)
        self.get_logger().info(f'Images will be saved to: {self._save_dir}')

        # ── CV Bridge ────────────────────────────────────────────────
        self._bridge = CvBridge()

        # ── Latest received frames (thread-safe via lock) ─────────────
        self._lock = threading.Lock()
        self._latest_color: Image | None = None
        self._latest_depth: Image | None = None

        # ── Synchronized subscribers ─────────────────────────────────
        self._color_sub = message_filters.Subscriber(
            self, Image, '/kinect2/qhd/image_color_rect'
        )
        self._depth_sub = message_filters.Subscriber(
            self, Image, '/kinect2/qhd/image_depth_rect'
        )

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub],
            queue_size=10,
            slop=0.1
        )
        self._sync.registerCallback(self._sync_callback)

        # ── Trigger flag (keyboard mode) ─────────────────────────────
        self._save_requested = threading.Event()

        # ── Start the mode-specific background thread ─────────────────
        if self._capture_mode == 'keyboard':
            self._kb_thread = threading.Thread(
                target=self._keyboard_listener, daemon=True
            )
            self._kb_thread.start()
            self.get_logger().info(
                "Capture mode: KEYBOARD — press 's' + Enter to capture a pair."
            )
        else:
            self._timer = self.create_timer(self._frame_time, self._timed_trigger)
            self.get_logger().info(
                f'Capture mode: TIMED — auto-saving every {self._frame_time:.1f} s.'
            )

        self.get_logger().info('Capture_images node ready.')

    # ─────────────────────────────────────────────────────────────────
    # Synchronized callback — stores latest matched pair
    # ─────────────────────────────────────────────────────────────────

    def _sync_callback(self, color_msg: Image, depth_msg: Image):
        with self._lock:
            self._latest_color = color_msg
            self._latest_depth = depth_msg

        # In keyboard mode the save is triggered separately.
        # In timed mode the timer fires _timed_trigger which calls _do_save.
        if self._capture_mode == 'keyboard' and self._save_requested.is_set():
            self._save_requested.clear()
            self._do_save(color_msg, depth_msg)

    # ─────────────────────────────────────────────────────────────────
    # Keyboard listener thread
    # ─────────────────────────────────────────────────────────────────

    def _keyboard_listener(self):
        """Block-reads stdin; 's' sets the save flag."""
        print(
            "\n[capture_images] Keyboard mode active.\n"
            "  Press 's' + Enter to capture a frame pair.\n"
            "  Press Ctrl-C to exit.\n"
        )
        while rclpy.ok():
            try:
                key = sys.stdin.readline().strip().lower()
            except (EOFError, OSError):
                # stdin closed (e.g. launched without a terminal)
                break
            if key == 's':
                self._save_requested.set()
                self.get_logger().info(
                    'Save requested — waiting for next synchronised frame pair...'
                )

    # ─────────────────────────────────────────────────────────────────
    # Timed trigger callback
    # ─────────────────────────────────────────────────────────────────

    def _timed_trigger(self):
        """Called by the ROS timer every frame_time seconds."""
        with self._lock:
            color_msg = self._latest_color
            depth_msg = self._latest_depth

        if color_msg is None or depth_msg is None:
            self.get_logger().warn(
                'Timed trigger fired but no synchronised frame pair received yet.'
            )
            return

        self._do_save(color_msg, depth_msg)

    # ─────────────────────────────────────────────────────────────────
    # Core save routine
    # ─────────────────────────────────────────────────────────────────

    def _do_save(self, color_msg: Image, depth_msg: Image):
        """Convert and write one colour + depth PNG pair."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        color_path = os.path.join(self._save_dir, f'color_{ts}.png')
        depth_path = os.path.join(self._save_dir, f'depth_{ts}.png')

        try:
            # ── Colour image ─────────────────────────────────────────
            cv_color = self._bridge.imgmsg_to_cv2(
                color_msg, desired_encoding=self._encoding
            )
            cv2.imwrite(color_path, cv_color)

            # ── Depth image ──────────────────────────────────────────
            # Keep native 16-bit (millimetres) so downstream nodes get real depth
            cv_depth = self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough'
            )
            # cv2.imwrite supports 16-bit PNG directly
            cv2.imwrite(depth_path, cv_depth)

            self.get_logger().info(
                f'[SAVED] color → {color_path}\n'
                f'        depth → {depth_path}'
            )

        except Exception as exc:
            self.get_logger().error(f'Failed to save frame pair: {exc}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = CaptureImagesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
