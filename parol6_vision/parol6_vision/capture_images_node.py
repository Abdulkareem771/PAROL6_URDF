#!/usr/bin/env python3
"""
Capture_images Node — PAROL6 Vision Pipeline (Stage 1)

Captures matched colour + depth image pairs from the Kinect v2 and publishes
them to ROS topics instead of saving to disk.

=============================================================================
CAPTURE MODES
=============================================================================

keyboard  (default)
    A background thread reads stdin.  Press ``s`` followed by Enter to publish
    one colour/depth pair immediately.  Any other key is ignored.

timed
    Publishes one pair automatically every ``frame_time`` seconds (default 10 s).

=============================================================================
TOPICS
=============================================================================

Subscribed
    /kinect2/qhd/image_color_rect  (sensor_msgs/Image)      — rectified colour
    /kinect2/qhd/image_depth_rect  (sensor_msgs/Image)      — aligned depth
    /kinect2/qhd/camera_info       (sensor_msgs/CameraInfo) — camera intrinsics

Published
    /vision/captured_image_color   (sensor_msgs/Image)      — captured colour frame
    /vision/captured_image_depth   (sensor_msgs/Image)      — captured depth frame
    /vision/captured_camera_info   (sensor_msgs/CameraInfo) — relayed camera info

=============================================================================
PARAMETERS
=============================================================================

    capture_mode    string   keyboard   (keyboard | timed)
    frame_time      float    10.0       seconds between auto-publishes (timed mode)
"""

import sys
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import message_filters


class CaptureImagesNode(Node):
    """
    Capture_images — publishes matched Kinect2 colour + depth frame pairs
    and relays CameraInfo to vision topics.

    Subscribed Topics
    -----------------
    /kinect2/qhd/image_color_rect : sensor_msgs/Image
    /kinect2/qhd/image_depth_rect : sensor_msgs/Image
    /kinect2/qhd/camera_info      : sensor_msgs/CameraInfo

    Published Topics
    ----------------
    /vision/captured_image_color  : sensor_msgs/Image
    /vision/captured_image_depth  : sensor_msgs/Image
    /vision/captured_camera_info  : sensor_msgs/CameraInfo
    """

    def __init__(self):
        super().__init__('capture_images')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('capture_mode', 'keyboard')   # keyboard | timed
        self.declare_parameter('frame_time', 10.0)

        self._capture_mode = self.get_parameter('capture_mode').value
        self._frame_time = self.get_parameter('frame_time').value

        # Validate mode
        if self._capture_mode not in ('keyboard', 'timed'):
            self.get_logger().error(
                f"Unknown capture_mode '{self._capture_mode}'. "
                "Use 'keyboard' or 'timed'. Defaulting to 'keyboard'."
            )
            self._capture_mode = 'keyboard'

        # ── Latest received frames (thread-safe via lock) ─────────────
        self._lock = threading.Lock()
        self._latest_color: Image | None = None
        self._latest_depth: Image | None = None

        # ── Publishers ───────────────────────────────────────────────
        self._pub_color = self.create_publisher(
            Image, '/vision/captured_image_color', 10
        )
        self._pub_depth = self.create_publisher(
            Image, '/vision/captured_image_depth', 10
        )
        self._pub_camera_info = self.create_publisher(
            CameraInfo, '/vision/captured_camera_info', 10
        )

        # ── Synchronized subscribers (colour + depth) ─────────────────
        self._color_sub = message_filters.Subscriber(
            self, Image, '/kinect2/qhd/image_color_rect'
        )
        self._depth_sub = message_filters.Subscriber(
            self, Image, '/kinect2/qhd/image_depth_rect'
        )

        # ── CameraInfo subscriber ─────────────────────────────────────
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            '/kinect2/qhd/camera_info',
            self._camera_info_callback,
            10
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
            self._do_publish(color_msg, depth_msg)

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

        self._do_publish(color_msg, depth_msg)

    # ─────────────────────────────────────────────────────────────────
    # Core publish routine
    # ─────────────────────────────────────────────────────────────────

    def _do_publish(self, color_msg: Image, depth_msg: Image):
        """Publish one captured colour + depth frame pair to vision topics."""
        self._pub_color.publish(color_msg)
        self._pub_depth.publish(depth_msg)
        self.get_logger().info(
            'Published captured colour + depth frame pair '
            '→ /vision/captured_image_color, /vision/captured_image_depth'
        )

    # ─────────────────────────────────────────────────────────────────
    # CameraInfo relay
    # ─────────────────────────────────────────────────────────────────

    def _camera_info_callback(self, msg: CameraInfo):
        """Relay /kinect2/qhd/camera_info to /vision/captured_camera_info."""
        self._pub_camera_info.publish(msg)


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
