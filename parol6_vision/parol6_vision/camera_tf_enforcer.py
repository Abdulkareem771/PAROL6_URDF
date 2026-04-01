#!/usr/bin/env python3
"""
camera_tf_enforcer.py — PAROL6 Vision
======================================
Reads the calibrated camera extrinsics from ~/.parol6/camera_tf.yaml
and publishes them at 100 Hz as a dynamic TF on /tf.

Because a dynamic TF (with a recent timestamp) takes precedence over a
static TF from /tf_static, this node effectively OVERRIDES the hardcoded
camera frame published by live_pipeline.launch.py — without any restart.

Workflow:
  1. Run ArUco calibration → calibrator saves ~/.parol6/camera_tf.yaml
  2. Start this node (from GUI or CLI)   → new camera frame is enforced live
  3. Stop this node (GUI or Ctrl+C)      → pipeline reverts to launch-file TF

Entry point (after colcon build):
  ros2 run parol6_vision camera_tf_enforcer
"""

from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros

import yaml
from scipy.spatial.transform import Rotation as R
import numpy as np

CAMERA_TF_PATH = Path.home() / '.parol6' / 'camera_tf.yaml'

# Defaults match current live_pipeline.launch.py (euler, radians)
_DEFAULTS = {
    'frame_id':       'base_link',
    'child_frame_id': 'kinect2',
    'x': 0.646, 'y': 0.1225, 'z': 1.015,
    'roll': -3.14159, 'pitch': 0.0, 'yaw': 1.603684,
}


def _load_camera_tf_yaml(path: Path) -> dict:
    """Load camera TF config from yaml, returning merged dict with defaults."""
    cfg = dict(_DEFAULTS)
    try:
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in loaded.items() if v is not None})
    except FileNotFoundError:
        pass  # Use defaults — file will be created on first calibration
    except Exception as e:
        pass  # Malformed yaml — fall back to defaults
    return cfg


def _cfg_to_transform(cfg: dict) -> TransformStamped:
    """Build a TransformStamped from the config dict.
    Supports both quaternion (qx/qy/qz/qw) and euler (roll/pitch/yaw) formats.
    """
    t = TransformStamped()
    t.header.frame_id = str(cfg['frame_id'])
    t.child_frame_id  = str(cfg['child_frame_id'])
    t.transform.translation.x = float(cfg.get('x', 0.0))
    t.transform.translation.y = float(cfg.get('y', 0.0))
    t.transform.translation.z = float(cfg.get('z', 0.0))

    if 'qx' in cfg:
        # Quaternion format (output of ArUco calibrator)
        t.transform.rotation.x = float(cfg['qx'])
        t.transform.rotation.y = float(cfg['qy'])
        t.transform.rotation.z = float(cfg['qz'])
        t.transform.rotation.w = float(cfg['qw'])
    else:
        # Euler format (legacy live_pipeline defaults)
        roll  = float(cfg.get('roll',  0.0))
        pitch = float(cfg.get('pitch', 0.0))
        yaw   = float(cfg.get('yaw',   0.0))
        q = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()  # [x, y, z, w]
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

    return t


class CameraTFEnforcer(Node):
    """
    ROS 2 node that enforces the calibrated camera TF at 100 Hz.

    Parameters
    ----------
    publish_rate : float
        TF publish frequency in Hz (default 100.0).
    """

    def __init__(self):
        super().__init__('camera_tf_enforcer')

        self.declare_parameter('publish_rate', 100.0)
        rate = self.get_parameter('publish_rate').value

        self._broadcaster = tf2_ros.TransformBroadcaster(self)
        self._tf_msg: TransformStamped | None = None
        self._bridge_gap_msg: TransformStamped | None = None

        self._load()

        period = 1.0 / max(rate, 1.0)
        self._timer = self.create_timer(period, self._publish)

    # ------------------------------------------------------------------
    def _load(self):
        cfg = _load_camera_tf_yaml(CAMERA_TF_PATH)
        self._tf_msg = _cfg_to_transform(cfg)

        t = self._tf_msg.transform.translation
        if CAMERA_TF_PATH.exists():
            source = str(CAMERA_TF_PATH)
        else:
            source = "built-in defaults (yaml not found)"

        self.get_logger().info(
            f"[CameraTFEnforcer] Loaded from {source}\n"
            f"  {cfg['frame_id']} → {cfg['child_frame_id']}\n"
            f"  xyz = ({t.x:.4f}, {t.y:.4f}, {t.z:.4f})"
        )

        # ── Bridge gap: kinect2 → kinect2_link (identity) ──────────────────
        # kinect2_bridge publishes its internal TF tree rooted at 'kinect2_link',
        # NOT at 'kinect2'. Without this connector, the two trees are disjoint:
        #   base_link → kinect2          (our enforcer, calibrated)
        #   kinect2_link → optical frames (bridge, floating island)
        # Publishing kinect2 → kinect2_link (identity) joins them into one tree.
        gap = TransformStamped()
        gap.header.frame_id  = 'kinect2'
        gap.child_frame_id   = 'kinect2_link'
        gap.transform.rotation.w = 1.0  # identity quaternion
        self._bridge_gap_msg = gap
        self.get_logger().info(
            "[CameraTFEnforcer] Also publishing identity bridge: kinect2 → kinect2_link"
        )

    # ------------------------------------------------------------------
    def _publish(self):
        if self._tf_msg is None:
            return
        now = self.get_clock().now().to_msg()
        self._tf_msg.header.stamp = now
        transforms = [self._tf_msg]
        if self._bridge_gap_msg is not None:
            self._bridge_gap_msg.header.stamp = now
            transforms.append(self._bridge_gap_msg)
        self._broadcaster.sendTransform(transforms)


def main(args=None):
    rclpy.init(args=args)
    node = CameraTFEnforcer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
