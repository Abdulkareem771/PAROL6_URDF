#!/usr/bin/env python3
"""
eye_to_hand_calibrator.py — PAROL6 Vision
==========================================
Eye-to-hand calibration using ArUco markers.

PREREQUISITES
─────────────
The aruco_ros 'single' node must be running and detecting the marker.
Launch it with:

  ros2 run aruco_ros single \\
      --ros-args --remap /image:=/kinect2/sd/image_color_rect \\
                 --remap /camera_info:=/kinect2/sd/camera_info \\
      -p marker_id:=6 \\
      -p marker_size:=0.04575 \\
      -p camera_frame:=kinect2_ir_optical_frame \\
      -p marker_frame:=detected_marker_frame \\
      -p corner_refinement:=SUBPIX \\
      -p image_is_rectified:=True \\
      -p marker_dict:=DICT_ARUCO_ORIGINAL

Or use the GUI's "📷 Cam Calibrate" tab to launch everything automatically.

ALGORITHM
─────────
Eye-to-hand calibration computes base_link → camera given:

  T_base_marker  — known physical position of the ArUco cube in base_link
                   (measured with a ruler; set via ROS parameters below)

  T_cam_marker   — detected position of the marker in the camera optical
                   frame (collected from the TF topic; averaged over N samples)

  T_base_cam = T_base_marker * inv(T_cam_marker)
  → This gives us base_link → kinect2_ir_optical_frame

  T_base_link_frame = T_base_cam * inv(T_link_to_optical)
  → This gives us base_link → kinect2_link (via a known intra-camera TF)

ROS PARAMETERS
──────────────
  marker_x / marker_y / marker_z   :  known marker origin in base_link (metres)
  marker_qx/qy/qz/qw               :  marker orientation in base_link (default: identity)
  samples_to_collect                :  number of ArUco frames to average (default 20)
  camera_optical_frame              :  frame reported by aruco_ros (source of T_cam_marker)
  camera_link_frame                 :  frame we want to anchor (output frame)
  output_path                       :  yaml file to write results to

OUTPUT
──────
Saves calibrated transform to ~/.parol6/camera_tf.yaml (or output_path).
The camera_tf_enforcer node can then apply the result live without restart.
"""

from pathlib import Path
import datetime
import yaml

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
from scipy.spatial.transform import Rotation as Rot

DEFAULT_OUTPUT = str(Path.home() / '.parol6' / 'camera_tf.yaml')


def _tf_to_matrix(transform) -> np.ndarray:
    """Convert a geometry_msgs/Transform to a 4×4 numpy matrix."""
    t = transform.translation
    q = transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3,  3] = [t.x, t.y, t.z]
    return T


def _matrix_to_dict(T: np.ndarray, frame_id: str, child_frame_id: str) -> dict:
    """Convert a 4×4 matrix to a yaml-serialisable dict (quaternion format)."""
    q = Rot.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
    euler = Rot.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return {
        'frame_id':       frame_id,
        'child_frame_id': child_frame_id,
        'x':  float(T[0, 3]),
        'y':  float(T[1, 3]),
        'z':  float(T[2, 3]),
        'qx': float(q[0]),
        'qy': float(q[1]),
        'qz': float(q[2]),
        'qw': float(q[3]),
        # Euler for human readability in the yaml
        '_euler_deg': {
            'roll':  round(float(euler[0]), 3),
            'pitch': round(float(euler[1]), 3),
            'yaw':   round(float(euler[2]), 3),
        },
    }


class EyeToHandCalibrator(Node):
    def __init__(self):
        super().__init__('eye_to_hand_calibrator')

        # ── ROS Parameters ───────────────────────────────────────────────
        # Known physical position of the ArUco marker in base_link frame.
        # Measure this with a ruler / CAD reference.
        self.declare_parameter('marker_x', 0.623)
        self.declare_parameter('marker_y', 0.080)
        self.declare_parameter('marker_z', 0.234)
        # Marker orientation in base_link (default: flat on table = identity)
        self.declare_parameter('marker_qx', 0.0)
        self.declare_parameter('marker_qy', 0.0)
        self.declare_parameter('marker_qz', 0.0)
        self.declare_parameter('marker_qw', 1.0)
        # Camera TF chain params
        self.declare_parameter('camera_optical_frame', 'kinect2_ir_optical_frame')
        self.declare_parameter('camera_link_frame',    'kinect2_link')
        self.declare_parameter('base_frame',           'base_link')
        # Calibration quality
        self.declare_parameter('samples_to_collect', 20)
        # Output file
        self.declare_parameter('output_path', DEFAULT_OUTPUT)

        # ── Internals ────────────────────────────────────────────────────
        self.source_frame = self.get_parameter('camera_optical_frame').value
        self.target_frame = 'detected_marker_frame'
        self.base_frame   = self.get_parameter('base_frame').value
        self.link_frame   = self.get_parameter('camera_link_frame').value
        self.n_samples    = self.get_parameter('samples_to_collect').value
        self.output_path  = self.get_parameter('output_path').value

        self._T_cam_marker_list: list[np.ndarray] = []

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f"Eye-to-hand calibration starting.\n"
            f"  Camera optical frame  : {self.source_frame}\n"
            f"  Marker frame          : {self.target_frame}\n"
            f"  Samples needed        : {self.n_samples}\n"
            f"  Output file           : {self.output_path}\n"
            f"Make sure aruco_ros 'single' is running and the marker is visible."
        )

        self._timer = self.create_timer(0.1, self._collect_sample)

    # ── Sample collection ────────────────────────────────────────────────────

    def _collect_sample(self):
        n = len(self._T_cam_marker_list)

        if n >= self.n_samples:
            self._timer.cancel()
            self._compute_and_save()
            return

        try:
            trans = self.tf_buffer.lookup_transform(
                self.source_frame,
                self.target_frame,
                rclpy.time.Time()
            )
            self._T_cam_marker_list.append(_tf_to_matrix(trans.transform))

            if n % 5 == 0:
                self.get_logger().info(
                    f"[Calibrator] Collected {n + 1}/{self.n_samples} samples…"
                )

        except TransformException as ex:
            self.get_logger().warn(
                f"[Calibrator] Waiting for marker TF ({n}/{self.n_samples}): {ex}"
            )

    # ── Math & output ────────────────────────────────────────────────────────

    def _compute_and_save(self):
        self.get_logger().info(
            "[Calibrator] Collection complete — computing average transform…"
        )

        # ── Average T_cam_marker ─────────────────────────────────────────
        translations = np.array([T[:3, 3] for T in self._T_cam_marker_list])
        quats = np.array([
            Rot.from_matrix(T[:3, :3]).as_quat()
            for T in self._T_cam_marker_list
        ])
        avg_t   = translations.mean(axis=0)
        avg_rot = Rot.from_quat(quats).mean()
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, :3] = avg_rot.as_matrix()
        T_cam_marker[:3,  3] = avg_t

        # ── Build T_base_marker from parameters ──────────────────────────
        mx  = self.get_parameter('marker_x').value
        my  = self.get_parameter('marker_y').value
        mz  = self.get_parameter('marker_z').value
        mqx = self.get_parameter('marker_qx').value
        mqy = self.get_parameter('marker_qy').value
        mqz = self.get_parameter('marker_qz').value
        mqw = self.get_parameter('marker_qw').value

        T_base_marker = np.eye(4)
        T_base_marker[:3, :3] = Rot.from_quat([mqx, mqy, mqz, mqw]).as_matrix()
        T_base_marker[:3,  3] = [mx, my, mz]

        # ── Eye-to-hand formula ──────────────────────────────────────────
        # T_base_cam = T_base_marker * inv(T_cam_marker)
        T_base_cam = T_base_marker @ np.linalg.inv(T_cam_marker)
        # T_base_cam is base_link → kinect2_ir_optical_frame

        # ── Step 1: base_link → kinect2_link  (accurate, uses working TF) ───
        # kinect2_link → kinect2_ir_optical_frame is a static TF published by
        # kinect2_bridge. Use a generous timeout so a transient 1“2 ms race
        # does not cause the lookup to fail immediately.
        lookup_timeout = rclpy.duration.Duration(seconds=5)
        try:
            link_to_optical = self.tf_buffer.lookup_transform(
                self.link_frame,   # kinect2_link
                self.source_frame, # kinect2_ir_optical_frame
                rclpy.time.Time(), # latest available
                timeout=lookup_timeout,
            )
            T_link_optical = _tf_to_matrix(link_to_optical.transform)
            T_base_kinect2_link = T_base_cam @ np.linalg.inv(T_link_optical)
        except TransformException as ex:
            self.get_logger().error(
                f"[Calibrator] ❌ ABORT — Could not look up "
                f"{self.link_frame}→{self.source_frame} after "
                f"{lookup_timeout.nanoseconds // 1_000_000_000} s: {ex}\n"
                "  Is kinect2_bridge running?  camera_tf.yaml NOT updated."
            )
            return  # ← abort WITHOUT touching the yaml

        # ── Step 2: promote to kinect2 (parent) to avoid dual-parent conflict ──
        # kinect2_bridge publishes kinect2 → kinect2_link. If our enforcer
        # also claims base_link → kinect2_link, that frame gets two parents → toggling.
        # Instead: look up the ACTUAL kinect2→kinect2_link TF and compose
        # it correctly so we publish base_link → kinect2 (the real root).
        try:
            root_to_link = self.tf_buffer.lookup_transform(
                'kinect2',         # root frame kinect2_bridge publishes from
                self.link_frame,   # kinect2_link
                rclpy.time.Time(),
                timeout=lookup_timeout,
            )
            T_root_to_link = _tf_to_matrix(root_to_link.transform)
            # T_base_kinect2 = T_base_kinect2_link @ inv(T_kinect2_to_kinect2_link)
            T_base_root = T_base_kinect2_link @ np.linalg.inv(T_root_to_link)
            output_child = 'kinect2'
            self.get_logger().info(
                "[Calibrator] Output frame: base_link → kinect2  "
                "(correctly composed via live kinect2→kinect2_link TF)"
            )
        except TransformException as ex:
            # kinect2 root frame not found — bridge may not publish it.
            # Save as kinect2_link (accurate math) with a warning.
            T_base_root = T_base_kinect2_link
            output_child = self.link_frame   # kinect2_link
            self.get_logger().warn(
                f"[Calibrator] Could not look up kinect2→kinect2_link: {ex}\n"
                f"  Saving as base_link → {output_child}.\n"
                "  ⚠️  If kinect2_bridge is running with publish_tf=true, "
                "you may see TF toggling. Set publish_tf: false in "
                "kinect2_bridge_gpu.yaml to resolve."
            )

        # ── Format result ────────────────────────────────────────────────
        result = _matrix_to_dict(T_base_root, self.base_frame, output_child)
        result['calibrated_at'] = datetime.datetime.now().isoformat(timespec='seconds')
        result['calibrated_by'] = 'eye_to_hand_calibrator (ArUco)'
        result['marker_position_base_link'] = {
            'x': float(mx), 'y': float(my), 'z': float(mz)
        }

        # ── Log human-readable result ─────────────────────────────────────
        euler = result['_euler_deg']
        # Camera → Marker (what ArUco detected)
        cm_t = avg_t
        cm_q = Rot.from_matrix(T_cam_marker[:3, :3]).as_quat()
        cm_e = Rot.from_matrix(T_cam_marker[:3, :3]).as_euler('xyz', degrees=True)
        self.get_logger().info(
            f"\n{'='*55}\n"
            f"  CAM → MARKER (raw ArUco detection)\n"
            f"{'='*55}\n"
            f"  X  = {cm_t[0]:.4f} m   Y  = {cm_t[1]:.4f} m   Z  = {cm_t[2]:.4f} m\n"
            f"  Qx = {cm_q[0]:.4f}  Qy = {cm_q[1]:.4f}  Qz = {cm_q[2]:.4f}  Qw = {cm_q[3]:.4f}\n"
            f"  (Euler: Roll={cm_e[0]:.2f}°  Pitch={cm_e[1]:.2f}°  Yaw={cm_e[2]:.2f}°)\n"
            f"\n  CALIBRATION RESULT: {result['frame_id']} → {result['child_frame_id']}\n"
            f"{'='*55}\n"
            f"  X  = {result['x']:.4f} m\n"
            f"  Y  = {result['y']:.4f} m\n"
            f"  Z  = {result['z']:.4f} m\n"
            f"  Qx = {result['qx']:.4f}  Qy = {result['qy']:.4f}  Qz = {result['qz']:.4f}  Qw = {result['qw']:.4f}\n"
            f"  (Euler: Roll={euler['roll']}°  Pitch={euler['pitch']}°  Yaw={euler['yaw']}°)\n"
            f"{'='*55}"
        )

        # ── Machine-parseable lines (parsed by vision GUI) ────────────────
        # Format: [CAL_<TAG>] x=V y=V z=V qx=V qy=V qz=V qw=V roll=V pitch=V yaw=V
        self.get_logger().info(
            f"[CAL_CAM_MARKER] "
            f"x={cm_t[0]:.4f} y={cm_t[1]:.4f} z={cm_t[2]:.4f} "
            f"qx={cm_q[0]:.4f} qy={cm_q[1]:.4f} qz={cm_q[2]:.4f} qw={cm_q[3]:.4f} "
            f"roll={cm_e[0]:.2f} pitch={cm_e[1]:.2f} yaw={cm_e[2]:.2f}"
        )
        self.get_logger().info(
            f"[CAL_BASE_CAM] "
            f"x={result['x']:.4f} y={result['y']:.4f} z={result['z']:.4f} "
            f"qx={result['qx']:.4f} qy={result['qy']:.4f} qz={result['qz']:.4f} qw={result['qw']:.4f} "
            f"roll={euler['roll']} pitch={euler['pitch']} yaw={euler['yaw']} "
            f"child={result['child_frame_id']}"
        )

        # ── Save to yaml ──────────────────────────────────────────────────
        out_path = Path(self.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            yaml.dump(result, f, default_flow_style=False, sort_keys=False)

        self.get_logger().info(
            f"[Calibrator] ✅ Saved to {out_path}\n"
            f"  Now start camera_tf_enforcer to apply without pipeline restart:\n"
            f"  ros2 run parol6_vision camera_tf_enforcer"
        )


def main(args=None):
    rclpy.init(args=args)
    node = EyeToHandCalibrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()