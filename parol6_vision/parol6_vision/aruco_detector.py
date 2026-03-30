#!/usr/bin/env python3
"""
aruco_detector.py — PAROL6 Vision
===================================
Standalone ArUco marker detector that publishes the detected marker as a TF frame.
Uses OpenCV's built-in aruco module — NO external aruco_ros package required.

Publishes:  /tf  (kinect2_ir_optical_frame → detected_marker_frame)
            /aruco/detected  (std_msgs/Bool)

ROS Parameters
──────────────
  marker_id           : int   ArUco marker ID to track              (default 6)
  marker_size         : float Physical side length of marker in m   (default 0.04575)
  marker_dict         : str   OpenCV ArUco dict name                (default DICT_ARUCO_ORIGINAL)
  camera_optical_frame: str   TF frame for the camera               (default kinect2_ir_optical_frame)
  marker_frame        : str   TF frame to publish for the marker    (default detected_marker_frame)
  image_topic         : str   Color/IR image to detect from         (default /kinect2/sd/image_color_rect)
  camera_info_topic   : str   CameraInfo for intrinsics             (default /kinect2/sd/camera_info)
  debug               : bool  Publish annotated image to /aruco/debug_image (default False)
"""

import cv2
import numpy as np
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge

# ── ArUco dictionary map ───────────────────────────────────────────────────────
_DICT_MAP = {
    'DICT_ARUCO_ORIGINAL':  cv2.aruco.DICT_ARUCO_ORIGINAL,
    'DICT_4X4_50':          cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100':         cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250':         cv2.aruco.DICT_4X4_250,
    'DICT_5X5_50':          cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100':         cv2.aruco.DICT_5X5_100,
    'DICT_6X6_50':          cv2.aruco.DICT_6X6_50,
    'DICT_6X6_100':         cv2.aruco.DICT_6X6_100,
    'DICT_7X7_50':          cv2.aruco.DICT_7X7_50,
}


def _rot_to_quat(rvec: np.ndarray) -> np.ndarray:
    """Convert OpenCV rotation vector to quaternion [qx, qy, qz, qw]."""
    R, _ = cv2.Rodrigues(rvec)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('marker_id',            6)
        self.declare_parameter('marker_size',          0.04575)
        self.declare_parameter('marker_dict',          'DICT_ARUCO_ORIGINAL')
        self.declare_parameter('camera_optical_frame', 'kinect2_ir_optical_frame')
        self.declare_parameter('marker_frame',         'detected_marker_frame')
        self.declare_parameter('image_topic',          '/kinect2/sd/image_color_rect')
        self.declare_parameter('camera_info_topic',    '/kinect2/sd/camera_info')
        self.declare_parameter('debug',                False)

        self._marker_id     = self.get_parameter('marker_id').value
        self._marker_size   = self.get_parameter('marker_size').value
        self._cam_frame     = self.get_parameter('camera_optical_frame').value
        self._marker_frame  = self.get_parameter('marker_frame').value
        self._debug         = self.get_parameter('debug').value

        dict_name = self.get_parameter('marker_dict').value
        dict_id   = _DICT_MAP.get(dict_name, cv2.aruco.DICT_ARUCO_ORIGINAL)

        # ── ArUco detector (supports both old and new OpenCV API) ──────────────
        self._aruco_dict   = cv2.aruco.getPredefinedDictionary(dict_id)
        self._aruco_params = cv2.aruco.DetectorParameters()
        try:
            # OpenCV ≥ 4.7 — new API
            self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
            self._use_new_api = True
        except AttributeError:
            # OpenCV < 4.7 — old API
            self._detector    = None
            self._use_new_api = False
            self.get_logger().warn("Using legacy cv2.aruco API (OpenCV < 4.7)")

        # ── Camera intrinsics (filled on first CameraInfo msg) ────────────────
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self._bridge   = CvBridge()
        self._tf_bcast = TransformBroadcaster(self)

        qos_sensor = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._img_sub  = self.create_subscription(
            Image, self.get_parameter('image_topic').value,
            self._image_cb, qos_sensor
        )
        self._info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self._camera_info_cb, 10
        )
        self._detected_pub = self.create_publisher(Bool, '/aruco/detected', 10)

        if self._debug:
            self._debug_pub = self.create_publisher(Image, '/aruco/debug_image', 1)
        else:
            self._debug_pub = None

        self.get_logger().info(
            f"ArUco detector ready.\n"
            f"  Marker ID   : {self._marker_id}\n"
            f"  Marker size : {self._marker_size:.4f} m\n"
            f"  Dictionary  : {dict_name}\n"
            f"  Image topic : {self.get_parameter('image_topic').value}\n"
            f"  Publishing  : kinect2 → {self._marker_frame}"
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._K is None:
            self._K = np.array(msg.k).reshape(3, 3)
            self._D = np.array(msg.d)
            self.get_logger().info(
                f"Camera intrinsics received. fx={self._K[0,0]:.1f} fy={self._K[1,1]:.1f}"
            )

    def _image_cb(self, msg: Image) -> None:
        if self._K is None:
            return  # Wait for camera info

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"CV bridge failed: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Detect ──────────────────────────────────────────────────────────
        if self._use_new_api:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._aruco_dict, parameters=self._aruco_params
            )

        detected = False
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id != self._marker_id:
                    continue

                # ── Pose estimation ─────────────────────────────────────────
                half = self._marker_size / 2.0
                obj_pts = np.array([
                    [-half,  half, 0.0],
                    [ half,  half, 0.0],
                    [ half, -half, 0.0],
                    [-half, -half, 0.0],
                ], dtype=np.float64)

                corner = corners[i].reshape(4, 2).astype(np.float64)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, corner, self._K, self._D,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not ok:
                    continue

                rvec = rvec.flatten()
                tvec = tvec.flatten()
                quat = _rot_to_quat(rvec)

                # ── Publish TF ──────────────────────────────────────────────
                tf_msg = TransformStamped()
                tf_msg.header.stamp    = msg.header.stamp
                tf_msg.header.frame_id = self._cam_frame
                tf_msg.child_frame_id  = self._marker_frame
                tf_msg.transform.translation.x = float(tvec[0])
                tf_msg.transform.translation.y = float(tvec[1])
                tf_msg.transform.translation.z = float(tvec[2])
                tf_msg.transform.rotation.x = float(quat[0])
                tf_msg.transform.rotation.y = float(quat[1])
                tf_msg.transform.rotation.z = float(quat[2])
                tf_msg.transform.rotation.w = float(quat[3])
                self._tf_bcast.sendTransform(tf_msg)

                detected = True

                self.get_logger().debug(
                    f"Marker {marker_id}: t=({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})"
                )

                # ── Debug image ─────────────────────────────────────────────
                if self._debug_pub is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    try:
                        cv2.drawFrameAxes(
                            frame, self._K, self._D,
                            rvec, tvec, self._marker_size * 0.5
                        )
                    except Exception:
                        pass

        # Publish detection status
        self._detected_pub.publish(Bool(data=detected))

        if self._debug_pub is not None:
            self._debug_pub.publish(
                self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            )


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
