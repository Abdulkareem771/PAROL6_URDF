#!/usr/bin/env python3
"""
manual_line_aligner_node.py
===========================
A hybrid computer vision node that dynamically aligns manually drawn welding
strokes to moving parts using ORB feature matching and Affine Transformations.

Subscribed:
  /vision/captured_image_color (sensor_msgs/Image)

Published:
  /vision/processing_mode/annotated_image (sensor_msgs/Image)
  /vision/processing_mode/debug_image (sensor_msgs/Image)
  /vision/processing_mode/seam_centroid (geometry_msgs/PointStamped)

Services:
  ~/set_strokes (std_srvs/Trigger) - Legacy fixed stroke injection
  ~/teach_reference (std_srvs/Trigger) - Taught via an ROI polygon to enable Auto-Align
  ~/reset_strokes (std_srvs/Trigger) - Clears in-memory cache and saved configurations
"""

import json
import base64
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import time

import cv2
import numpy as np

_DEFAULT_IMAGE_TOPIC = '/vision/captured_image_color'
_DEFAULT_CONFIG_PATH = os.path.expanduser('~/.parol6/manual_aligner_config.json')

class ManualLineAlignerNode(Node):
    def __init__(self):
        super().__init__('manual_line_aligner')

        self.declare_parameter('image_topic',   _DEFAULT_IMAGE_TOPIC)
        self.declare_parameter('stroke_color',  [0, 0, 255])
        self.declare_parameter('stroke_width',  5)
        self.declare_parameter('strokes_json',  '')
        self.declare_parameter('publish_debug', True)

        self._in_topic     = self.get_parameter('image_topic').value
        self._stroke_color = list(self.get_parameter('stroke_color').value)
        self._stroke_width = int(self.get_parameter('stroke_width').value)
        self._publish_debug = self.get_parameter('publish_debug').value

        self._strokes = []
        self._roi_polygon = []
        self._ref_kpts = []
        self._ref_desc = None
        self._ref_size = None

        self._last_image = None
        self._last_matrix = None  # For temporal smoothing
        self._alpha = 0.5         # Exponential smoothing factor

        self._bridge = CvBridge()
        self._orb = cv2.ORB_create(1000)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # False, so knnMatch works

        self._ann_pub = self.create_publisher(Image, '/vision/processing_mode/annotated_image', 10)
        self._centroid_pub = self.create_publisher(PointStamped, '/vision/processing_mode/seam_centroid', 10)
        if self._publish_debug:
            self._debug_pub = self.create_publisher(Image, '/vision/processing_mode/debug_image', 10)

        self._sub = self.create_subscription(Image, self._in_topic, self._image_callback, 10)

        # Services
        self.create_service(Trigger, '~/set_strokes', self._svc_set_strokes)
        self.create_service(Trigger, '~/teach_reference', self._svc_teach_reference)
        self.create_service(Trigger, '~/reset_strokes', self._svc_reset_strokes)

        self._config_path = _DEFAULT_CONFIG_PATH
        self._load_config()

        self.get_logger().info('manual_line_aligner initialized.')

    def _load_config(self):
        # Primary: aligner's own config
        cfg_path = self._config_path
        # Fallback: load legacy fixed-mode strokes if aligner config is absent
        _legacy_path = os.path.expanduser('~/.parol6/manual_line_config.json')
        if not os.path.isfile(cfg_path) and os.path.isfile(_legacy_path):
            cfg_path = _legacy_path
            self.get_logger().warn(f'[Aligner] No aligner config found — falling back to legacy manual_line config.')

        if not os.path.isfile(cfg_path):
            return
        try:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            self._stroke_color = cfg.get('color', self._stroke_color)
            self._stroke_width = int(cfg.get('width', self._stroke_width))
            self._strokes      = cfg.get('strokes', [])
            
            ref = cfg.get('reference', None)
            if ref and 'descriptors' in ref and 'shape' in ref:
                self._roi_polygon = ref.get('polygon', [])
                self._ref_size = ref.get('image_size', None)
                
                b64_data = ref['descriptors']['data']
                shape = tuple(ref['descriptors']['shape'])
                desc_bytes = base64.b64decode(b64_data)
                self._ref_desc = np.frombuffer(desc_bytes, dtype=np.uint8).reshape(shape)
                
                kpts_data = ref.get('keypoints', [])
                self._ref_kpts = [cv2.KeyPoint(x=pt[0], y=pt[1], size=10) for pt in kpts_data]
                
            self.get_logger().info(f'[Aligner] Loaded {len(self._strokes)} strokes, {len(self._ref_kpts)} features.')
        except Exception as exc:
            self.get_logger().error(f'[Aligner] Failed to load config: {exc}')

    def _save_config(self):
        try:
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            cfg = {
                'color': self._stroke_color,
                'width': self._stroke_width,
                'strokes': self._strokes
            }
            if self._ref_desc is not None:
                b64_data = base64.b64encode(self._ref_desc.tobytes()).decode('ascii')
                cfg['reference'] = {
                    'polygon': self._roi_polygon,
                    'keypoints': [[k.pt[0], k.pt[1]] for k in self._ref_kpts],
                    'descriptors': {
                        'data': b64_data,
                        'shape': list(self._ref_desc.shape)
                    },
                    'image_size': self._ref_size
                }
            with open(self._config_path, 'w') as f:
                json.dump(cfg, f)
            self.get_logger().info('[Aligner] Config saved.')
        except Exception as exc:
            self.get_logger().error(f'[Aligner] Save failed: {exc}')

    def _svc_set_strokes(self, req, response):
        """Legacy pass-through or fixed mode if no ROI is taught."""
        try:
            raw = self.get_parameter('strokes_json').value
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    self._strokes = data.get('strokes', [])
                    self._stroke_color = data.get('color', self._stroke_color)
                    self._stroke_width = int(data.get('width', self._stroke_width))
                    self._roi_polygon = data.get('roi_polygon', [])
                elif isinstance(data, list):
                    self._strokes = data
                    self._roi_polygon = []
                self._ref_kpts = []
                self._ref_desc = None
                self._last_matrix = None
                self._save_config()
            response.success = True
            response.message = f'Loaded fixed strokes: {len(self._strokes)}.'
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _svc_teach_reference(self, req, response):
        """Extracts ORB features from the current ROI using the last cached image."""
        try:
            if self._last_image is None:
                response.success = False
                response.message = "No frame received yet."
                return response
            
            raw = self.get_parameter('strokes_json').value
            if not raw:
                response.success = False
                response.message = "strokes_json empty."
                return response
                
            data = json.loads(raw)
            if not isinstance(data, dict):
                response.success = False
                response.message = "strokes_json must be a dict for teach mode."
                return response

            self._strokes = data.get('strokes', [])
            self._roi_polygon = data.get('roi_polygon', [])
            self._stroke_color = data.get('color', self._stroke_color)
            self._stroke_width = int(data.get('width', self._stroke_width))
            
            if len(self._roi_polygon) < 3:
                response.success = False
                response.message = "ROI polygon must have at least 3 points."
                return response
                
            area = cv2.contourArea(np.array(self._roi_polygon, np.int32))
            if area < 500:
                response.success = False
                response.message = f"ROI area too small ({area} < 500 px)."
                return response

            # 1. ROI Feature Extraction
            img = self._last_image
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            poly_pts = np.array(self._roi_polygon, np.int32)
            cv2.fillPoly(mask, [poly_pts], 255)
            
            kpts, desc = self._orb.detectAndCompute(img, mask=mask)
            
            if desc is None or len(kpts) < 10:
                response.success = False
                response.message = f"Only found {len(kpts)} features in ROI. Needs >10."
                return response
                
            self._ref_kpts = kpts
            self._ref_desc = desc
            self._ref_size = [img.shape[1], img.shape[0]]
            self._last_matrix = None
            
            self._save_config()
            response.success = True
            response.message = f"Taught {len(kpts)} features successfully."
        except Exception as exc:
            response.success = False
            response.message = f"Teach failed: {exc}"
        return response

    def _svc_reset_strokes(self, req, response):
        self._strokes = []
        self._roi_polygon = []
        self._ref_kpts = []
        self._ref_desc = None
        self._last_matrix = None
        if os.path.isfile(self._config_path):
            os.remove(self._config_path)
        response.success = True
        return response

    def _image_callback(self, msg: Image):
        start_t = time.time()
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._last_image = img.copy()
        except:
            return

        annotated = img.copy()
        debug = img.copy() if self._publish_debug else None
        bgr_color = tuple(int(c) for c in self._stroke_color)

        if not self._strokes:
            self._publish_results(msg, annotated, debug, None)
            return

        # Fixed Mode (No ROI taught)
        if self._ref_desc is None:
            self._draw_fixed(annotated, debug, bgr_color)
            if debug is not None:
                dt = time.time() - start_t
                cv2.putText(debug, f"FPS: {1.0/(dt+1e-5):.1f}", (debug.shape[1] - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            self._publish_results(msg, annotated, debug, self._calc_centroid(annotated.shape, self._strokes))
            return

        # Adaptive Mode (Run Phase)
        centroid_px = None
        curr_kpts, curr_desc = self._orb.detectAndCompute(img, mask=None)

        if curr_desc is None or len(curr_kpts) < 10:
            if debug is not None: self._draw_banner(debug, "LOW FEATURES IN FRAME", (0, 0, 255))
            self._publish_results(msg, annotated, debug, None)
            return

        # KNN Match + Lowe's Ratio Test
        matches = self._bf.knnMatch(self._ref_desc, curr_desc, k=2)
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            if debug is not None: self._draw_banner(debug, f"LOW MATCHES: {len(good_matches)}", (0, 0, 255))
            self._publish_results(msg, annotated, debug, None)
            return

        # Spatial Filtering / Keep Top 50
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]

        src_pts = np.float32([self._ref_kpts[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([curr_kpts[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Check spatial spread
        std = np.std(src_pts.reshape(-1, 2), axis=0)
        if std[0] < 10 or std[1] < 10:
            if debug is not None: self._draw_banner(debug, "POOR SPATIAL SPREAD", (0, 0, 255))
            self._publish_results(msg, annotated, debug, None)
            return

        matrix, inliers_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        if matrix is None or inliers_mask is None:
            if debug is not None: self._draw_banner(debug, "RANSAC FAIL", (0, 0, 255))
            self._publish_results(msg, annotated, debug, None)
            return

        # Validation: Inlier Ratio
        inlier_ratio = np.sum(inliers_mask) / len(inliers_mask)
        if inlier_ratio < 0.4:
            if debug is not None: self._draw_banner(debug, f"LOW INLIER RATIO: {inlier_ratio:.2f}", (0, 0, 255))
            self._publish_results(msg, annotated, debug, None)
            return

        # Validation: Determinant & Scale Check
        det = abs(np.linalg.det(matrix[:, :2]))
        scale = np.linalg.norm(matrix[:, 0])
        if det < 0.1 or scale < 0.5 or scale > 2.0:
            if debug is not None: self._draw_banner(debug, "TRANSFORM DEGENERATE", (0, 0, 255))
            self._publish_results(msg, annotated, debug, None)
            return

        # Temporal Smoothing: TRANSLATION ONLY (Prevent matrix skew/distortion buildup)
        if self._last_matrix is not None:
            matrix[:, 2] = self._alpha * matrix[:, 2] + (1.0 - self._alpha) * self._last_matrix[:, 2]
        self._last_matrix = matrix

        # Transform Strokes
        transformed_strokes = []
        for stroke in self._strokes:
            pts = np.float32(stroke).reshape(-1, 1, 2)
            t_pts = cv2.transform(pts, matrix)
            transformed_strokes.append(t_pts.reshape(-1, 2).astype(np.int32).tolist())

        # Draw Output
        for t_stroke in transformed_strokes:
            if len(t_stroke) >= 2:
                pts = np.array(t_stroke, dtype=np.int32)
                cv2.polylines(annotated, [pts], False, bgr_color, self._stroke_width, cv2.LINE_AA)

        centroid_px = self._calc_centroid(annotated.shape, transformed_strokes)

        if debug is not None:
            # 1. Transformed vs Original Strokes
            for orig in self._strokes:
                cv2.polylines(debug, [np.array(orig, np.int32)], False, (150, 150, 150), 2, cv2.LINE_AA) # faded gray
            for t_stroke in transformed_strokes:
                cv2.polylines(debug, [np.array(t_stroke, np.int32)], False, (0, 255, 0), self._stroke_width, cv2.LINE_AA)
            
            # 2. Inlier matches
            if inliers_mask is not None:
                inliers = inliers_mask.ravel().tolist()
                drawn = 0
                for i, match in enumerate(good_matches):
                    if inliers[i]:
                        pt_ref = np.float32([[[self._ref_kpts[match.queryIdx].pt[0], self._ref_kpts[match.queryIdx].pt[1]]]])
                        pt1_transformed = cv2.transform(pt_ref, matrix)[0][0]
                        pt1 = tuple(np.int32(pt1_transformed))
                        pt2 = tuple(np.int32(curr_kpts[match.trainIdx].pt))
                        
                        cv2.circle(debug, pt2, 4, (0, 255, 0), -1)
                        cv2.line(debug, pt1, pt2, (0, 150, 0), 1)
                        drawn += 1
                        if drawn > 20: break  # limit density

            # 3. ROI Overlay
            if self._roi_polygon:
                pts = np.float32(self._roi_polygon).reshape(-1, 1, 2)
                t_roi = cv2.transform(pts, matrix)
                cv2.polylines(debug, [np.int32(t_roi)], True, (255, 100, 0), 2, cv2.LINE_AA)
            
            self._draw_banner(debug, "ALIGNMENT OK", (0, 255, 0))

            # FPS
            dt = time.time() - start_t
            cv2.putText(debug, f"FPS: {1.0/(dt+1e-5):.1f}", (debug.shape[1] - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        self._publish_results(msg, annotated, debug, centroid_px)

    def _draw_fixed(self, annotated, debug, color):
        for stroke in self._strokes:
            if len(stroke) >= 2:
                cv2.polylines(annotated, [np.array(stroke, np.int32)], False, color, self._stroke_width, cv2.LINE_AA)
                if debug is not None:
                    cv2.polylines(debug, [np.array(stroke, np.int32)], False, color, self._stroke_width, cv2.LINE_AA)
        if debug is not None:
            self._draw_banner(debug, "FIXED MODE (NO ROI)", (255, 150, 0))

    def _calc_centroid(self, shape, strokes):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for stroke in strokes:
            if len(stroke) >= 2:
                cv2.polylines(mask, [np.array(stroke, np.int32)], False, 255, self._stroke_width)
        ys, xs = np.where(mask > 0)
        return (float(xs.mean()), float(ys.mean())) if len(xs) > 0 else None

    def _draw_banner(self, img, text, color):
        cv2.rectangle(img, (0, 0), (320, 26), (30, 30, 30), -1)
        cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def _publish_results(self, msg, ann, dbg, centroid):
        ann_msg = self._bridge.cv2_to_imgmsg(ann, encoding='bgr8')
        ann_msg.header = msg.header
        self._ann_pub.publish(ann_msg)

        if dbg is not None and self._publish_debug:
            dbg_msg = self._bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            dbg_msg.header = msg.header
            self._debug_pub.publish(dbg_msg)

        if centroid is not None:
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x, pt.point.y, pt.point.z = centroid[0], centroid[1], 0.0
            self._centroid_pub.publish(pt)

def main(args=None):
    rclpy.init(args=args)
    node = ManualLineAlignerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
