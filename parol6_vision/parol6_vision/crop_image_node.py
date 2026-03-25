#!/usr/bin/env python3
"""
crop_image_node.py — PAROL6 Vision Pipeline (Stage 1b)
=======================================================
Always-active node that sits between capture_images and the rest of the pipeline.

  /vision/captured_image_raw  ──►  [crop_image_node]  ──►  /vision/captured_image_color

Behaviour
---------
  • On startup: auto-loads config from `~/.parol6/crop_config.json`.
    - If file is missing or `enabled: false` → full frame pass-through.
  • Supports two modes (set via config `mode` field):
      "mask"  — zero out everything OUTSIDE the polygon; output has the SAME
                resolution as input. Pixel coordinates are preserved, so depth
                maps and downstream nodes remain correctly aligned.  ← Default
      "crop"  — rectangular crop to the polygon's bounding box; output is a
                smaller image. Pixel coordinates change — use only if downstream
                nodes do not depend on absolute pixel positions.
  • Each incoming frame is processed once (no polling loop).
  • Service `~/reload_roi`  — re-read config file from disk; saves & applies live.
  • Service `~/clear_roi`   — disable processing; saves `enabled: false` to JSON.

Config file format (~/.parol6/crop_config.json)
-------------------------------------------------
Mask mode (recommended — default for new configs):
{
  "enabled": true,
  "mode": "mask",
  "polygon": [[x1,y1], [x2,y2], ...],   # image pixel coordinates
  "mask_color": [0, 0, 0]               # RGB fill for masked region (default = black)
}

Crop mode (legacy / backward-compat):
{
  "enabled": true,
  "mode": "crop",
  "x": 120, "y": 80,
  "width": 640, "height": 400
}
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from rclpy.parameter import Parameter

_DEFAULT_CONFIG = Path.home() / ".parol6" / "crop_config.json"
_DEFAULT_IN     = "/vision/captured_image_raw"
_DEFAULT_OUT    = "/vision/captured_image_color"


class CropImageNode(Node):
    """
    Always-active relay node.

    Subscribed Topics:
        /vision/captured_image_raw  (sensor_msgs/Image)

    Published Topics:
        /vision/captured_image_color  (sensor_msgs/Image)

    Services:
        ~/reload_roi   (std_srvs/Trigger) — re-read config file from disk
        ~/clear_roi    (std_srvs/Trigger) — disable processing, saves config

    ROS Parameters:
        input_topic   (str)   — default /vision/captured_image_raw
        output_topic  (str)   — default /vision/captured_image_color
        config_path   (str)   — default ~/.parol6/crop_config.json
        roi           (int[4]) — [x, y, width, height] bounding box; sets crop mode
    """

    def __init__(self):
        super().__init__("crop_image")

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter("input_topic",  _DEFAULT_IN)
        self.declare_parameter("output_topic", _DEFAULT_OUT)
        self.declare_parameter("config_path",  str(_DEFAULT_CONFIG))
        self.declare_parameter("roi", rclpy.Parameter.Type.INTEGER_ARRAY)

        self._in_topic  = self.get_parameter("input_topic").value
        self._out_topic = self.get_parameter("output_topic").value
        self._cfg_path  = Path(self.get_parameter("config_path").value)

        # ── State ─────────────────────────────────────────────────────
        self._enabled    = False
        self._mode:      str  = "mask"      # "mask" or "crop"
        self._polygon:   list | None = None # [[x,y],...] image coords (mask mode)
        self._roi:       tuple | None = None # (x,y,w,h) pixel coords (crop mode)
        self._mask_color: list = [0, 0, 0]  # RGB fill color for masked region

        self._bridge    = CvBridge()
        self._latest_msg: Image | None = None

        # Load config from disk
        self._load_config()

        # ── Publishers / Subscribers ──────────────────────────────────
        self._pub = self.create_publisher(Image, self._out_topic, 10)
        self._sub = self.create_subscription(
            Image, self._in_topic, self._image_callback, 10
        )

        # ── Services ──────────────────────────────────────────────────
        self.create_service(Trigger, "~/reload_roi", self._svc_reload)
        self.create_service(Trigger, "~/clear_roi",  self._svc_clear)

        # Parameter-change callback (GUI can push roi via ros2 param set)
        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            f"Crop Image node ready.\n"
            f"  Input  : {self._in_topic}\n"
            f"  Output : {self._out_topic}\n"
            f"  Config : {self._cfg_path}\n"
            f"  Mode   : {self._mode}\n"
            f"  Active : {'YES — poly=' + str(self._polygon) if self._enabled and self._mode == 'mask' else 'YES — roi=' + str(self._roi) if self._enabled else 'NO (pass-through)'}"
        )

    # ── Image callback ────────────────────────────────────────────────

    def _image_callback(self, msg: Image) -> None:
        self._latest_msg = msg
        self._publish_current(msg)

    def _publish_current(self, msg: Image | None = None) -> None:
        """Publish the latest frame using the currently active processing settings."""
        if msg is None:
            msg = self._latest_msg
        if msg is None:
            return

        if not self._enabled:
            self._pub.publish(msg)
            return

        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            if self._mode == "mask" and self._polygon:
                result = self._apply_polygon_mask(cv_img, self._polygon)
            elif self._mode == "crop" and self._roi:
                result = self._apply_crop(cv_img, self._roi)
            else:
                # Misconfigured — pass through
                self._pub.publish(msg)
                return

            out_msg = self._bridge.cv2_to_imgmsg(result, encoding=msg.encoding)
            out_msg.header = msg.header
            self._pub.publish(out_msg)

        except Exception as exc:
            self.get_logger().error(f"Processing error: {exc}")
            self._pub.publish(msg)   # Fallback: pass original

    # ── Processing helpers ────────────────────────────────────────────

    def _apply_polygon_mask(self, cv_img: np.ndarray, polygon: list) -> np.ndarray:
        """
        Fill all pixels outside the polygon with self._mask_color.
        Output has the SAME shape as input — depth coordinates are preserved.
        """
        ih, iw = cv_img.shape[:2]
        pts = np.array(
            [[max(0, min(x, iw - 1)), max(0, min(y, ih - 1))] for x, y in polygon],
            dtype=np.int32,
        )
        mask = np.zeros((ih, iw), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Build the fill color array matching the image shape
        r, g, b = self._mask_color
        if cv_img.ndim == 3:
            channels = cv_img.shape[2]
            if channels == 3:
                # BGR order for OpenCV images
                fill = np.empty_like(cv_img, dtype=cv_img.dtype)
                fill[:] = [b, g, r]
            else:
                # Fallback for other channel counts
                fill = np.zeros_like(cv_img, dtype=cv_img.dtype)
            mask3 = mask[:, :, np.newaxis]
            result = np.where(mask3 == 255, cv_img, fill)
        else:
            # Grayscale: use luminance of mask color
            lum = int(0.299 * r + 0.587 * g + 0.114 * b)
            fill = np.full_like(cv_img, lum, dtype=cv_img.dtype)
            result = np.where(mask == 255, cv_img, fill)

        return result.astype(cv_img.dtype)

    def _apply_crop(self, cv_img: np.ndarray, roi: tuple) -> np.ndarray:
        """Rectangular crop to roi (x, y, w, h). Changes image dimensions."""
        x, y, w, h = roi
        ih, iw = cv_img.shape[:2]
        x  = max(0, min(x, iw - 1))
        y  = max(0, min(y, ih - 1))
        x2 = min(x + w, iw)
        y2 = min(y + h, ih)
        return cv_img[y:y2, x:x2]

    # ── Config helpers ────────────────────────────────────────────────

    def _load_config(self) -> None:
        """Load config from JSON file.  Missing file = pass-through."""
        if not self._cfg_path.exists():
            self.get_logger().info("No crop config found — pass-through mode.")
            self._enabled = False
            self._mode    = "mask"
            self._polygon = None
            self._roi     = None
            return

        try:
            with open(self._cfg_path) as f:
                cfg = json.load(f)

            self._enabled = bool(cfg.get("enabled", False))
            # Default to "mask" for new configs; "crop" only if explicitly set.
            # Old configs without a "mode" field are assumed to be crop (they
            # won't have a "polygon" key, so mask would fail gracefully anyway).
            self._mode = cfg.get("mode", "mask" if "polygon" in cfg else "crop")
            self._mask_color = cfg.get("mask_color", [0, 0, 0])

            if self._enabled:
                if self._mode == "mask":
                    raw_poly = cfg.get("polygon", [])
                    if len(raw_poly) >= 3:
                        self._polygon = [[int(x), int(y)] for x, y in raw_poly]
                        self._roi = None
                    elif "x" in cfg:
                        # Graceful fallback: polygon missing but bbox present —
                        # convert bbox to 4-corner polygon so mask mode still works.
                        bx, by = int(cfg["x"]), int(cfg["y"])
                        bw, bh = int(cfg["width"]), int(cfg["height"])
                        self._polygon = [
                            [bx, by], [bx + bw, by],
                            [bx + bw, by + bh], [bx, by + bh]
                        ]
                        self._roi = None
                        self.get_logger().warning(
                            "Mask mode with no polygon — falling back to bbox as rectangle mask."
                        )
                    else:
                        self.get_logger().warning(
                            "Mask mode but no polygon and no bbox — disabling."
                        )
                        self._enabled = False
                        self._polygon = None
                else:
                    # Crop mode (legacy)
                    self._roi = (
                        int(cfg["x"]),
                        int(cfg["y"]),
                        int(cfg["width"]),
                        int(cfg["height"]),
                    )
                    self._polygon = None
            else:
                self._polygon = None
                self._roi     = None

            self.get_logger().info(
                f"Config loaded: enabled={self._enabled} mode={self._mode} "
                f"polygon_pts={len(self._polygon) if self._polygon else 0} "
                f"roi={self._roi}"
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to load crop config: {exc}")
            self._enabled = False
            self._polygon = None
            self._roi     = None

    def _save_config(self) -> None:
        """Persist current config to JSON file."""
        self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg: dict = {
            "enabled":    self._enabled,
            "mode":       self._mode,
            "mask_color": self._mask_color,
        }
        if self._polygon:
            cfg["polygon"] = self._polygon
        if self._roi:
            x, y, w, h = self._roi
            cfg.update({"x": x, "y": y, "width": w, "height": h})
        with open(self._cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        self.get_logger().info(f"Config saved to {self._cfg_path}")

    # ── Services ──────────────────────────────────────────────────────

    def _svc_reload(self, _req, response):
        """Re-read config file from disk without restarting."""
        self._load_config()
        self._publish_current()
        response.success = True
        response.message = (
            f"ROI reloaded: enabled={self._enabled} mode={self._mode} "
            f"polygon_pts={len(self._polygon) if self._polygon else 0} "
            f"roi={self._roi}"
        )
        return response

    def _svc_clear(self, _req, response):
        """Disable processing and save disabled config."""
        self._enabled = False
        self._polygon = None
        self._roi     = None
        self._save_config()
        self._publish_current()
        response.success = True
        response.message = "Processing cleared — pass-through mode."
        self.get_logger().info("Processing disabled (pass-through).")
        return response

    # ── Parameter-change callback ─────────────────────────────────────

    def _on_param_change(self, params):
        """
        Respond to `ros2 param set /crop_image roi "[x,y,w,h]"`.
        Sets crop mode (legacy behaviour, kept for compatibility).
        For mask mode, write the config file and call ~/reload_roi instead.
        """
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "roi" and p.type_ == Parameter.Type.INTEGER_ARRAY:
                vals = list(p.value)
                if len(vals) == 4 and all(v >= 0 for v in vals) and vals[2] > 0 and vals[3] > 0:
                    self._roi     = tuple(int(v) for v in vals)
                    self._mode    = "crop"
                    self._polygon = None
                    self._enabled = True
                    self._save_config()
                    self._publish_current()
                    self.get_logger().info(f"ROI updated via param (crop mode): {self._roi}")
        return SetParametersResult(successful=True)


# ── Entry point ────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = CropImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
