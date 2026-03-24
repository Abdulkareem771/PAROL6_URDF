#!/usr/bin/env python3
"""
crop_image_node.py — PAROL6 Vision Pipeline (Stage 1b)
=======================================================
Always-active node that sits between capture_images and the rest of the pipeline.

  /vision/captured_image_raw  ──►  [crop_image_node]  ──►  /vision/captured_image_color

Behaviour
---------
  • On startup: auto-loads ROI from `~/.parol6/crop_config.json`.
    - If file is missing or `enabled: false` → full frame pass-through.
  • Each incoming frame is processed once (no polling loop).
  • Service `~/update_roi`  — push new ROI from the GUI; saves JSON and applies live.
  • Service `~/clear_roi`   — disable crop; saves `enabled: false` to JSON.

Config file format (~/.parol6/crop_config.json)
-------------------------------------------------
{
  "enabled": true,
  "x": 120, "y": 80,
  "width": 640, "height": 400,
  "source_image_size": [1920, 1080]
}
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, Trigger
from cv_bridge import CvBridge

# Custom service for sending ROI: reuse standard Int32MultiArray via a parameter service
# We use SetBool for clear_roi (True → clear) and a custom-param approach for update.
# For simplicity we use the ROS 2 parameter server: the GUI can call set_parameters.
# Alternatively, we expose a simple Float64MultiArray topic for ROI updates.
# ↓ We use a dedicated service via std_msgs approach below.

# Because we can't ship a custom .srv without rebuilding, we use the parameter service:
#   ros2 param set /crop_image roi "[x, y, w, h]" then call ~/reload_roi

from rcl_interfaces.msg import ParameterType
from rclpy.parameter import Parameter

_DEFAULT_CONFIG = Path.home() / ".parol6" / "crop_config.json"
_DEFAULT_IN     = "/vision/captured_image_raw"
_DEFAULT_OUT    = "/vision/captured_image_color"


class CropImageNode(Node):
    """
    Always-active crop relay node.

    Subscribed Topics:
        /vision/captured_image_raw  (sensor_msgs/Image)

    Published Topics:
        /vision/captured_image_color  (sensor_msgs/Image)

    Services:
        ~/reload_roi   (std_srvs/Trigger) — re-read config file from disk
        ~/clear_roi    (std_srvs/Trigger) — disable crop (pass-through), saves config

    ROS Parameters:
        input_topic   (str)  — default /vision/captured_image_raw
        output_topic  (str)  — default /vision/captured_image_color
        config_path   (str)  — default ~/.parol6/crop_config.json
        roi           (int[4]) — [x, y, width, height]; set via ros2 param set
    """

    def __init__(self):
        super().__init__("crop_image")

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter("input_topic",  _DEFAULT_IN)
        self.declare_parameter("output_topic", _DEFAULT_OUT)
        self.declare_parameter("config_path",  str(_DEFAULT_CONFIG))
        # roi: [x, y, width, height] — overrides config file if non-empty
        self.declare_parameter("roi", rclpy.Parameter.Type.INTEGER_ARRAY)

        self._in_topic  = self.get_parameter("input_topic").value
        self._out_topic = self.get_parameter("output_topic").value
        self._cfg_path  = Path(self.get_parameter("config_path").value)

        # ── State ─────────────────────────────────────────────────────
        self._enabled = False
        self._roi: tuple[int, int, int, int] | None = None   # x, y, w, h
        self._bridge = CvBridge()

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

        # Parameter-change callback (so GUI can push ROI via ros2 param set)
        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            f"Crop Image node ready.\n"
            f"  Input  : {self._in_topic}\n"
            f"  Output : {self._out_topic}\n"
            f"  Config : {self._cfg_path}\n"
            f"  Crop   : {'ENABLED — ROI=' + str(self._roi) if self._enabled else 'DISABLED (pass-through)'}"
        )

    # ── Image callback (one call per frame, no loop) ──────────────────

    def _image_callback(self, msg: Image) -> None:
        if not self._enabled or self._roi is None:
            # Pass-through: republish unchanged
            self._pub.publish(msg)
            return

        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            x, y, w, h = self._roi
            ih, iw = cv_img.shape[:2]

            # Clamp ROI to actual image size
            x  = max(0, min(x, iw - 1))
            y  = max(0, min(y, ih - 1))
            x2 = min(x + w, iw)
            y2 = min(y + h, ih)

            cropped = cv_img[y:y2, x:x2]
            out_msg = self._bridge.cv2_to_imgmsg(cropped, encoding=msg.encoding)
            out_msg.header = msg.header
            self._pub.publish(out_msg)

        except Exception as exc:
            self.get_logger().error(f"Crop error: {exc}")
            self._pub.publish(msg)   # Fallback: pass original

    # ── Config helpers ────────────────────────────────────────────────

    def _load_config(self) -> None:
        """Load ROI from JSON config file.  Missing file = pass-through."""
        if not self._cfg_path.exists():
            self.get_logger().info("No crop config found — pass-through mode.")
            self._enabled = False
            self._roi     = None
            return

        try:
            with open(self._cfg_path) as f:
                cfg = json.load(f)
            self._enabled = bool(cfg.get("enabled", False))
            if self._enabled:
                self._roi = (
                    int(cfg["x"]),
                    int(cfg["y"]),
                    int(cfg["width"]),
                    int(cfg["height"]),
                )
            else:
                self._roi = None
            self.get_logger().info(
                f"Config loaded: enabled={self._enabled}  roi={self._roi}"
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to load crop config: {exc}")
            self._enabled = False
            self._roi     = None

    def _save_config(self) -> None:
        """Persist current ROI to JSON config file."""
        self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg: dict = {"enabled": self._enabled}
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
        response.success = True
        response.message = (
            f"ROI reloaded: enabled={self._enabled} roi={self._roi}"
        )
        return response

    def _svc_clear(self, _req, response):
        """Disable cropping and save disabled config."""
        self._enabled = False
        self._roi     = None
        self._save_config()
        response.success = True
        response.message = "Crop cleared — pass-through mode."
        self.get_logger().info("Crop disabled (pass-through).")
        return response

    # ── Parameter-change callback ─────────────────────────────────────

    def _on_param_change(self, params):
        """
        Respond to `ros2 param set /crop_image roi "[x,y,w,h]"`.
        After setting roi param, call ~/reload_roi or the node applies immediately.
        """
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "roi" and p.type_ == Parameter.Type.INTEGER_ARRAY:
                vals = list(p.value)
                if len(vals) == 4 and all(v >= 0 for v in vals) and vals[2] > 0 and vals[3] > 0:
                    self._roi     = tuple(int(v) for v in vals)
                    self._enabled = True
                    self._save_config()
                    self.get_logger().info(f"ROI updated via param: {self._roi}")
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
