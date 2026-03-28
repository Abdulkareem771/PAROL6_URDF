#!/usr/bin/env python3
"""
inject_path_node — PAROL6 Vision Pipeline (Test Path Source)
=============================================================
Receives a hand-authored nav_msgs/Path from the GUI and forwards it to the
staging topic /vision/welding_path/injected as a TRANSIENT_LOCAL latch.

This node is the proper ROS replacement for every bash/python inject_path
shell script. The GUI publishes to /vision/inject_path using its own
persistent rclpy publisher (no subprocess, no DDS discovery race) and
this node holds and re-latches the result.

Pipeline position
-----------------
  GUI ──► /vision/inject_path ──► inject_path_node ──► /vision/welding_path/injected
                                                                  │
                                                            path_holder (reads staging)

Subscribed Topics
-----------------
  /vision/inject_path  (nav_msgs/Path, VOLATILE)

Published Topics
----------------
  /vision/welding_path/injected  (nav_msgs/Path, TRANSIENT_LOCAL)

Services
--------
  ~/clear_path  (std_srvs/Trigger) — clear cached path, log a warning
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Path
from std_srvs.srv import Trigger


class InjectPathNode(Node):

    def __init__(self):
        super().__init__('inject_path_node')

        # ── Output QoS: latched so path_holder receives it on join ────────────
        latch_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._pub = self.create_publisher(Path, '/vision/welding_path/injected', latch_qos)

        # ── Input: VOLATILE is fine — GUI holds the persistent publisher ──────
        self._sub = self.create_subscription(
            Path,
            '/vision/inject_path',
            self._on_path,
            10,
        )

        self._latest_path: Path | None = None

        self.create_service(Trigger, '~/clear_path', self._svc_clear)

        self.get_logger().info(
            'inject_path_node ready.\n'
            '  Listening : /vision/inject_path\n'
            '  Publishing: /vision/welding_path/injected (TRANSIENT_LOCAL)'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_path(self, msg: Path) -> None:
        n = len(msg.poses)
        if n == 0:
            self.get_logger().warn('Received empty path — ignoring.')
            return

        # Ensure header is set
        if not msg.header.frame_id:
            msg.header.frame_id = 'base_link'
        msg.header.stamp = self.get_clock().now().to_msg()
        for pose in msg.poses:
            pose.header.stamp = msg.header.stamp
            if not pose.header.frame_id:
                pose.header.frame_id = msg.header.frame_id

        self._latest_path = msg
        self._pub.publish(msg)
        self.get_logger().info(
            f'Injected path with {n} waypoints → /vision/welding_path/injected (latched)'
        )

    def _svc_clear(self, _req, response: Trigger.Response) -> Trigger.Response:
        self._latest_path = None
        response.success = True
        response.message = 'Inject path cache cleared. Next inject will replace it.'
        self.get_logger().warn('Inject path cleared by service call.')
        return response


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = InjectPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
