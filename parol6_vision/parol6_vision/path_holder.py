#!/usr/bin/env python3
"""
path_holder — PAROL6 Vision Pipeline (Authoritative Path Publisher)
====================================================================
The single authoritative publisher of /vision/welding_path.

This node acts as a mux between two sources:
  • "generated" — live vision path from path_generator
  • "injected"  — hand-authored test path from inject_path_node

It caches both inputs independently and publishes one at a time to the
final /vision/welding_path topic as TRANSIENT_LOCAL. No other node should
publish directly to /vision/welding_path.

Source selection is explicit: the operator chooses via the GUI source
selector or via the ~/set_source service. Switching immediately
republishes the cached path of the new source (or fails cleanly if none).

================================================================================
Pipeline Position
================================================================================

  path_generator  ──► /vision/welding_path/generated (TRANSIENT_LOCAL) ──┐
                                                                           ├──► path_holder ──► /vision/welding_path (TRANSIENT_LOCAL)
  inject_path_node ──► /vision/welding_path/injected (TRANSIENT_LOCAL) ──┘               │
                                                                                   moveit_controller

================================================================================
Topics
================================================================================

Subscribed
  /vision/welding_path/generated  (nav_msgs/Path, TRANSIENT_LOCAL)
  /vision/welding_path/injected   (nav_msgs/Path, TRANSIENT_LOCAL)

Published
  /vision/welding_path            (nav_msgs/Path, TRANSIENT_LOCAL)

Services
  ~/set_source  (std_srvs/Trigger) — set active source via parameter then call
  ~/get_status  (std_srvs/Trigger) — returns current source + pose counts

Parameters
  active_source  (string) "generated" | "injected"  (default: "generated")
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point


_SOURCES = ('generated', 'injected')

_STAGING = {
    'generated': '/vision/welding_path/generated',
    'injected':  '/vision/welding_path/injected',
}


class PathHolder(Node):

    def __init__(self):
        super().__init__('path_holder')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('active_source', 'generated')

        self._active: str = self.get_parameter('active_source').value
        if self._active not in _SOURCES:
            self.get_logger().warn(
                f"Unknown active_source '{self._active}' — defaulting to 'generated'."
            )
            self._active = 'generated'

        # Cache for each source
        self._cache: dict[str, Path | None] = {s: None for s in _SOURCES}

        # ── QoS ───────────────────────────────────────────────────────────────
        latch_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        self._pub = self.create_publisher(Path, '/vision/welding_path', latch_qos)

        # Visualization: standard VOLATILE so RViz always sees it
        self._marker_pub = self.create_publisher(
            MarkerArray, '/vision/path_holder/markers', 10
        )

        # ── Subscribers (TRANSIENT_LOCAL to receive cached messages on join) ──
        self._sub_gen = self.create_subscription(
            Path,
            _STAGING['generated'],
            lambda msg: self._on_path('generated', msg),
            latch_qos,
        )
        self._sub_inj = self.create_subscription(
            Path,
            _STAGING['injected'],
            lambda msg: self._on_path('injected', msg),
            latch_qos,
        )

        # ── Services ──────────────────────────────────────────────────────────
        self.create_service(Trigger, '~/set_source',  self._svc_set_source)
        self.create_service(Trigger, '~/get_status',  self._svc_get_status)

        self.get_logger().info(
            f'path_holder ready — active_source={self._active}\n'
            f'  Holding  : /vision/welding_path (TRANSIENT_LOCAL, sole publisher)\n'
            f'  Watching : {_STAGING["generated"]}\n'
            f'  Watching : {_STAGING["injected"]}'
        )

    # ── Path callbacks ────────────────────────────────────────────────────────

    def _on_path(self, source: str, msg: Path) -> None:
        """Receive a new path from one of the staging topics and cache it."""
        n = len(msg.poses)
        self._cache[source] = msg
        self.get_logger().info(
            f'Cached {source} path ({n} waypoints).'
        )
        # If this is the active source, publish immediately
        if source == self._active:
            self._publish(msg, reason=f'new {source} path received')

    # ── Source management ─────────────────────────────────────────────────────

    def _switch_source(self, new_source: str) -> tuple[bool, str]:
        """
        Switch active source. Immediately republishes from cache if available.
        Returns (success, message).
        """
        if new_source not in _SOURCES:
            return False, f"Unknown source '{new_source}'. Valid: {_SOURCES}"

        cached = self._cache[new_source]
        if cached is None:
            return (
                False,
                f"Source '{new_source}' has no cached path yet. "
                "Current held path unchanged. "
                "Trigger the upstream pipeline first, then switch.",
            )

        old = self._active
        self._active = new_source
        # Set the parameter to reflect state
        self.set_parameters([
            rclpy.parameter.Parameter('active_source', value=new_source)
        ])
        self._publish(cached, reason=f'source switched {old}→{new_source}')
        n = len(cached.poses)
        return True, f"Source switched to '{new_source}' ({n} waypoints republished)."

    def _publish(self, path: Path, *, reason: str = '') -> None:
        """Stamp and publish path to the authoritative topic + RViz markers."""
        path.header.stamp = self.get_clock().now().to_msg()
        if not path.header.frame_id:
            path.header.frame_id = 'base_link'
        self._pub.publish(path)
        self._publish_markers(path)
        n = len(path.poses)
        self.get_logger().info(
            f'Published /vision/welding_path ({n} poses, source={self._active}'
            + (f', {reason}' if reason else '') + ')'
        )

    def _publish_markers(self, path: Path) -> None:
        """Publish path as MarkerArray to /vision/path_holder/markers (VOLATILE).
        
        Emits two markers per held path:
          id=0  LINE_STRIP  — the trajectory line (cyan)
          id=1  SPHERE_LIST — individual waypoints (orange for generated, purple for injected)
        Always uses the latest wall-clock stamp to prevent RViz from discarding as stale.
        """
        if not path.poses:
            return

        stamp = self.get_clock().now().to_msg()
        frame = path.header.frame_id or 'base_link'

        # Colour code by source
        if self._active == 'injected':
            wpt_color = ColorRGBA(r=0.7, g=0.3, b=1.0, a=0.9)   # purple
            line_color = ColorRGBA(r=0.5, g=0.0, b=1.0, a=0.8)
        else:
            wpt_color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9)   # orange
            line_color = ColorRGBA(r=0.0, g=0.9, b=0.9, a=0.8)  # cyan

        # 1) LINE_STRIP
        m_line = Marker()
        m_line.header.frame_id = frame
        m_line.header.stamp = stamp
        m_line.ns = 'path_holder'
        m_line.id = 0
        m_line.type = Marker.LINE_STRIP
        m_line.action = Marker.ADD
        m_line.scale.x = 0.003   # 3 mm line width
        m_line.color = line_color
        m_line.points = [Point(x=ps.pose.position.x,
                               y=ps.pose.position.y,
                               z=ps.pose.position.z) for ps in path.poses]

        # 2) SPHERE_LIST (individual waypoints)
        m_pts = Marker()
        m_pts.header.frame_id = frame
        m_pts.header.stamp = stamp
        m_pts.ns = 'path_holder'
        m_pts.id = 1
        m_pts.type = Marker.SPHERE_LIST
        m_pts.action = Marker.ADD
        m_pts.scale.x = 0.008
        m_pts.scale.y = 0.008
        m_pts.scale.z = 0.008
        m_pts.color = wpt_color
        m_pts.points = m_line.points

        ma = MarkerArray()
        ma.markers = [m_line, m_pts]
        self._marker_pub.publish(ma)


    # ── Services ──────────────────────────────────────────────────────────────

    def _svc_set_source(self, _req: Trigger.Request, response: Trigger.Response):
        """
        Switch active source.
        Caller should set the `active_source` parameter first:
            ros2 param set /path_holder active_source injected
            ros2 service call /path_holder/set_source std_srvs/srv/Trigger '{}'
        Or the GUI can call it after updating the parameter via SetParameters.
        """
        # Read the current parameter value (GUI updates it before calling service)
        requested = self.get_parameter('active_source').value
        ok, msg = self._switch_source(requested)
        response.success = ok
        response.message = msg
        lvl = self.get_logger().info if ok else self.get_logger().warn
        lvl(f'set_source: {msg}')
        return response

    def _svc_get_status(self, _req: Trigger.Request, response: Trigger.Response):
        """Return a human-readable status of current source and cache state."""
        parts = [f'active_source={self._active}']
        for src in _SOURCES:
            c = self._cache[src]
            parts.append(
                f'{src}={"ready (" + str(len(c.poses)) + " poses)" if c else "empty"}'
            )
        msg = ' | '.join(parts)
        response.success = True
        response.message = msg
        self.get_logger().info(f'Status: {msg}')
        return response


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PathHolder()
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
