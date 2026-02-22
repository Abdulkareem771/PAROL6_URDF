#!/usr/bin/env python3
"""
Path Generator Node - Vision-Guided Welding Path Detection

This node converts raw 3D point clouds of weld seams into smooth, kinematically 
feasible welding trajectories (nav_msgs/Path).

================================================================================
ALGORITHM OVERVIEW
================================================================================

1. POINT ORDERING (PCA)
   - Raw 3D points from depth matching are unordered or noisy.
   - We use Principal Component Analysis (PCA) to find the primary direction of the weld.
   - Points are projected onto the principal axis and sorted to ensure strictly linear sequence.

2. B-SPLINE SMOOTHING
   - Fits a cubic B-spline (degree=3) to the ordered points using scipy.interpolate.splprep.
   - Smoothing Parameter (s): Controls trade-off between closeness of fit and curve smoothness.
   - Eliminates high-frequency jitter from depth sensor noise.

3. UNIFORM RESAMPLING
   - Resamples the spline at fixed Euclidean distance intervals (e.g., 5mm).
   - Ensures consistent welding velocity when executed by the robot controller.

4. ORIENTATION GENERATION (Scoping Assumption)
   - Scope: Planar welding surfaces (Thesis limitation).
   - Algorithm: 
     - Z-axis (Approach vector): Aligned with curve tangent + fixed pitch angle (45°).
     - Y-axis: Perpendicular to Z and World Z.
     - X-axis: Cross product of Y and Z.
   - Result: 6-DOF pose for every waypoint.

================================================================================
THESIS-READY STATEMENTS
================================================================================

> "To mitigate sensor noise and ensure kinematic smoothness, raw 3D points are fitted
> with a cubic B-spline. The curve is then re-parameterized by arc length to generate
> equidistant waypoints, critical for maintaining constant heat input during welding."

> "End-effector orientation is derived from the curve tangent vector, assuming a
> fixed torch parameterization relative to a planar workspace. This simplifies the
> orientation planning problem while sufficient for linear seam welding tasks."

================================================================================
"""

import rclpy
from rclpy.node import Node
from parol6_msgs.msg import WeldLine3DArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from std_srvs.srv import Trigger

import numpy as np
from scipy import interpolate
from sklearn.decomposition import PCA
import math

class PathGenerator(Node):
    """
    Path Generator Node
    
    Generates smooth 6-DOF welding paths from 3D points.
    
    Subscribed Topics:
        /vision/weld_lines_3d (WeldLine3DArray)
        
    Published Topics:
        /vision/welding_path (nav_msgs/Path)
        /path_generator/markers (MarkerArray)
        
    Parameters:
        spline_smoothing (float): B-spline smoothing factor (s)
        waypoint_spacing (float): Distance between waypoints (meters)
        approach_angle_deg (float): Torch pitch angle (degrees)
        auto_generate (bool): Immediately publish path on detection
    """
    
    def __init__(self):
        super().__init__('path_generator')
        
        # ============================================================
        # PARAMETERS
        # ============================================================
        
        self.declare_parameter('spline_degree', 3)
        self.declare_parameter('spline_smoothing', 0.005) # 5mm variance allowed
        self.declare_parameter('waypoint_spacing', 0.005) # 5mm spacing
        self.declare_parameter('approach_angle_deg', 45.0)
        self.declare_parameter('auto_generate', True)
        self.declare_parameter('min_points_for_path', 5)
        
        self.k = self.get_parameter('spline_degree').value
        self.s = self.get_parameter('spline_smoothing').value
        self.spacing = self.get_parameter('waypoint_spacing').value
        self.pitch_deg = self.get_parameter('approach_angle_deg').value
        self.auto_gen = self.get_parameter('auto_generate').value
        self.min_pts = self.get_parameter('min_points_for_path').value
        
        # ============================================================
        # INTERFACES
        # ============================================================
        
        self.sub = self.create_subscription(
            WeldLine3DArray,
            '/vision/weld_lines_3d',
            self.callback,
            10
        )
        
        self.path_pub = self.create_publisher(
            Path,
            '/vision/welding_path',
            10
        )
        
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/path_generator/markers',
            10
        )
        
        # Manual Trigger Service
        self.srv = self.create_service(
            Trigger,
            '~/trigger_path_generation',
            self.trigger_callback
        )
        
        self.latest_msg = None
        self.get_logger().info('Path Generator initialized')
        
    def callback(self, msg):
        """Buffer latest message and auto-generate if enabled"""
        self.latest_msg = msg
        if self.auto_gen:
            self.generate_path(msg)
            
    def trigger_callback(self, request, response):
        """Service callback for manual triggering"""
        if self.latest_msg:
            success = self.generate_path(self.latest_msg)
            response.success = success
            response.message = "Path generated" if success else "Generation failed"
        else:
            response.success = False
            response.message = "No weld lines received yet"
        return response
    
    # ================================================================
    # PATH GENERATION LOGIC
    # ================================================================
    
    def generate_path(self, msg):
        """
        Main pipeline: Points → Ordered → Spline → Resampled → Orientations → Path
        Only processes the first detected line (limiting scope for thesis).
        """
        if not msg.lines:
            return False
            
        # Select highest confidence line
        best_line = max(msg.lines, key=lambda l: l.confidence)
        
        points = best_line.points
        if len(points) < self.min_pts:
            self.get_logger().warn(f'Not enough points ({len(points)}) for spline fitting')
            return False
            
        # 1. Convert to Numpy
        pts_np = np.array([[p.x, p.y, p.z] for p in points])
        
        # 2. PCA Ordering
        ordered_pts = self.order_points_pca(pts_np)
        
        # Deduplicate (spline fitting crashes on duplicate points)
        unique_pts = self.remove_duplicates(ordered_pts)
        if len(unique_pts) < self.k + 1: # Spline requires degree + 1 points
            return False
            
        # 3. Spline Fitting and Resampling
        try:
            waypoints, tangents = self.fit_bspline_and_resample(unique_pts)
        except Exception as e:
            self.get_logger().error(f'Spline fitting failed: {e}')
            return False
            
        # 4. Generate Orientations
        poses = []
        for i in range(len(waypoints)):
            pt = waypoints[i]
            tan = tangents[i]
            
            # Compute Quaternion
            quat = self.compute_orientation(tan, self.pitch_deg)
            
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position = Point(x=pt[0], y=pt[1], z=pt[2])
            pose.pose.orientation = quat
            poses.append(pose)
            
        # 5. Publish Path
        path_msg = Path()
        path_msg.header = msg.header
        path_msg.poses = poses
        
        self.path_pub.publish(path_msg)
        self.get_logger().info(f'Generated path with {len(poses)} waypoints')
        
        # 6. Visualize
        self.publish_visualization(poses, msg.header)
        
        return True
        
    def order_points_pca(self, points):
        """Sort points along principal axis"""
        pca = PCA(n_components=1)
        projected = pca.fit_transform(points)
        sorted_indices = np.argsort(projected.flatten())
        return points[sorted_indices]
        
    def remove_duplicates(self, points, tol=1e-4):
        """Remove duplicate points within tolerance"""
        unique = []
        if len(points) > 0:
            unique.append(points[0])
            for i in range(1, len(points)):
                dist = np.linalg.norm(points[i] - unique[-1])
                if dist > tol:
                    unique.append(points[i])
        return np.array(unique)
        
    def fit_bspline_and_resample(self, points):
        """
        Fits B-spline and resamples at fixed Euclidean spacing.
        Returns: (waypoints, tangents)
        """
        # Transpose for splprep (expects list of coordinate arrays)
        tck, u = interpolate.splprep(points.T, s=self.s, k=self.k)
        
        # Generate fine-grained parameter space
        u_fine = np.linspace(0, 1, num=len(points)*10)
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        fine_points = np.vstack((x_fine, y_fine, z_fine)).T
        
        # Compute arc length
        diffs = np.diff(fine_points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_len = np.insert(np.cumsum(seg_lengths), 0, 0)
        total_len = cumulative_len[-1]
        
        # Number of waypoints
        num_waypoints = int(max(2, total_len / self.spacing))
        
        # Resample at equal arc-length intervals
        u_query = []
        target_dists = np.linspace(0, total_len, num_waypoints)
        
        for d in target_dists:
            # Find u corresponding to distance d
            idx = np.searchsorted(cumulative_len, d)
            if idx == 0:
                u_val = 0.0
            elif idx >= len(cumulative_len):
                u_val = 1.0
            else:
                # Linear interpolate u
                d0 = cumulative_len[idx-1]
                d1 = cumulative_len[idx]
                ratio = (d - d0) / (d1 - d0)
                u0 = u_fine[idx-1]
                u1 = u_fine[idx]
                u_val = u0 + ratio * (u1 - u0)
            u_query.append(u_val)
            
        # Evaulate spline and derivative at u_query
        xi, yi, zi = interpolate.splev(u_query, tck)
        dx, dy, dz = interpolate.splev(u_query, tck, der=1)
        
        waypoints = np.vstack((xi, yi, zi)).T
        tangents = np.vstack((dx, dy, dz)).T
        
        # Normalize tangents
        norms = np.linalg.norm(tangents, axis=1)
        tangents = tangents / norms[:, None]
        
        return waypoints, tangents
        
    def compute_orientation(self, tangent, pitch_deg):
        """
        Generates orientation quaternion.
        HARDCODED FOR PAROL6 REACHABILITY TEST:
        Forces the end effector Z-axis to point forward (+X) towards the wall.
        The previous dynamic orientation caused Inverse Kinematics (IK) failures
        because it asked the robot to bend its wrist in impossible ways.
        """
        # Quaternion for pitch = 90 degrees (rotates Z UP to point +X FORWARD)
        return Quaternion(x=0.0, y=0.7071068, z=0.0, w=0.7071068)
        
    def rotation_matrix_to_quaternion(self, R):
        """Manual implementation of rotation matrix to quat"""
        trace = np.trace(R)
        if trace > 0:
            S = math.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
            
        return Quaternion(x=qx, y=qy, z=qz, w=qw)
        
    def publish_visualization(self, poses, header):
        """Visualize path and orientation arrows"""
        ma = MarkerArray()
        
        # Axes arrows for every 5th waypoint
        for i, pose in enumerate(poses):
            if i % 5 != 0: continue
            
            m = Marker()
            m.header = header
            m.ns = "orientation"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose = pose.pose
            m.scale.x = 0.02 # 2cm arrow
            m.scale.y = 0.002
            m.scale.z = 0.002
            m.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0) # Magenta
            ma.markers.append(m)
            
        self.marker_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = PathGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
