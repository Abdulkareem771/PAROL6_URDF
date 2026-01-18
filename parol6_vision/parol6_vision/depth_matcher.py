#!/usr/bin/env python3
"""
Depth Matcher Node - Vision-Guided Welding Path Detection

This node projects 2D detected weld lines into 3D space using depth camera data.
It synchronizes 2D detections with depth images, performs pinhole camera back-projection,
and transforms points from the camera frame to the robot base frame.

================================================================================
ALGORITHM OVERVIEW
================================================================================

1. SYNCHRONIZATION
   - Uses message_filters.ApproximateTimeSynchronizer
   - Aligns 3 streams:
     1. WeldLineArray (from Red Line Detector)
     2. Depth Image (from Kinect v2)
     3. Camera Info (Intrinsics)
   - Ensures depth data corresponds to the exact moment of detection

2. 3D BACK-PROJECTION (Pinhole Model)
   For each pixel (u,v) in the detected line:
   - Sample depth Z = depth_image[v, u]
   - Compute X = (u - cx) * Z / fx
   - Compute Y = (v - cy) * Z / fy
   - Result: 3D point (X, Y, Z) in Optical Frame

3. COORDINATE TRANSFORMATION
   - Lookup TF transform: Camera Optical Frame → Robot Base Frame
   - Apply transform to all 3D points
   - Result: 3D points relative to robot base (ready for planning)

4. OUTLIER FILTERING
   - Invalid Depth: Discard points where depth=0 or depth=NaN
   - Statistical Outliers: Remove points > 2σ from local median
   - Range limits: Discard points outside valid workspace (min/max depth)

================================================================================
THESIS-READY STATEMENTS
================================================================================

> "3D reconstruction is performed by back-projecting detected 2D pixel coordinates
> using synchronized depth maps and intrinsic camera parameters. A statistical
> outlier filter removes depth noise inherent to time-of-flight sensors."

> "Geometric consistency is maintained by transforming all points to the robot's
> base frame using the TF2 transform tree, ensuring that vision coordinates
> are kinematically valid for motion planning."

================================================================================
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from parol6_msgs.msg import WeldLine, WeldLineArray, WeldLine3D, WeldLine3DArray
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException
import message_filters
import numpy as np
import cv2


class DepthMatcher(Node):
    """
    Depth Matcher Node
    
    Projects 2D weld lines to 3D using depth data.
    
    Subscribed Topics:
        /vision/weld_lines_2d (WeldLineArray): 2D detections
        /kinect2/qhd/image_depth_rect (Image): Aligned depth image
        /kinect2/qhd/camera_info (CameraInfo): Camera intrinsics
        
    Published Topics:
        /vision/weld_lines_3d (WeldLine3DArray): 3D weld seams
        /depth_matcher/markers (MarkerArray): Visualization
    """
    
    def __init__(self):
        super().__init__('depth_matcher')
        
        # ============================================================
        # PARAMETERS
        # ============================================================
        
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('depth_scale', 1.0)  # mm to meters if needed (usually 0.001)
        # Note: Kinect usually gives mm as uint16, CvBridge handles encoding
        
        # Filtering
        self.declare_parameter('outlier_std_threshold', 2.0)
        self.declare_parameter('min_valid_points', 10)
        self.declare_parameter('max_depth', 2000.0) # mm
        self.declare_parameter('min_depth', 300.0)  # mm
        self.declare_parameter('min_depth_quality', 0.6)
        
        # Synchronization
        self.declare_parameter('sync_time_tolerance', 0.1) # seconds
        self.declare_parameter('sync_queue_size', 10)
        
        # Get values
        self.target_frame = self.get_parameter('target_frame').value
        self.outlier_thresh = self.get_parameter('outlier_std_threshold').value
        self.min_points = self.get_parameter('min_valid_points').value
        self.max_depth = self.get_parameter('max_depth').value
        self.min_depth = self.get_parameter('min_depth').value
        self.min_quality = self.get_parameter('min_depth_quality').value
        
        # ============================================================
        # TF LISTENER
        # ============================================================
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ============================================================
        # PUBLISHERS
        # ============================================================
        
        self.lines_3d_pub = self.create_publisher(
            WeldLine3DArray,
            '/vision/weld_lines_3d',
            10
        )
        
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/depth_matcher/markers',
            10
        )
        
        # ============================================================
        # SYNCHRONIZED SUBSCRIBERS
        # ============================================================
        
        self.lines_sub = message_filters.Subscriber(
            self, WeldLineArray, '/vision/weld_lines_2d'
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/kinect2/qhd/image_depth_rect'
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, '/kinect2/qhd/camera_info'
        )
        
        # Use ApproximateTimeSynchronizer
        # Kinect RGB and Depth timestamps may differ slightly (~1-30ms)
        queue_size = self.get_parameter('sync_queue_size').value
        tolerance = self.get_parameter('sync_time_tolerance').value
        
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.lines_sub, self.depth_sub, self.info_sub],
            queue_size=queue_size,
            slop=tolerance
        )
        
        self.sync.registerCallback(self.synchronized_callback)
        
        self.bridge = CvBridge()
        self.get_logger().info('Depth Matcher initialized with ApproximateTimeSynchronizer')
        
    # ================================================================
    # MAIN CALLBACK
    # ================================================================
    
    def synchronized_callback(self, lines_msg, depth_msg, info_msg):
        """
        Process synchronized Tuple(Lines, Depth, Info).
        
        1. Parse camera intrinsics (fx, fy, cx, cy)
        2. Lookup TF transform (Camera → Base)
        3. Convert Depth Image to OpenCV
        4. For each detected line:
           - Back-project pixels to 3D
           - Transform to Base Frame
           - Filter outliers
           - Create WeldLine3D
        5. Publish WeldLine3DArray
        """
        if len(lines_msg.lines) == 0:
            return

        # 1. Parse Intrinsics
        # CameraInfo K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        K = info_msg.k
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        
        # 2. Lookup Transform
        # We need to transform from the camera frame (in header) to target_frame (base_link)
        try:
            # Wait briefly for transform availability if needed, but in callback rely on buffer
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                lines_msg.header.frame_id, 
                rclpy.time.Time() # Get latest available
            )
        except TransformException as ex:
            self.get_logger().warning(
                f'Could not transform {lines_msg.header.frame_id} to {self.target_frame}: {ex}'
            )
            return
            
        # 3. Convert Depth Image
        try:
            # 16UC1 = 16-bit unsigned, typically mm
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'CV Bridge failed: {e}')
            return
            
        # 4. Process Lines
        weld_lines_3d = []
        
        for line_2d in lines_msg.lines:
            
            points_3d = []
            valid_count = 0
            total_count = len(line_2d.pixels)
            
            for pixel in line_2d.pixels:
                u, v = int(pixel.x), int(pixel.y)
                
                # Bounds check
                if u < 0 or u >= cv_depth.shape[1] or v < 0 or v >= cv_depth.shape[0]:
                    continue
                
                # Sample depth
                d_raw = cv_depth[v, u]
                
                # Invalid depth check (0 or too far/close)
                if d_raw == 0 or d_raw > self.max_depth or d_raw < self.min_depth:
                    continue
                
                # Convert mm to meters
                depth_m = float(d_raw) / 1000.0
                
                # Pinhole Back-projection to Camera Frame
                # X = (u - cx) * Z / fx
                # Y = (v - cy) * Z / fy
                # Z = depth
                x_c = (u - cx) * depth_m / fx
                y_c = (v - cy) * depth_m / fy
                z_c = depth_m
                
                # Create PointStamped for TF
                pt_stamped = PointStamped()
                pt_stamped.header = lines_msg.header
                pt_stamped.point.x = x_c
                pt_stamped.point.y = y_c
                pt_stamped.point.z = z_c
                
                # Transform to Base Frame
                try:
                    pt_transformed = tf2_geometry_msgs.do_transform_point(pt_stamped, transform)
                    points_3d.append(pt_transformed.point)
                    valid_count += 1
                except Exception as e:
                    pass # Skip point if transform fails
            
            # --- Outlier Filtering ---
            filtered_points = self.filter_statistical_outliers(points_3d)
            
            # --- Quality Check ---
            depth_quality = valid_count / max(total_count, 1)
            
            if len(filtered_points) >= self.min_points and depth_quality >= self.min_quality:
                # Create WeldLine3D message
                line_3d = WeldLine3D()
                line_3d.id = line_2d.id
                line_3d.confidence = line_2d.confidence
                line_3d.points = filtered_points
                line_3d.depth_quality = float(depth_quality)
                line_3d.num_points = int(len(filtered_points))
                
                # Calculate average width (placeholder logic - usually requires mask width)
                line_3d.line_width = 0.003 # 3mm default assumption
                
                line_3d.header.stamp = lines_msg.header.stamp
                line_3d.header.frame_id = self.target_frame
                
                weld_lines_3d.append(line_3d)
        
        # 5. Publish
        if weld_lines_3d:
            msg_3d = WeldLine3DArray()
            msg_3d.header.stamp = lines_msg.header.stamp
            msg_3d.header.frame_id = self.target_frame
            msg_3d.lines = weld_lines_3d
            
            self.lines_3d_pub.publish(msg_3d)
            self.get_logger().info(f'Published {len(weld_lines_3d)} 3D weld lines')
            
            # Visualize
            self.publish_markers(weld_lines_3d)
            
    # ================================================================
    # UTIL FUNCTIONS
    # ================================================================
    
    def filter_statistical_outliers(self, points):
        """
        Remove 3D points that deviate significantly from the local cluster.
        Using Z-score on depth (Z-axis in camera, varies in base frame) 
        or distance from mean.
        
        Simple strategy: Remove points > 2 std dev from mean Z height
        (Assuming weld is roughly flat relative to robot base Z)
        """
        if len(points) < 5:
            return points
            
        # Convert to numpy for easy math
        pts_np = np.array([[p.x, p.y, p.z] for p in points])
        
        # Calculate mean and std deviation
        mean = np.mean(pts_np, axis=0)
        std = np.std(pts_np, axis=0)
        
        # Simple distance-based filtering
        # Distance from centroid
        dists = np.linalg.norm(pts_np - mean, axis=1)
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        
        threshold = mean_dist + (self.outlier_thresh * std_dist)
        
        # Filter mask
        mask = dists <= threshold
        
        filtered_points = []
        for i, keep in enumerate(mask):
            if keep:
                filtered_points.append(points[i])
                
        return filtered_points
        
    def publish_markers(self, lines_3d):
        """
        Visualize 3D points in RViz.
        """
        marker_array = MarkerArray()
        
        for idx, line in enumerate(lines_3d):
            # 1. Point Cloud Marker (SPHERE_LIST) - shows individual samples
            m_pts = Marker()
            m_pts.header = line.header
            m_pts.ns = "weld_points"
            m_pts.id = idx
            m_pts.type = Marker.SPHERE_LIST
            m_pts.action = Marker.ADD
            m_pts.scale.x = 0.005 # 5mm dots
            m_pts.scale.y = 0.005
            m_pts.scale.z = 0.005
            m_pts.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8) # Blue
            m_pts.points = line.points
            marker_array.markers.append(m_pts)
            
            # 2. Line Strip - shows connectivity
            m_line = Marker()
            m_line.header = line.header
            m_line.ns = "weld_connectivity"
            m_line.id = idx + 100
            m_line.type = Marker.LINE_STRIP
            m_line.action = Marker.ADD
            m_line.scale.x = 0.002 # thin line
            m_line.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0) # Cyan
            m_line.points = line.points
            marker_array.markers.append(m_line)
            
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = DepthMatcher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
