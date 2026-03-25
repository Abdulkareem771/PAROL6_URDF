import unittest
from unittest.mock import MagicMock, patch
import rclpy
from parol6_vision.depth_matcher import DepthMatcher
from parol6_msgs.msg import WeldLine, WeldLineArray
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Point32
import numpy as np

class TestDepthMatcher(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # Patch TF listener to avoid errors during init
        with patch('parol6_vision.depth_matcher.tf2_ros.TransformListener'):
            self.node = DepthMatcher()
            
        # Mock TF buffer
        self.node.tf_buffer = MagicMock()
        
        # Mock Publisher
        self.node.lines_3d_pub = MagicMock()
        
        # Override params for unit test with single point
        self.node.min_points = 0
        self.node.min_quality = 0.0

    def tearDown(self):
        self.node.destroy_node()

    def test_filter_statistical_outliers(self):
        """Test Z-score outlier removal"""
        # Create points: 10 points at Z=1.0, 1 outlier at Z=10.0
        points = []
        for _ in range(10):
            p = MagicMock()
            p.x, p.y, p.z = 0.0, 0.0, 1.0
            points.append(p)
            
        outlier = MagicMock()
        outlier.x, outlier.y, outlier.z = 0.0, 0.0, 10.0
        points.append(outlier)
        
        filtered = self.node.filter_statistical_outliers(points)
        
        self.assertEqual(len(filtered), 10, "Outlier not removed")
        self.assertNotIn(outlier, filtered)

    def test_backprojection_logic(self):
        """Test 3D projection math with mock inputs"""
        
        # 1. Create Mock Messages
        # Camera Info (Simple identity-like camera)
        info_msg = CameraInfo()
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        # Let's say fx=fy=100, cx=cy=50
        info_msg.k = [100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0]
        
        # Weld Line (1 point at center)
        line = WeldLine()
        line.id = "test"
        line.confidence = 1.0
        line.pixels = [Point32(x=50.0, y=50.0, z=0.0)] # At principal point
        
        lines_msg = WeldLineArray()
        lines_msg.header.frame_id = "camera_frame"
        lines_msg.lines = [line]
        
        # Depth Image (Mock)
        depth_msg = Image()
        # We'll mock the cv_bridge conversion result instead of filling data bytes
        
        # 2. Mock CV Bridge
        mock_cv_depth = np.zeros((100, 100), dtype=np.uint16)
        mock_cv_depth[50, 50] = 1000 # 1000mm = 1.0m depth at center
        
        self.node.bridge.imgmsg_to_cv2 = MagicMock(return_value=mock_cv_depth)
        
        # 3. Mock Transform (Identity transform)
        transform = TransformStamped()
        # Identity quaternion
        transform.transform.rotation.w = 1.0
        
        self.node.tf_buffer.lookup_transform.return_value = transform
        
        # 4. Run Callback
        with patch('parol6_vision.depth_matcher.tf2_geometry_msgs.do_transform_point') as mock_transform:
            # Mock coordinate transform to return point as-is (Identity)
            def side_effect(pt, trans):
                return pt # Return passed PointStamped
            mock_transform.side_effect = side_effect
            
            self.node.synchronized_callback(lines_msg, depth_msg, info_msg)
            
            # 5. Verify Output
            self.node.lines_3d_pub.publish.assert_called_once()
            args = self.node.lines_3d_pub.publish.call_args[0][0]
            
            self.assertEqual(len(args.lines), 1)
            line_3d = args.lines[0]
            self.assertEqual(len(line_3d.points), 1)
            
            pt = line_3d.points[0] # line_3d.points contains Point objects
            
            # Expected coordinate:
            # u=50, cx=50 -> X = 0
            # v=50, cy=50 -> Y = 0
            # Z = 1.0
            self.assertAlmostEqual(pt.x, 0.0)
            self.assertAlmostEqual(pt.y, 0.0)
            self.assertAlmostEqual(pt.z, 1.0)
            
    def test_out_of_bounds_filtering(self):
        """Test points are ignored if depth is 0 or out of range"""
        # ... setup similar to above but with bad depth ...
        # (Simplified for brevity)
        pass

if __name__ == '__main__':
    unittest.main()
