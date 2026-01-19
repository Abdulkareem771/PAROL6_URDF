import unittest
from unittest.mock import MagicMock, patch
import rclpy
from parol6_vision.path_generator import PathGenerator
from parol6_msgs.msg import WeldLine3DArray, WeldLine3D
from geometry_msgs.msg import Point
import numpy as np
import math

class TestPathGenerator(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = PathGenerator()
        # Mock publishers to verify output
        self.node.path_pub = MagicMock()
        self.node.marker_pub = MagicMock()

    def tearDown(self):
        self.node.destroy_node()

    def test_order_points_pca(self):
        """Test if PCA correctly orders shuffled points along a line"""
        # Create points along X axis: 0, 1, 2, 3, 4
        true_points = np.array([
            [0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0]
        ], dtype=float)
        
        # Shuffle them
        shuffled = true_points.copy()
        np.random.shuffle(shuffled)
        
        ordered = self.node.order_points_pca(shuffled)
        
        # Check if start and end match extrema (0 or 4)
        start = ordered[0]
        end = ordered[-1]
        
        # PCA direction sign is arbitrary, so order could be 0->4 or 4->0
        is_fwd = np.allclose(start, [0,0,0]) and np.allclose(end, [4,0,0])
        is_rev = np.allclose(start, [4,0,0]) and np.allclose(end, [0,0,0])
        
        self.assertTrue(is_fwd or is_rev, f"Points not ordered correctly. Start={start}, End={end}")

    def test_fit_bspline_and_resample(self):
        """Test spline fitting reduces noise and resamples correctly"""
        # Create a noisy line
        x = np.linspace(0, 1.0, 50)
        y = np.random.normal(0, 0.001, 50)  # Small noise
        z = np.zeros(50)
        
        points = np.vstack((x, y, z)).T
        
        waypoints, tangents = self.node.fit_bspline_and_resample(points)
        
        # Check resampling count
        # Total length approx 1.0m. Spacing default 0.005m (5mm).
        # Expected ~200 points
        expected_count = int(1.0 / 0.005)
        self.assertAlmostEqual(len(waypoints), expected_count, delta=5)
        
        # Check smoothness: mean Y should be close to 0
        mean_y = np.mean(np.abs(waypoints[:, 1]))
        self.assertLess(mean_y, 0.002, "Spline should smooth out noise")
        
        # Check tangents: For a horizontal line, tangent should be approx (1,0,0) or (-1,0,0)
        t_x = np.abs(tangents[:, 0])
        self.assertTrue(np.all(t_x > 0.9), "Tangents should be aligned with X axis")

    def test_compute_orientation(self):
        """Test orientation generation logic"""
        # Case 1: Tangent along X axis (1, 0, 0)
        tangent = np.array([1.0, 0.0, 0.0])
        pitch_deg = 45.0
        
        quat = self.node.compute_orientation(tangent, pitch_deg)
        
        # If Tangent=X, Down=-Z.
        # Check if quaternion represents expected rotation.
        # This is a bit complex math-wise to assert exactly without a reference implementation.
        # But we can check norm is 1.
        norm = math.sqrt(quat.x**2 + quat.y**2 + quat.z**2 + quat.w**2)
        self.assertAlmostEqual(norm, 1.0)
        
        # Case 2: Verify consistency (same input -> same output)
        quat2 = self.node.compute_orientation(tangent, pitch_deg)
        self.assertEqual(quat.w, quat2.w)
        
    def test_generate_path_end_to_end(self):
        """Test full pipeline with mock message"""
        msg = WeldLine3DArray()
        msg.header.frame_id = "base_link"
        
        line = WeldLine3D()
        line.confidence = 1.0
        # Create valid, orderable points
        for i in range(10):
            p = Point()
            p.x, p.y, p.z = float(i)*0.01, 0.0, 0.0
            line.points.append(p)
            
        msg.lines = [line]
        
        success = self.node.generate_path(msg)
        
        self.assertTrue(success)
        self.node.path_pub.publish.assert_called_once()
        
        # Verify published path
        path_arg = self.node.path_pub.publish.call_args[0][0]
        self.assertGreater(len(path_arg.poses), 0)

if __name__ == '__main__':
    unittest.main()
