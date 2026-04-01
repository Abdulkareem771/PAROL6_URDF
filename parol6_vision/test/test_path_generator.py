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
        
        # Check resampling count.
        # Total length approx 1.0m. Spacing default 0.005m (5mm) → ~200 pts,
        # BUT max_waypoints=80 cap is applied for OMPL stability, so expect 80.
        max_wp = self.node.get_parameter('max_waypoints').value
        raw_expected = int(1.0 / 0.005)
        expected_count = min(raw_expected, max_wp)
        self.assertAlmostEqual(len(waypoints), expected_count, delta=5)
        
        # Check smoothness: mean Y should be close to 0
        mean_y = np.mean(np.abs(waypoints[:, 1]))
        self.assertLess(mean_y, 0.002, "Spline should smooth out noise")
        
        # Check tangents: For a horizontal line, tangent should be approx (1,0,0) or (-1,0,0)
        t_x = np.abs(tangents[:, 0])
        self.assertTrue(np.all(t_x > 0.9), "Tangents should be aligned with X axis")

    def test_compute_orientation(self):
        """Tangent-based orientation: different tangents must produce different quaternions"""
        tangent_x = np.array([1.0, 0.0, 0.0])
        tangent_y = np.array([0.0, 1.0, 0.0])
        pitch_deg = 45.0

        quat_x = self.node.compute_orientation(tangent_x, pitch_deg)
        quat_y = self.node.compute_orientation(tangent_y, pitch_deg)

        # Both must be unit quaternions
        for q in (quat_x, quat_y):
            norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
            self.assertAlmostEqual(norm, 1.0, places=5,
                msg="Quaternion must be unit length")

        # Orientations must differ for different tangent directions
        same = (
            math.isclose(quat_x.x, quat_y.x, abs_tol=1e-4) and
            math.isclose(quat_x.y, quat_y.y, abs_tol=1e-4) and
            math.isclose(quat_x.z, quat_y.z, abs_tol=1e-4) and
            math.isclose(quat_x.w, quat_y.w, abs_tol=1e-4)
        )
        self.assertFalse(same,
            "Quaternions for +X and +Y tangents must differ (tangent-aware orientation)")

        # Consistency: same input → same output
        quat_x2 = self.node.compute_orientation(tangent_x, pitch_deg)
        self.assertAlmostEqual(quat_x.w, quat_x2.w, places=5)
        
    def test_generate_path_end_to_end(self):
        """Test full pipeline with mock message"""
        msg = WeldLine3DArray()
        msg.header.frame_id = "kinect2_rgb_optical_frame"  # camera frame
        
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

    def test_path_frame_id_is_base_link(self):
        """Published path frame_id must be 'base_link', not the camera optical frame (Bug 3 guard)"""
        msg = WeldLine3DArray()
        msg.header.frame_id = 'kinect2_rgb_optical_frame'

        line = WeldLine3D()
        line.confidence = 1.0
        for i in range(10):
            p = Point()
            p.x, p.y, p.z = float(i) * 0.01, 0.0, 0.0
            line.points.append(p)
        msg.lines = [line]

        self.node.generate_path(msg)
        path_arg = self.node.path_pub.publish.call_args[0][0]
        self.assertEqual(path_arg.header.frame_id, 'base_link',
            "Path must be published in 'base_link' frame for MoveIt Cartesian planning")

if __name__ == '__main__':
    unittest.main()
