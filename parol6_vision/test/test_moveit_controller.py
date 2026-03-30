import unittest
from unittest.mock import MagicMock, patch
from asyncio import Future
import rclpy
from parol6_vision.moveit_controller import MoveItController
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory

class TestMoveItController(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # Patch Action Clients and Service Clients to prevent real ROS calls
        with patch('parol6_vision.moveit_controller.ActionClient'), \
             patch('parol6_vision.moveit_controller.Node.create_client'):
            self.node = MoveItController()
            
        # Mock the specific clients
        self.node.cartesian_client = MagicMock()
        self.node.execute_client = MagicMock()
        self.node.move_group_client = MagicMock()

    def tearDown(self):
        self.node.destroy_node()

    def create_mock_future(self, fraction, error_code=MoveItErrorCodes.SUCCESS):
        """Helper to create a Future for service call"""
        f = Future()
        result = MagicMock()
        result.error_code.val = error_code
        result.fraction = fraction
        result.solution = RobotTrajectory() # Mock empty trajectory
        f.set_result(result)
        return f

    def test_plan_cartesian_success_first_try(self):
        """Test planning succeeds on first attempt (2mm)"""
        # Mock service to return 1.0 (100%) fraction immediately
        self.node.cartesian_client.wait_for_service.return_value = True
        self.node.cartesian_client.call_async.return_value = self.create_mock_future(1.0)
        
        path = Path()
        path.poses = [PoseStamped()] # Mock path
        
        traj = self.node.plan_cartesian_with_fallback(path)
        
        self.assertIsNotNone(traj)
        # Should verify it called with step=0.002
        args = self.node.cartesian_client.call_async.call_args[0][0]
        self.assertAlmostEqual(args.max_step, 0.002)

    def test_plan_cartesian_fallback(self):
        """Test planning fails first two tries, succeeds on third (10mm)"""
        self.node.cartesian_client.wait_for_service.return_value = True
        
        # Setup side effects for consecutive calls
        # 1. Step=0.002 -> Fraction=0.5 (Fail)
        # 2. Step=0.005 -> Fraction=0.8 (Fail)
        # 3. Step=0.010 -> Fraction=0.95 (Success)
        
        f1 = self.create_mock_future(0.5)
        f2 = self.create_mock_future(0.8)
        f3 = self.create_mock_future(0.95)
        
        self.node.cartesian_client.call_async.side_effect = [f1, f2, f3]
        
        path = Path()
        path.poses = [PoseStamped()]
        
        traj = self.node.plan_cartesian_with_fallback(path)
        
        self.assertIsNotNone(traj)
        self.assertEqual(self.node.cartesian_client.call_async.call_count, 3)
        
        # Verify last call was with 0.010 step
        last_call_args = self.node.cartesian_client.call_async.call_args_list[2][0][0]
        self.assertAlmostEqual(last_call_args.max_step, 0.010)

    def test_plan_cartesian_total_failure(self):
        """Test planning fails all attempts"""
        self.node.cartesian_client.wait_for_service.return_value = True
        
        # All fail
        f_fail = self.create_mock_future(0.1)
        self.node.cartesian_client.call_async.return_value = f_fail
        
        path = Path()
        path.poses = [PoseStamped()]
        
        traj = self.node.plan_cartesian_with_fallback(path)
        
        self.assertIsNone(traj)
        # Should have tried all 3 steps
        self.assertEqual(self.node.cartesian_client.call_async.call_count, 3)

    def test_execute_welding_sequence(self):
        """Test full execution sequence logic"""
        # Mock helpers
        self.node.plan_to_home = MagicMock(return_value=RobotTrajectory())
        self.node.plan_approach_with_fallback = MagicMock(return_value=RobotTrajectory())
        self.node.plan_cartesian_with_fallback = MagicMock(return_value=RobotTrajectory())
        self.node.execute_trajectory_action = MagicMock(return_value=True)
        
        path = Path()
        path.poses = [PoseStamped()]
        
        success = self.node.execute_welding_sequence(path)
        
        self.assertTrue(success)
        # Verify sequence calls
        self.node.plan_to_home.assert_called_once()
        self.node.plan_approach_with_fallback.assert_called_once()
        self.node.plan_cartesian_with_fallback.assert_called_once() # Plan Weld
        self.assertEqual(self.node.execute_trajectory_action.call_count, 3)

    def test_plan_approach_with_fallback_relaxes_constraints(self):
        """Approach planning should fall back to position-only before giving up."""
        first_pose = PoseStamped()
        first_pose.pose.position.x = 0.46
        first_pose.pose.position.y = 0.05
        first_pose.pose.position.z = 0.23
        first_pose.pose.orientation.x = 0.707
        first_pose.pose.orientation.z = -0.707

        self.node.approach_dist = 0.15
        self.node.plan_pose = MagicMock(side_effect=[None, None, RobotTrajectory()])

        traj = self.node.plan_approach_with_fallback(first_pose, start_state_traj=RobotTrajectory())

        self.assertIsNotNone(traj)
        self.assertEqual(self.node.plan_pose.call_count, 3)
        third_call = self.node.plan_pose.call_args_list[2]
        self.assertFalse(third_call.kwargs['use_orientation_constraint'])

    def test_trigger_execution_uses_snapshot_copy(self):
        """Service-triggered execution should pass a copy of the cached path."""
        path = Path()
        pose = PoseStamped()
        pose.pose.position.x = 0.1
        path.poses = [pose]
        self.node.latest_path = path
        self.node._start_execution_async = MagicMock()

        response = type('Resp', (), {})()
        result = self.node.trigger_execution(None, response)

        self.assertTrue(result.success)
        started_path = self.node._start_execution_async.call_args.args[0]
        self.assertIsNot(started_path, path)
        self.assertEqual(len(started_path.poses), 1)

    def test_make_path_reachable_clamps_to_workspace(self):
        """Test that points outside workspace bounds are clamped into bounds."""
        self.node.enforce_reachable_test_path = True
        self.node.workspace_min = [0.20, -0.35, 0.10]
        self.node.workspace_max = [0.65,  0.35, 0.55]
        self.node.min_radius_xy = 0.20
        self.node.max_radius_xy = 0.70

        path = Path()
        # Point with x far outside max (1.5 > 0.65)
        ps_far = PoseStamped()
        ps_far.pose.position.x = 1.5
        ps_far.pose.position.y = 0.0
        ps_far.pose.position.z = 0.30
        ps_far.pose.orientation.w = 1.0
        # Point already inside bounds
        ps_in = PoseStamped()
        ps_in.pose.position.x = 0.40
        ps_in.pose.position.y = 0.00
        ps_in.pose.position.z = 0.30
        ps_in.pose.orientation.w = 1.0
        path.poses = [ps_far, ps_in]

        result = self.node._make_path_reachable(path)

        self.assertEqual(len(result.poses), 2)
        # Far point clamped to within workspace max x
        self.assertLessEqual(result.poses[0].pose.position.x, 0.65)
        # Far point also clamped by radial max (radius = x since y=0)
        self.assertLessEqual(result.poses[0].pose.position.x, 0.70)
        # In-bounds point unchanged
        self.assertAlmostEqual(result.poses[1].pose.position.x, 0.40)

    def test_make_path_reachable_enforces_min_radius(self):
        """Test that points too close to origin are pushed out to min_radius."""
        self.node.enforce_reachable_test_path = True
        self.node.workspace_min = [0.20, -0.35, 0.10]
        self.node.workspace_max = [0.65,  0.35, 0.55]
        self.node.min_radius_xy = 0.20
        self.node.max_radius_xy = 0.70

        path = Path()
        ps = PoseStamped()
        ps.pose.position.x = 0.05   # inside min_radius
        ps.pose.position.y = 0.0
        ps.pose.position.z = 0.30
        ps.pose.orientation.w = 1.0
        path.poses = [ps]

        result = self.node._make_path_reachable(path)
        r = (result.poses[0].pose.position.x ** 2 + result.poses[0].pose.position.y ** 2) ** 0.5
        self.assertGreaterEqual(r, 0.20 - 1e-6)

if __name__ == '__main__':
    unittest.main()
