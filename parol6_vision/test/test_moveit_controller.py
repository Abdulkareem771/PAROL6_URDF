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
        self.node.move_to_pose = MagicMock(return_value=True)
        self.node.plan_cartesian_with_fallback = MagicMock(return_value=RobotTrajectory())
        self.node.execute_trajectory_action = MagicMock(return_value=True)
        
        path = Path()
        path.poses = [PoseStamped()]
        
        success = self.node.execute_welding_sequence(path)
        
        self.assertTrue(success)
        # Verify sequence calls
        self.node.move_to_pose.assert_called_once() # Approach
        self.node.plan_cartesian_with_fallback.assert_called_once() # Plan Weld
        self.node.execute_trajectory_action.assert_called_once() # Execute Weld

if __name__ == '__main__':
    unittest.main()
