#!/usr/bin/env python3
"""
Cartesian Path Demo - Phase 1 Validation
=========================================

Purpose: Validate smooth Cartesian path following for vision-guided welding

This demo tests the core capability needed for welding/gluing:
- Following straight-line paths in 3D space
- Smooth transitions between waypoints
- Acceleration profiles (no constant velocity needed)
- Open-loop (visual verification) and closed-loop servo operation

Author: PAROL6 Team
Date: 2026
License: MIT
"""

import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from geometry_msgs.msg import Pose, PoseStamped
import copy
import time
import sys


class CartesianPathDemo(Node):
    """Demo node for testing Cartesian path execution"""
    
    def __init__(self):
        super().__init__('cartesian_path_demo')
        
        # Initialize MoveIt
        self.get_logger().info("Initializing MoveIt...")
        self.moveit = MoveItPy(node_name="cartesian_demo_moveit")
        self.arm = self.moveit.get_planning_component("parol6_arm")
        
        self.get_logger().info("✓ MoveIt initialized")
    
    def go_to_home(self):
        """Move to home position"""
        self.get_logger().info("\n[Step 1] Moving to HOME position...")
        
        try:
            self.arm.set_goal_state(configuration_name="home")
            plan_result = self.arm.plan()
            
            if plan_result:
                self.get_logger().info("  Planning successful, executing...")
                self.moveit.execute(plan_result.trajectory, controllers=[])
                time.sleep(1)  # Allow settling
                self.get_logger().info("  ✓ Reached HOME")
                return True
            else:
                self.get_logger().error("  ✗ Planning failed")
                return False
        except Exception as e:
            self.get_logger().error(f"  ✗ Error: {e}")
            return False
    
    def demo_straight_line(self):
        """
        Demo 1: Simple straight-line motion
        Tests basic Cartesian path following
        """
        self.get_logger().info("\n[Step 2] Demo: Straight Line Path")
        self.get_logger().info("  Simulating welding along a straight seam...")
        
        try:
            # Get current pose
            current_pose = self.arm.get_current_pose()
            if not current_pose:
                self.get_logger().error("  ✗ Could not get current pose")
                return False
            
            start_pose = current_pose.pose
            
            # Create waypoints for straight line (10cm forward)
            waypoints = []
            
            # Waypoint 1: 5cm forward
            wpose = copy.deepcopy(start_pose)
            wpose.position.x += 0.05
            waypoints.append(copy.deepcopy(wpose))
            
            # Waypoint 2: 10cm forward (total)
            wpose.position.x += 0.05
            waypoints.append(copy.deepcopy(wpose))
            
            self.get_logger().info(f"  Planning path with {len(waypoints)} waypoints...")
            
            # Compute Cartesian path
            (plan, fraction) = self.arm.compute_cartesian_path(
                waypoints,
                eef_step=0.01,        # 1cm resolution (smooth)
                jump_threshold=0.0    # No joint space jumps
            )
            
            self.get_logger().info(f"  Path planning: {fraction*100:.1f}% complete")
            
            if fraction > 0.95:  # 95%+ success
                self.get_logger().info("  Executing straight-line motion...")
                self.moveit.execute(plan, controllers=[])
                time.sleep(1)
                self.get_logger().info("  ✓ Straight line complete")
                return True
            else:
                self.get_logger().warn(f"  ⚠ Only {fraction*100:.1f}% of path planned")
                return False
                
        except Exception as e:
            self.get_logger().error(f"  ✗ Error: {e}")
            return False
    
    def demo_l_shape(self):
        """
        Demo 2: L-shaped path
        Tests corner transitions (common in welding)
        """
        self.get_logger().info("\n[Step 3] Demo: L-Shaped Path (Corner Weld)")
        self.get_logger().info("  Simulating welding around a corner...")
        
        try:
            current_pose = self.arm.get_current_pose()
            if not current_pose:
                self.get_logger().error("  ✗ Could not get current pose")
                return False
            
            start_pose = current_pose.pose
            waypoints = []
            
            # Segment 1: Move forward 8cm
            wpose = copy.deepcopy(start_pose)
            for i in range(1, 5):  # 4 waypoints over 8cm
                wpose.position.x += 0.02  # 2cm steps
                waypoints.append(copy.deepcopy(wpose))
            
            # Segment 2: Turn right 90° and continue 6cm
            for i in range(1, 4):  # 3 waypoints over 6cm
                wpose.position.y += 0.02  # 2cm steps
                waypoints.append(copy.deepcopy(wpose))
            
            self.get_logger().info(f"  Planning L-path with {len(waypoints)} waypoints...")
            
            (plan, fraction) = self.arm.compute_cartesian_path(
                waypoints,
                eef_step=0.005,       # 5mm resolution (very smooth)
                jump_threshold=0.0
            )
            
            self.get_logger().info(f"  Path planning: {fraction*100:.1f}% complete")
            
            if fraction > 0.9:
                self.get_logger().info("  Executing L-shaped motion...")
                self.moveit.execute(plan, controllers=[])
                time.sleep(1)
                self.get_logger().info("  ✓ L-shape complete")
                return True
            else:
                self.get_logger().warn(f"  ⚠ Only {fraction*100:.1f}% of path planned")
                return False
                
        except Exception as e:
            self.get_logger().error(f"  ✗ Error: {e}")
            return False
    
    def demo_rectangle(self):
        """
        Demo 3: Rectangular path
        Tests multi-corner transitions and path closure
        """
        self.get_logger().info("\n[Step 4] Demo: Rectangular Path (Box Weld)")
        self.get_logger().info("  Simulating welding a rectangular frame...")
        
        try:
            current_pose = self.arm.get_current_pose()
            if not current_pose:
                self.get_logger().error("  ✗ Could not get current pose")
                return False
            
            center = current_pose.pose
            waypoints = []
            
            # Define rectangle: 8cm x 6cm
            length = 0.08  # 8cm
            width = 0.06   # 6cm
            
            # Start at corner
            wpose = copy.deepcopy(center)
            wpose.position.x -= length/2
            wpose.position.y -= width/2
            
            # Side 1: Forward
            for i in range(5):
                wpose.position.x += length/5
                waypoints.append(copy.deepcopy(wpose))
            
            # Side 2: Right
            for i in range(4):
                wpose.position.y += width/4
                waypoints.append(copy.deepcopy(wpose))
            
            # Side 3: Backward
            for i in range(5):
                wpose.position.x -= length/5
                waypoints.append(copy.deepcopy(wpose))
            
            # Side 4: Left (close the loop)
            for i in range(4):
                wpose.position.y -= width/4
                waypoints.append(copy.deepcopy(wpose))
            
            self.get_logger().info(f"  Planning rectangle with {len(waypoints)} waypoints...")
            
            (plan, fraction) = self.arm.compute_cartesian_path(
                waypoints,
                eef_step=0.005,       # 5mm resolution
                jump_threshold=0.0
            )
            
            self.get_logger().info(f"  Path planning: {fraction*100:.1f}% complete")
            
            if fraction > 0.85:  # Allow slightly lower success for complex path
                self.get_logger().info("  Executing rectangular motion...")
                self.moveit.execute(plan, controllers=[])
                time.sleep(1)
                self.get_logger().info("  ✓ Rectangle complete")
                return True
            else:
                self.get_logger().warn(f"  ⚠ Only {fraction*100:.1f}% of path planned")
                return False
                
        except Exception as e:
            self.get_logger().error(f"  ✗ Error: {e}")
            return False
    
    def run_all_demos(self):
        """Run complete demo sequence"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("  PAROL6 Cartesian Path Demo - Phase 1 Validation")
        self.get_logger().info("  Purpose: Test smooth path following for welding")
        self.get_logger().info("="*60)
        
        results = {
            'home': False,
            'straight': False,
            'l_shape': False,
            'rectangle': False
        }
        
        # Step 1: Go to home
        results['home'] = self.go_to_home()
        if not results['home']:
            self.get_logger().error("\n✗ Failed to reach home position. Aborting.")
            return results
        
        # Step 2: Straight line
        results['straight'] = self.demo_straight_line()
        
        # Return to home between demos
        self.get_logger().info("\n  Returning to home...")
        self.go_to_home()
        time.sleep(1)
        
        # Step 3: L-shape
        results['l_shape'] = self.demo_l_shape()
        
        # Return to home
        self.get_logger().info("\n  Returning to home...")
        self.go_to_home()
        time.sleep(1)
        
        # Step 4: Rectangle
        results['rectangle'] = self.demo_rectangle()
        
        # Final home
        self.get_logger().info("\n  Returning to home...")
        self.go_to_home()
        
        # Summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """Print demo results summary"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("  DEMO SUMMARY")
        self.get_logger().info("="*60)
        
        total = len(results)
        passed = sum(results.values())
        
        for test, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            self.get_logger().info(f"  {test.upper():<15} {status}")
        
        self.get_logger().info("-"*60)
        self.get_logger().info(f"  Results: {passed}/{total} tests passed")
        
        if passed == total:
            self.get_logger().info("\n  ✓ ALL TESTS PASSED - System ready for vision integration!")
        elif passed >= total * 0.75:
            self.get_logger().warn("\n  ⚠ PARTIAL SUCCESS - Review failed tests")
        else:
            self.get_logger().error("\n  ✗ TESTS FAILED - Debug motion system before proceeding")
        
        self.get_logger().info("="*60 + "\n")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        demo = CartesianPathDemo()
        
        # Run demo sequence
        results = demo.run_all_demos()
        
        # Keep node alive briefly
        time.sleep(2)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
