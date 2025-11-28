#!/usr/bin/env python3
"""
PAROL6 MoveIt Python Example
Demonstrates basic motion planning using the MoveIt Python API
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory
from geometry_msgs.msg import Pose
import sys

# Note: This requires moveit_py or moveit_commander
# Install with: sudo apt-get install ros-humble-moveit-py

try:
    from moveit.planning import MoveItPy
    from moveit.core.robot_state import RobotState
except ImportError:
    print("ERROR: MoveIt Python bindings not found!")
    print("Install with: sudo apt-get install ros-humble-moveit-py")
    sys.exit(1)


class PAROL6Controller(Node):
    """Simple controller for PAROL6 robot using MoveIt"""
    
    def __init__(self):
        super().__init__('parol6_controller')
        
        # Initialize MoveIt
        self.get_logger().info("Initializing MoveIt...")
        self.moveit = MoveItPy(node_name="parol6_moveit_py")
        
        # Get the planning component
        self.parol6_arm = self.moveit.get_planning_component("parol6_arm")
        
        self.get_logger().info("PAROL6 Controller initialized!")
        
    def go_to_named_state(self, state_name):
        """Move to a predefined named state (e.g., 'home', 'ready')"""
        self.get_logger().info(f"Planning to named state: {state_name}")
        
        # Set the goal to a named state
        self.parol6_arm.set_goal_state(configuration_name=state_name)
        
        # Plan the motion
        plan_result = self.parol6_arm.plan()
        
        if plan_result:
            self.get_logger().info("Plan successful! Executing...")
            # Execute the plan
            robot_trajectory = plan_result.trajectory
            self.moveit.execute(robot_trajectory, controllers=[])
            return True
        else:
            self.get_logger().error("Planning failed!")
            return False
    
    def go_to_joint_positions(self, joint_positions):
        """
        Move to specific joint positions
        
        Args:
            joint_positions: List of 6 joint angles in radians [L1, L2, L3, L4, L5, L6]
        """
        if len(joint_positions) != 6:
            self.get_logger().error("Must provide exactly 6 joint positions!")
            return False
        
        self.get_logger().info(f"Planning to joint positions: {joint_positions}")
        
        # Create a robot state with the desired joint positions
        robot_state = RobotState(self.moveit.get_robot_model())
        joint_names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
        
        for name, position in zip(joint_names, joint_positions):
            robot_state.set_joint_positions(name, [position])
        
        # Set the goal state
        self.parol6_arm.set_goal_state(robot_state=robot_state)
        
        # Plan and execute
        plan_result = self.parol6_arm.plan()
        
        if plan_result:
            self.get_logger().info("Plan successful! Executing...")
            robot_trajectory = plan_result.trajectory
            self.moveit.execute(robot_trajectory, controllers=[])
            return True
        else:
            self.get_logger().error("Planning failed!")
            return False
    
    def go_to_pose(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """
        Move end effector to a specific pose
        
        Args:
            x, y, z: Position in meters
            roll, pitch, yaw: Orientation in radians
        """
        self.get_logger().info(f"Planning to pose: x={x}, y={y}, z={z}")
        
        # Create pose goal
        pose_goal = Pose()
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z
        
        # Convert RPY to quaternion (simplified - you may want to use tf_transformations)
        from math import sin, cos
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)
        
        pose_goal.orientation.w = cr * cp * cy + sr * sp * sy
        pose_goal.orientation.x = sr * cp * cy - cr * sp * sy
        pose_goal.orientation.y = cr * sp * cy + sr * cp * sy
        pose_goal.orientation.z = cr * cp * sy - sr * sp * cy
        
        # Set the goal pose
        self.parol6_arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="L6")
        
        # Plan and execute
        plan_result = self.parol6_arm.plan()
        
        if plan_result:
            self.get_logger().info("Plan successful! Executing...")
            robot_trajectory = plan_result.trajectory
            self.moveit.execute(robot_trajectory, controllers=[])
            return True
        else:
            self.get_logger().error("Planning failed!")
            return False


def main():
    """Main function with example usage"""
    
    # Initialize ROS 2
    rclpy.init()
    
    # Create controller
    controller = PAROL6Controller()
    
    print("\n" + "="*50)
    print("PAROL6 MoveIt Python Controller")
    print("="*50)
    print("\nAvailable commands:")
    print("1. Go to 'home' position")
    print("2. Go to 'ready' position")
    print("3. Move to custom joint positions")
    print("4. Move to custom pose")
    print("5. Exit")
    print("="*50 + "\n")
    
    try:
        while rclpy.ok():
            choice = input("Enter choice (1-5): ").strip()
            
            if choice == '1':
                controller.go_to_named_state('home')
                
            elif choice == '2':
                controller.go_to_named_state('ready')
                
            elif choice == '3':
                print("Enter 6 joint angles in radians (space-separated):")
                print("Example: 0.0 -0.5 0.5 0.0 0.0 0.0")
                try:
                    positions = [float(x) for x in input("> ").split()]
                    controller.go_to_joint_positions(positions)
                except ValueError:
                    print("Invalid input! Please enter 6 numbers.")
                    
            elif choice == '4':
                print("Enter target pose (x y z roll pitch yaw):")
                print("Example: 0.3 0.0 0.4 0.0 0.0 0.0")
                try:
                    pose = [float(x) for x in input("> ").split()]
                    if len(pose) == 6:
                        controller.go_to_pose(*pose)
                    else:
                        print("Invalid input! Please enter 6 numbers.")
                except ValueError:
                    print("Invalid input! Please enter 6 numbers.")
                    
            elif choice == '5':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice!")
            
            # Spin once to process callbacks
            rclpy.spin_once(controller, timeout_sec=0.1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
