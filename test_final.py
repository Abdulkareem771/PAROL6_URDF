import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def test_action():
    rclpy.init()
    node = Node('test_node')
    
    action_client = ActionClient(node, FollowJointTrajectory, '/parol6_arm_controller/follow_joint_trajectory')
    
    print('Waiting for action server...')
    if not action_client.wait_for_server(timeout_sec=5.0):
        print('Action server not available!')
        return False
    
    goal_msg = FollowJointTrajectory.Goal()
    goal_msg.trajectory = JointTrajectory()
    goal_msg.trajectory.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']
    
    point = JointTrajectoryPoint()
    point.positions = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    point.time_from_start.sec = 2
    goal_msg.trajectory.points.append(point)
    
    print('Sending test goal...')
    future = action_client.send_goal_async(goal_msg)
    
    rclpy.spin_until_future_complete(node, future)
    
    if future.result() is not None:
        goal_handle = future.result()
        if goal_handle.accepted:
            print('SUCCESS: Goal accepted! The robot should move.')
            return True
        else:
            print('FAILED: Goal rejected')
            return False
    else:
        print('FAILED: No response from action server')
        return False

if __name__ == '__main__':
    test_action()
