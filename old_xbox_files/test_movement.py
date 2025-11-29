import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

def main():
    rclpy.init()
    node = Node('test_movement_sender')
    
    pub = node.create_publisher(
        JointTrajectory, 
        '/parol6_arm_controller/joint_trajectory', 
        10
    )
    
    # Wait for connection
    time.sleep(1)
    
    msg = JointTrajectory()
    msg.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']
    
    point = JointTrajectoryPoint()
    # Move to a safe test position
    point.positions = [0.2, -0.2, 0.2, 0.0, 0.0, 0.0]
    point.time_from_start.sec = 2
    
    msg.points.append(point)
    
    node.get_logger().info('Publishing test trajectory...')
    pub.publish(msg)
    
    # Keep alive briefly to ensure send
    time.sleep(1)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
