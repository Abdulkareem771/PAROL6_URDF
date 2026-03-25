import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class DummyJointPublisher(Node):
    def __init__(self):
        super().__init__('dummy_joint_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        
        # Joint names for PAROL6 (matching actual URDF)
        self.joint_names = [
            'joint_L1', 
            'joint_L2', 
            'joint_L3', 
            'joint_L4', 
            'joint_L5', 
            'joint_L6'
        ]

    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [0.0] * len(self.joint_names)
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DummyJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
