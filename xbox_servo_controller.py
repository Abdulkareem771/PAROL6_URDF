import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped
from control_msgs.msg import JointJog
import numpy as np

class XboxServoController(Node):
    def __init__(self):
        super().__init__('xbox_servo_controller')
        self.twist_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/servo_node/delta_joint_cmds', 10)
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        
        self.declare_parameter('linear_scale', 0.1)
        self.declare_parameter('angular_scale', 0.3)
        self.declare_parameter('joint_scale', 0.5)
        self.declare_parameter('deadzone', 0.15)
        
        self.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']
        self.get_logger().info('🎮 Xbox Servo Controller READY')

    def apply_deadzone(self, value):
        deadzone = self.get_parameter('deadzone').value
        if abs(value) < deadzone: return 0.0
        return value

    def joy_callback(self, msg):
        cartesian_mode = msg.buttons[4]  # LB
        joint_mode = msg.buttons[5]      # RB
        
        if cartesian_mode:
            self.publish_twist_command(msg)
        elif joint_mode:
            self.publish_joint_command(msg)
        else:
            self.publish_twist_command(msg)

    def publish_twist_command(self, msg):
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'link_base'
        
        linear_scale = self.get_parameter('linear_scale').value
        angular_scale = self.get_parameter('angular_scale').value
        
        twist_msg.twist.linear.x = self.apply_deadzone(msg.axes[1]) * linear_scale
        twist_msg.twist.linear.y = self.apply_deadzone(msg.axes[0]) * linear_scale
        twist_msg.twist.linear.z = self.apply_deadzone(msg.axes[4]) * linear_scale
        twist_msg.twist.angular.x = (msg.axes[5] - msg.axes[2]) * 0.5 * angular_scale
        twist_msg.twist.angular.y = self.apply_deadzone(msg.axes[3]) * angular_scale
        twist_msg.twist.angular.z = 0.0
        
        self.twist_pub.publish(twist_msg)

    def publish_joint_command(self, msg):
        joint_scale = self.get_parameter('joint_scale').value
        deadzone = self.get_parameter('deadzone').value
        
        if abs(msg.axes[0]) > deadzone:
            joint_msg = JointJog()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.joint_names = ['joint_L1']
            joint_msg.velocities = [msg.axes[0] * joint_scale]
            self.joint_pub.publish(joint_msg)
            
        if abs(msg.axes[1]) > deadzone:
            joint_msg = JointJog()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.joint_names = ['joint_L2']
            joint_msg.velocities = [-msg.axes[1] * joint_scale]
            self.joint_pub.publish(joint_msg)

def main():
    rclpy.init()
    node = XboxServoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
