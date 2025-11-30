import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped

class XboxToServo(Node):
    def __init__(self):
        super().__init__('xbox_to_servo')
        self.twist_pub = self.create_publisher(TwistStamped, '/servo/delta_twist_cmds', 10)
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.get_logger().info('Xbox to Servo Bridge READY')

    def joy_callback(self, msg):
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'link_base'
        
        twist_msg.twist.linear.x = msg.axes[1] * 0.1
        twist_msg.twist.linear.y = msg.axes[0] * 0.1
        twist_msg.twist.linear.z = msg.axes[4] * 0.1
        twist_msg.twist.angular.x = (msg.axes[5] - msg.axes[2]) * 0.2
        twist_msg.twist.angular.y = msg.axes[3] * 0.2
        twist_msg.twist.angular.z = 0.0
        
        self.twist_pub.publish(twist_msg)

def main():
    rclpy.init()
    node = XboxToServo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
