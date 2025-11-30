#!/usr/bin/env python3
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
        
        # Scaling factors (adjust these to change speed)
        LINEAR_SCALE = 0.2
        ANGULAR_SCALE = 0.5

        # Xbox Mapping
        # Left Stick (Axes 0,1) -> Linear X/Y
        # Right Stick Vertical (Axis 4) -> Linear Z
        # Right Stick Horizontal (Axis 3) -> Angular Y (Pitch)
        # D-Pad Vertical (Axis 7) -> Angular X (Roll) - changed from triggers for simplicity
        # Triggers (Axis 2, 5) -> Angular Z (Yaw)

        twist_msg.twist.linear.x = msg.axes[1] * LINEAR_SCALE
        twist_msg.twist.linear.y = msg.axes[0] * LINEAR_SCALE
        twist_msg.twist.linear.z = msg.axes[4] * LINEAR_SCALE
        
        twist_msg.twist.angular.x = msg.axes[3] * ANGULAR_SCALE
        twist_msg.twist.angular.y = msg.axes[7] * ANGULAR_SCALE 
        # Combine triggers for Z rotation (Right trigger - Left trigger)
        trigger_val = (msg.axes[5] - msg.axes[2]) / 2.0
        twist_msg.twist.angular.z = trigger_val * ANGULAR_SCALE
        
        self.twist_pub.publish(twist_msg)

def main():
    rclpy.init()
    node = XboxToServo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
