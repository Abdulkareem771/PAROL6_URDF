import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import json

class MobileBridge(Node):
    def __init__(self):
        super().__init__('mobile_bridge')
        
        # Publishers for robot control
        self.joy_pub = self.create_publisher(Twist, '/mobile_joy', 10)
        self.command_pub = self.create_publisher(String, '/mobile_commands', 10)
        
        # Subscriber for web commands (from rosbridge)
        self.create_subscription(String, '/web_commands', self.web_command_callback, 10)
        
        self.get_logger().info('Mobile ROS Bridge Started')

    def web_command_callback(self, msg):
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type')
            
            if cmd_type == 'joystick':
                # Handle joystick commands
                twist = Twist()
                twist.linear.x = command.get('x', 0.0)
                twist.angular.z = command.get('z', 0.0)
                self.joy_pub.publish(twist)
                self.get_logger().info(f'Joystick command: x={twist.linear.x}, z={twist.angular.z}')
                
            elif cmd_type == 'moveit_command':
                # Forward to MoveIt
                cmd_msg = String()
                cmd_msg.data = json.dumps(command)
                self.command_pub.publish(cmd_msg)
                self.get_logger().info(f'MoveIt command: {command.get("action")}')
                
        except Exception as e:
            self.get_logger().error(f'Error processing web command: {e}')

def main():
    rclpy.init()
    node = MobileBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
