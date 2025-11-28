import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import json

class MobileBridge(Node):
    def __init__(self):
        super().__init__('mobile_bridge')
        
        # Publisher for joint commands
        self.joint_pub = self.create_publisher(String, '/joint_commands', 10)
        # Publisher for direct control
        self.control_pub = self.create_publisher(Twist, '/mobile_control', 10)
        
        # Subscriber for web commands
        self.create_subscription(String, '/web_commands', self.web_command_callback, 10)
        
        self.get_logger().info('📱 Mobile ROS Bridge Started! Ready for phone control.')

    def web_command_callback(self, msg):
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type')
            
            if cmd_type == 'joint_command':
                # Send joint position commands
                positions = command.get('positions', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                joint_cmd = {
                    'joints': ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6'],
                    'positions': positions
                }
                cmd_msg = String()
                cmd_msg.data = json.dumps(joint_cmd)
                self.joint_pub.publish(cmd_msg)
                self.get_logger().info(f'📤 Joint command: {positions}')
                
            elif cmd_type == 'control_command':
                # Send direct control commands
                twist = Twist()
                twist.linear.x = command.get('linear_x', 0.0)
                twist.angular.z = command.get('angular_z', 0.0)
                self.control_pub.publish(twist)
                self.get_logger().info(f'📤 Control command: x={twist.linear.x}, z={twist.angular.z}')
                
            elif cmd_type == 'moveit_command':
                action = command.get('action')
                if action == 'home':
                    # Send home position
                    home_cmd = {
                        'joints': ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6'],
                        'positions': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    }
                    cmd_msg = String()
                    cmd_msg.data = json.dumps(home_cmd)
                    self.joint_pub.publish(cmd_msg)
                    self.get_logger().info('🏠 Home command sent')
                    
                elif action == 'stop':
                    # Emergency stop
                    stop_cmd = {'action': 'emergency_stop'}
                    cmd_msg = String()
                    cmd_msg.data = json.dumps(stop_cmd)
                    self.joint_pub.publish(cmd_msg)
                    self.get_logger().warn('🛑 Emergency stop sent')
                
        except Exception as e:
            self.get_logger().error(f'❌ Error processing web command: {e}')

def main():
    rclpy.init()
    node = MobileBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Mobile bridge shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
