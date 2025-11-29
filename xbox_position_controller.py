import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64

class XboxPositionController(Node):
    def __init__(self):
        super().__init__('xbox_position_controller')
        
        # Create individual joint position publishers
        self.joint_pubs = []
        joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']
        for joint in joint_names:
            pub = self.create_publisher(Float64, f'/position_controllers/commands', 10)
            self.joint_pubs.append(pub)
        
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        
        self.positions = [0.0] * 6
        self.deadzone = 0.15
        
        self.get_logger().info('🎮 Xbox Position Controller Started!')

    def joy_callback(self, msg):
        # Simple direct control
        base = msg.axes[0] * 0.5 if abs(msg.axes[0]) > self.deadzone else 0.0
        shoulder = -msg.axes[1] * 0.5 if abs(msg.axes[1]) > self.deadzone else 0.0
        elbow = msg.axes[3] * 0.5 if abs(msg.axes[3]) > self.deadzone else 0.0
        wrist_pitch = -msg.axes[4] * 0.5 if abs(msg.axes[4]) > self.deadzone else 0.0
        wrist_roll = (msg.axes[5] - msg.axes[2]) * 0.25
        
        # Update positions
        self.positions[0] += base * 0.1
        self.positions[1] += shoulder * 0.1
        self.positions[2] += elbow * 0.1
        self.positions[3] += wrist_pitch * 0.1
        self.positions[4] += wrist_roll * 0.05
        
        # Publish positions
        for i, pos in enumerate(self.positions):
            msg = Float64()
            msg.data = pos
            self.joint_pubs[i].publish(msg)
        
        if msg.buttons[0]:  # Reset
            self.positions = [0.0] * 6
            self.get_logger().info('Reset positions')

def main():
    rclpy.init()
    node = XboxPositionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
