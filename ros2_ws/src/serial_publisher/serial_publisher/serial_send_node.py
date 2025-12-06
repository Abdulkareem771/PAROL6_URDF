#!/usr/bin/env python3
# serial_writer.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import traceback

class SerialWriter(Node):
    def __init__(self):
        super().__init__('serial_writer')
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        port = self.get_parameter('port').value
        baud = self.get_parameter('baudrate').value

        try:
            baud = int(baud)
        except Exception:
            baud = 115200

        self.get_logger().info(f'Opening serial: {port} @ {baud}')
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            self.get_logger().info('Serial opened')
        except Exception as e:
            self.get_logger().error(f'Failed to open serial: {e}')
            self.get_logger().error('Traceback: ' + traceback.format_exc())
            self.ser = None

        self.sub = self.create_subscription(String, 'esp_command', self.cb, 10)

    def cb(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        out = text + '\n'
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(out.encode('utf-8'))
                self.get_logger().info(f"Sent to serial: '{text}'")
            except Exception as e:
                self.get_logger().error(f"Write error: {e}")
        else:
            self.get_logger().warn("Serial not open - cannot send")

    def destroy_node(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SerialWriter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# send one command
#ros2 topic pub /esp_command std_msgs/String "data: 'ON'"

# send OFF
#ros2 topic pub /esp_command std_msgs/String "data: 'OFF'"

# send BLINK
#ros2 topic pub /esp_command std_msgs/String "data: 'BLINK'"
