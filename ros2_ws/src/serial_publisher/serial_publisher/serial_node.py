#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import threading
import time
import traceback

class SerialPublisher(Node):
    def __init__(self):
        super().__init__('serial_publisher')
        # declare with typed defaults
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)

        # retrieve values correctly and defend against invalid types
        try:
            port = self.get_parameter('port').value
        except Exception:
            port = '/dev/ttyUSB0'
        try:
            baud_val = self.get_parameter('baudrate').value
            # sometimes param may come as string; coerce safely
            if isinstance(baud_val, str):
                baud = int(baud_val) if baud_val.isdigit() else 115200
            else:
                baud = int(baud_val)
        except Exception:
            baud = 115200

        self.get_logger().info(f'Using serial port: {port}, baud: {baud}')

        self.pub = self.create_publisher(String, 'esp_serial', 10)

        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            self.get_logger().info('Serial port opened successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            self.get_logger().error('Traceback: ' + traceback.format_exc())
            self.ser = None

        if self.ser:
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                line = self.ser.readline()
                if not line:
                    continue
                text = line.decode(errors='ignore').strip()
                if text:
                    msg = String()
                    msg.data = text
                    self.pub.publish(msg)
                    self.get_logger().debug(f'Published: {text}')
            except Exception as e:
                self.get_logger().error(f'Serial read error: {e}')
                time.sleep(0.5)

    def destroy_node(self):
        if hasattr(self, '_stop'):
            self._stop.set()
            self._thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SerialPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
