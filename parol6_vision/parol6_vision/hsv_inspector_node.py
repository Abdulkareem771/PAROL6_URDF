#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class HSVInspector(Node):

    def __init__(self):
        super().__init__('hsv_inspector')

        self.image_topic = '/kinect2/qhd/image_color_rect'

        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.cv_image = None
        self.hsv_image = None
        self.display = None

        cv2.namedWindow("HSV Inspector")
        cv2.setMouseCallback("HSV Inspector", self.mouse_callback)

        self.get_logger().info(f"HSV Inspector running on: {self.image_topic}")

    def image_callback(self, msg):
        try:
            # Convert ROS Image → OpenCV (BGR)
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert to HSV
            self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

            # If display not yet created, initialize it
            if self.display is None:
                self.display = self.cv_image.copy()

            cv2.imshow("HSV Inspector", self.display)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if self.cv_image is None or self.hsv_image is None:
            return

        if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:

            # Read pixel values
            b, g, r = self.cv_image[y, x]
            h, s, v = self.hsv_image[y, x]

            # Always redraw from clean image
            self.display = self.cv_image.copy()

            # Overlay text
            cv2.putText(self.display, f"Pixel: ({x}, {y})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(self.display, f"HSV: ({h}, {s}, {v})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Optional: keep RGB too (you can remove this if you want)
            cv2.putText(self.display, f"RGB: ({r}, {g}, {b})", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Draw cursor marker
            cv2.circle(self.display, (x, y), 5, (0, 255, 255), 2)


def main(args=None):
    rclpy.init(args=args)
    node = HSVInspector()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
