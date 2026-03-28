""" 
First Run the ArUco node by the following code:

ros2 run aruco_ros single \
    --ros-args --remap /image:=/kinect2/sd/image_color_rect \
    --ros-args --remap /camera_info:=/kinect2/sd/camera_info \
    --ros-args \
    -p marker_id:=6 \
    -p marker_size:=0.0545 \
    -p camera_frame:=kinect2_ir_optical_frame \
    -p marker_frame:=detected_marker_frame \
    -p corner_refinement:=SUBPIX \
    -p image_is_rectified:=True \
    -p marker_dict:=DICT_ARUCO_ORIGINAL


"""


import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class EyeToHandCalibrator(Node):
    def __init__(self):
        super().__init__('eye_to_hand_calibrator')

        # --- CONFIGURATION ---
        self.source_frame = 'kinect2_ir_optical_frame' # Camera Frame
        self.target_frame = 'detected_marker_frame'               # ArUco Marker Frame
        self.samples_to_collect = 100
        # ---------------------

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.translations = []
        self.quaternions = []
        
        self.get_logger().info(f"Starting calibration. Looking for transform from {self.source_frame} to {self.target_frame}...")
        
        # Timer to check for transforms
        self.timer = self.create_timer(0.1, self.collect_samples)

    def collect_samples(self):
        if len(self.translations) >= self.samples_to_collect:
            self.compute_final_transform()
            self.timer.cancel()
            return

        try:
            # Get the latest transform
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.source_frame,
                self.target_frame,
                now)

            # Store Translation
            t = trans.transform.translation
            self.translations.append([t.x, t.y, t.z])

            # Store Rotation (Quaternion)
            q = trans.transform.rotation
            self.quaternions.append([q.x, q.y, q.z, q.w])

            if len(self.translations) % 10 == 0:
                self.get_logger().info(f"Collected {len(self.translations)}/{self.samples_to_collect} samples...")

        except TransformException as ex:
            self.get_logger().warn(f"Could not find transform: {ex}")

    def compute_final_transform(self):
        self.get_logger().info("Collection complete. Calculating average...")

        # 1. Average Translation (Simple Mean)
        avg_translation = np.mean(self.translations, axis=0)

        # 2. Average Rotation 
        # Note: You can't just mean quaternions. We use Scipy's rotation averaging.
        rots = R.from_quat(self.quaternions)
        avg_rotation = rots.mean() # Computes the geometric mean of rotations
        avg_quat = avg_rotation.as_quat()
        euler = avg_rotation.as_euler('xyz', degrees=True)

        self.get_logger().info("\n--- CALIBRATION RESULTS (Camera to Marker) ---")
        print(f"Translation (meters): X={avg_translation[0]:.4f}, Y={avg_translation[1]:.4f}, Z={avg_translation[2]:.4f}")
        print(f"Quaternion: x={avg_quat[0]:.4f}, y={avg_quat[1]:.4f}, z={avg_quat[2]:.4f}, w={avg_quat[3]:.4f}")
        print(f"Euler (degrees): Roll={euler[0]:.2f}, Pitch={euler[1]:.2f}, Yaw={euler[2]:.2f}")
        
        print("\n--- SUGGESTED STATIC TRANSFORM COMMAND ---")
        print(f"ros2 run tf2_ros static_transform_publisher {avg_translation[0]:.4f} {avg_translation[1]:.4f} {avg_translation[2]:.4f} "
              f"{avg_quat[0]:.4f} {avg_quat[1]:.4f} {avg_quat[2]:.4f} {avg_quat[3]:.4f} {self.source_frame} {self.target_frame}")
        
        self.get_logger().info("Calibration finished. You can close this node.")

def main(args=None):
    rclpy.init(args=args)
    node = EyeToHandCalibrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()