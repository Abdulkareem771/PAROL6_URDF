#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import subprocess
import time
import sys
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

class AutoTestRunner(Node):
    def __init__(self, shape):
        super().__init__('auto_test_runner')
        self.shape = shape
        # Use transient-local (latching) QoS so the controller receives the
        # path even if it subscribes after the publish call.
        latch_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.path_pub = self.create_publisher(Path, '/vision/welding_path', latch_qos)
        self.client = self.create_client(Trigger, '/moveit_controller/execute_welding_path')
        self.status_client = self.create_client(Trigger, '/moveit_controller/is_execution_idle')
        self.controller_proc = None

    def run(self):
        # 0. Kill stale nodes and wait long enough for DDS to de-register their endpoints.
        #    0.5s is not enough — DDS endpoint liveliness lease can take 2-3 seconds.
        #    Without this, wait_for_service() returns True on the dead service immediately.
        self.get_logger().info("Killing any stale moveit_controller processes...")
        subprocess.run(
            ["pkill", "-9", "-f", "moveit_controller"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(3.0)  # Must wait for DDS to expire old endpoints

        # 1. Start a fresh moveit_controller
        self.get_logger().info("Starting moveit_controller in the background...")
        self.controller_proc = subprocess.Popen(
            ["ros2", "run", "parol6_vision", "moveit_controller"]
        )

        # 2. Wait for the NEW service to become available (now DDS endpoints are clean)
        self.get_logger().info("Waiting for execute_welding_path service...")
        attempts = 0
        while not self.client.wait_for_service(timeout_sec=1.0):
            attempts += 1
            if attempts > 15:
                self.get_logger().error("Service never became available. Aborting.")
                self.cleanup()
                return
            self.get_logger().info('Service not available yet, waiting...')

        # 3. Give the node an extra moment to finish setting up its subscriptions
        #    (service advertised slightly before subscriptions are fully wired in DDS)
        time.sleep(0.5)

        # 4. Publish path and confirm delivery via re-publish loop
        if self.shape != 'Live Camera (No Inject)':
            path_msg = self.generate_path(self.shape)
            path_received = False
            for attempt in range(5):
                self.get_logger().info(
                    f"Publishing '{self.shape}' path (attempt {attempt+1}/5)..."
                )
                self.path_pub.publish(path_msg)
                # Yield to let callbacks fire
                time.sleep(0.6)
                rclpy.spin_once(self, timeout_sec=0.1)

                # Ask the controller if it has a path yet via a dry-run trigger
                if self.status_client.wait_for_service(timeout_sec=0.5):
                    check_fut = self.status_client.call_async(Trigger.Request())
                    rclpy.spin_until_future_complete(self, check_fut, timeout_sec=1.0)
                    try:
                        r = check_fut.result()
                        # is_execution_idle returns success=True when idle AND path received
                        if r is not None:
                            path_received = True
                            break
                    except Exception:
                        pass

            if not path_received:
                self.get_logger().warn("Could not confirm path delivery — proceeding anyway.")
        else:
            self.get_logger().info("Live Camera Mode selected. Waiting 3s for real vision path...")
            time.sleep(3.0)


        # 5. Call execute service
        self.get_logger().info("Triggering execution via service call...")
        req = Trigger.Request()
        future = self.client.call_async(req)
        
        # We must spin the node so the rclpy future processes correctly
        rclpy.spin_until_future_complete(self, future)
        
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f"Execution Successfully Triggered: {res.message}")
            else:
                self.get_logger().error(f"❌ Execution Failed to trigger: {res.message}")
                self.cleanup()
                return
        except Exception as e:
            self.get_logger().error(f"Service call failed with exception: {e}")
            self.cleanup()
            return
            
        # 5.5 Wait for execution to finish
        self.get_logger().info("Monitoring path execution progress. Waiting for robot to finish...")
        time.sleep(1.0) # Give the background thread time to lock the 'execution_in_progress' flag
        
        while True:
            if not self.status_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Status service disconnected.")
                break
                
            status_req = Trigger.Request()
            status_future = self.status_client.call_async(status_req)
            rclpy.spin_until_future_complete(self, status_future)
            try:
                status_res = status_future.result()
                if status_res.success:
                    self.get_logger().info("✅ Execution Sequence fully completed!")
                    break
            except Exception as e:
                self.get_logger().warn(f"Error checking execution status: {e}")
            
            time.sleep(1.0)

        # 6. Cleanup
        self.cleanup()

    def cleanup(self):
        self.get_logger().info("Test routine complete. Shutting down background moveit_controller.")
        if self.controller_proc:
            self.controller_proc.terminate()
            self.controller_proc.wait()

    def generate_path(self, shape):
        path = Path()
        path.header.frame_id = 'base_link'
        path.header.stamp = self.get_clock().now().to_msg()
        
        # Center of reachable test area for XY-plane patterns (tabletop):
        # Shifted outwards (+X) and low (+Z) to draw on the ground plane
        cx, cy, cz = 0.29, 0.0, 0.005
        points = []

        if shape == 'Straight':
            # Straight horizontal stroke in XY plane (12 cm in +Y)
            for i in range(7):
                points.append((cx, cy + i * 0.02 - 0.06, cz))
        elif shape == 'Curve':
            # Arc in XY plane (quarter circle, 8 cm radius)
            r = 0.08
            for i in range(31):
                theta = (math.pi / 2.0) * (i / 30.0)  # 0 -> pi/2
                points.append((cx + r * math.cos(theta) - r, cy + r * math.sin(theta), cz))
        elif shape == 'Circle':
            # Full circle in XY plane (10 cm diameter)
            r = 0.05
            for i in range(61):
                theta = (2 * math.pi / 60.0) * i
                points.append((cx + r * math.cos(theta), cy + r * math.sin(theta), cz))
        elif shape == 'ZigZag':
            # Zig-zag in XY plane: alternate X while moving Y (12 cm long, 8cm wide)
            for i in range(7):
                x_val = cx + (0.04 if i % 2 == 1 else -0.04)
                y_val = cy + i * 0.02 - 0.06
                points.append((x_val, y_val, cz))
        else: # default straight
            for i in range(7):
                points.append((cx, cy + i * 0.02 - 0.06, cz))

        for (x, y, z) in points:
            pose = PoseStamped()
            pose.header.frame_id = 'base_link'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            
            # Pointing the end effector STRAIGHT DOWN at the table
            # A 180-deg rotation around X-axis (roll) flips Z down and Y right
            pose.pose.orientation.x = 1.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 0.0
            path.poses.append(pose)

        return path

def main(args=None):
    rclpy.init(args=args)
    shape = sys.argv[1] if len(sys.argv) > 1 else 'Straight'
    node = AutoTestRunner(shape)
    
    try:
        node.run()
    except KeyboardInterrupt:
        node.cleanup()
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
