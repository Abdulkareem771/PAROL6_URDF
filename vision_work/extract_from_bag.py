import os
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
# Path to the bag folder (the folder that contains metadata.yaml + .db3)
BAG_FOLDER = "/workspace/vision_work/rosbags/ai_dataset_2"   # adjust to your bag location
OUT_RGB = "/workspace/vision_work/extracted_dataset/rgb"
OUT_DEPTH = "/workspace/vision_work/extracted_dataset/depth"

os.makedirs(OUT_RGB, exist_ok=True)
os.makedirs(OUT_DEPTH, exist_ok=True)

bridge = CvBridge()

storage_options = StorageOptions(uri=BAG_FOLDER, storage_id="sqlite3")
converter_options = ConverterOptions("", "")

reader = SequentialReader()
reader.open(storage_options, converter_options)

# prepare topic name variables
#rgb_topic = "/kinect2/qhd/image_color"
#depth_topic = "/kinect2/sd/image_depth"
rgb_topic = "/kinect2/hd/image_color"
depth_topic = "/kinect2/hd/image_depth"


count = 0
while reader.has_next():
    (topic, data, t) = reader.read_next()
    # t is a builtin_interfaces/Time in serialized form; rosbag returns nanoseconds int typically
    try:
        msg = deserialize_message(data, Image)
    except Exception as e:
        print("Failed to deserialize message:", e)
        continue

    # Use the header stamp for filenames when available
    stamp = None
    try:
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    except Exception:
        # fallback to counter
        stamp = count

    if topic == rgb_topic:
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(f"{OUT_RGB}/{count}.png", cv_img)
    elif topic == depth_topic:
        # Depth often encoded as 16UC1 or 32FC1; use passthrough and save as PNG (scaled if needed)
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # Optionally normalize for visualization:
        # cv2.normalize(cv_img, cv_img, 0, 65535, cv2.NORM_MINMAX)
        cv2.imwrite(f"{OUT_DEPTH}/{count}.png", cv_img)

    count += 1

print("Done. Extracted", count, "messages.")
