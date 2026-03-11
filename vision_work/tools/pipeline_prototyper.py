import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QLineEdit, QFileDialog, QCheckBox,
    QComboBox, QMessageBox
)
from PySide6.QtCore import Qt, QTimer

import time
import importlib.util

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# Add parent dir to path to import vision_core
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.qt_base_gui import BaseVisionApp, run_app, C

class PipelinePrototyper(BaseVisionApp):
    def __init__(self):
        super().__init__(title="Pipeline Prototyper (Linear 4-Step)", width=1400, height=800)
        
        self.ros_node = None
        self.ros_sub = None
        self.ros_pub = None
        self.bridge = None
        self.yolo_model = None
        self.custom_module = None
        
        if ROS2_AVAILABLE:
            if not rclpy.ok():
                rclpy.init()
            self.ros_node = rclpy.create_node('pipeline_prototyper_node')
            self.bridge = CvBridge()
            
            self.ros_timer = QTimer(self)
            self.ros_timer.timeout.connect(self._spin_ros)
            self.ros_timer.start(10) # 100hz
            
        self.setup_ui()
        
    def _spin_ros(self):
        if self.ros_node:
            rclpy.spin_once(self.ros_node, timeout_sec=0.005)
        
    def setup_ui(self):
        # Slot 1: Input (Extend base image loader with ROS topic)
        self._build_default_image_loader()
        
        self.sidebar_layout.addSpacing(10)
        self.ros_in_topic = QLineEdit()
        self.ros_in_topic.setPlaceholderText("ROS Input Topic (e.g. /camera/image_raw)")
        self.sidebar_layout.addWidget(self.ros_in_topic)
        
        btn_ros_in = QPushButton("📡 Subscribe to ROS Topic")
        btn_ros_in.setStyleSheet(f"background-color: {C['warn']}; color: {C['bg']};")
        btn_ros_in.clicked.connect(self._subscribe_ros)
        self.sidebar_layout.addWidget(btn_ros_in)

        # Slot 2: ML Inference
        self._add_section_header("Slot 2: ML Inference")
        self.model_path_entry = QLineEdit()
        self.model_path_entry.setReadOnly(True)
        self.model_path_entry.setPlaceholderText("No model loaded...")
        self.sidebar_layout.addWidget(self.model_path_entry)
        self._add_button("🧠 Load Weights (.pt/.onnx)", self._load_model)

        # Slot 3: Custom Script
        self._add_section_header("Slot 3: Custom Script")
        self.script_path_entry = QLineEdit()
        self.script_path_entry.setReadOnly(True)
        self.script_path_entry.setPlaceholderText("No script loaded...")
        self.sidebar_layout.addWidget(self.script_path_entry)
        self._add_button("🐍 Load Python Script", self._load_script)

        # Slot 4: ROS Output
        self._add_section_header("Slot 4: Output Destination")
        self.ros_out_topic = QLineEdit()
        self.ros_out_topic.setPlaceholderText("ROS Output Topic (e.g. /vision/result)")
        self.sidebar_layout.addWidget(self.ros_out_topic)
        
        btn_ros_out = QPushButton("🚀 Publish live to ROS")
        btn_ros_out.setStyleSheet(f"background-color: {C['green']}; color: {C['bg']};")
        btn_ros_out.clicked.connect(self._publish_ros)
        self.sidebar_layout.addWidget(btn_ros_out)
        
        # Profiling stats
        self._add_section_header("Performance Profiling")
        self.lbl_fps = QLabel("FPS: -- | Latency: -- ms")
        self.sidebar_layout.addWidget(self.lbl_fps)

        # Run button
        self.sidebar_layout.addStretch()
        self._add_button("▶ Run Pipeline", self._run_pipeline, primary=True)

    def _load_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Model Weights", "", "Models (*.pt *.onnx)")
        if p:
            if not ULTRALYTICS_AVAILABLE:
                QMessageBox.critical(self, "Error", "ultralytics is not installed.")
                return
            try:
                self.yolo_model = YOLO(p)
                self.model_path_entry.setText(p)
            except Exception as e:
                QMessageBox.critical(self, "YOLO Error", str(e))

    def _load_script(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Python Script", "", "Python (*.py)")
        if p:
            try:
                spec = importlib.util.spec_from_file_location("custom_node", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.custom_module = mod
                self.script_path_entry.setText(p)
            except Exception as e:
                QMessageBox.critical(self, "Script Error", f"Failed to load script:\n{e}")

    def _run_pipeline(self):
        if self._rgb is None:
            return
            
        start_t = time.perf_counter()
        img = self._rgb.copy()
        
        # --- SLOT 2: ML Inference ---
        if self.yolo_model is not None:
            results = self.yolo_model.predict(img, verbose=False)
            img_bgr = results[0].plot()
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
        # --- SLOT 3: Custom Script Injector ---
        if self.custom_module is not None:
            try:
                tmp_in = "/tmp/prototyper_in.jpg"
                cv2.imwrite(tmp_in, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                if hasattr(self.custom_module, 'segment_blocks'):
                    res = self.custom_module.segment_blocks(tmp_in)
                    if res and len(res) >= 3 and res[2] is not None:
                        img = res[2] # Teammate's script returns RGB array
                elif hasattr(self.custom_module, 'process_image'):
                    img_bgr = self.custom_module.process_image(tmp_in)
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Custom Script Error: {e}")
                
        # --- Profiling & UI Update ---
        end_t = time.perf_counter()
        latency_ms = (end_t - start_t) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0
        self.lbl_fps.setText(f"FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms")
        self._display_rgb(img)
        
        # --- SLOT 4: Output Publish Loop ---
        if self.ros_pub and img is not None:
             msg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
             self.ros_pub.publish(msg)

    def _subscribe_ros(self):
        if not ROS2_AVAILABLE:
            QMessageBox.critical(self, "ROS Error", "rclpy is not available in this Python environment.")
            return
        topic = self.ros_in_topic.text().strip()
        if not topic:
            return
            
        if self.ros_sub:
            self.ros_node.destroy_subscription(self.ros_sub)
            
        self.ros_sub = self.ros_node.create_subscription(Image, topic, self._ros_image_callback, 10)
        self.path_entry.setText(f"[ROS2] {topic}")

    def _ros_image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self._set_rgb_image(cv_img)
            self._run_pipeline() # Automatically trigger pipeline on new frames
        except Exception as e:
            print(f"Error converting ROS Image: {e}")

    def _publish_ros(self):
        if not ROS2_AVAILABLE:
            QMessageBox.critical(self, "ROS Error", "rclpy is not available.")
            return
        topic = self.ros_out_topic.text().strip()
        if not topic:
            return
            
        # Create Publisher if it doesn't exist or topic changed
        if self.ros_pub and self.ros_pub.topic_name != topic:
            self.ros_node.destroy_publisher(self.ros_pub)
            self.ros_pub = None
            
        if not self.ros_pub:
             self.ros_pub = self.ros_node.create_publisher(Image, topic, 10)
             QMessageBox.information(self, "ROS Topic", f"Created publisher on: {topic}")
        
        # Actually publish the current canvas
        current_img = self.get_current_rgb()
        if current_img is not None:
             msg = self.bridge.cv2_to_imgmsg(current_img, encoding="rgb8")
             self.ros_pub.publish(msg)

if __name__ == "__main__":
    run_app(PipelinePrototyper)
