"""
parol6_vision - Vision-guided welding path detection

This package implements the complete perception → planning → execution pipeline
for red line welding path detection on the PAROL6 robot.

Nodes:
    - red_line_detector: Detect red marker lines in images
    - depth_matcher: Project 2D detections to 3D using depth
    - path_generator: Generate smooth welding trajectories
    - moveit_controller: Execute paths using MoveIt2
"""

__version__ = '1.0.0'
__author__ = 'PAROL6 Vision Team'
