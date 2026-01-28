# WeldLine3D.msg
# Represents a detected welding seam in 3D space (robot frame)

# Unique identifier for this weld line (matches 2D detection)
string id

# Detection confidence (0.0 to 1.0)
# Computed as: (valid_pixels_after_filtering / total_detected_pixels) * line_continuity_score
# where line_continuity_score measures smoothness and connectedness
float32 confidence

# Ordered sequence of 3D points along the seam
# Points are in the frame specified by header.frame_id (typically base_link)
# Uses standard geometry_msgs/Point for full 3D coordinates
geometry_msgs/Point[] points

# Optional: Average line width in meters (from depth analysis)
float32 line_width

# Optional: Quality metrics
float32 depth_quality  # Percentage of valid depth points (0.0 to 1.0)
int32 num_points       # Number of valid 3D points

# Coordinate frame
std_msgs/Header header
