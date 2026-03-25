# WeldLine.msg
# Represents a detected welding seam in 2D image space

# Unique identifier for this weld line
string id

# Detection confidence (0.0 to 1.0)
# Computed as: (valid_pixels_after_filtering / total_detected_pixels) * line_continuity_score
# where line_continuity_score measures smoothness and connectedness
float32 confidence

# Ordered sequence of points along the line in image coordinates (pixels)
# Points should be ordered from start to end of the seam
# Uses Point32: x,y = pixel coordinates, z = 0 (2D image)
geometry_msgs/Point32[] pixels

# Optional: Bounding box for visualization/region of interest
# Can be computed from min/max of pixels
geometry_msgs/Point bbox_min
geometry_msgs/Point bbox_max

# Image frame this detection is relative to
std_msgs/Header header
