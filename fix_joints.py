import re

# Read the file
with open('mobile_bridge.py', 'r') as f:
    content = f.read()

# Replace joint names - update this based on the actual joint names
# If the joints are joint_L1, joint_L2, joint_L3, joint_R1, joint_R2, joint_R3, etc.
# Adjust this pattern based on what you see from the joint_states topic
content = re.sub(
    rjoint_names
