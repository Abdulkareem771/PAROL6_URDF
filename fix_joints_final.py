# Read the file
with open('mobile_bridge.py', 'r') as f:
    content = f.read()

# Replace the joint names with the correct ones
content = content.replace("['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']", "['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']")

# Write back
with open('mobile_bridge.py', 'w') as f:
    f.write(content)
    
print('Joint names updated successfully')
