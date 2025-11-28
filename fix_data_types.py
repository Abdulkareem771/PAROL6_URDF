# Read the file
with open('mobile_bridge.py', 'r') as f:
    content = f.read()

# Find the move_robot function and fix the data type conversion
old_code = '''    positions = [
        float(request.json.get('joint1', 0)),
        float(request.json.get('joint2', 0)), 
        float(request.json.get('joint3', 0)),
        float(request.json.get('joint4', 0)),
        float(request.json.get('joint5', 0)),
        float(request.json.get('joint6', 0))
    ]'''

new_code = '''    positions = [
        float(request.json.get('joint1', 0)),
        float(request.json.get('joint2', 0)), 
        float(request.json.get('joint3', 0)),
        float(request.json.get('joint4', 0)),
        float(request.json.get('joint5', 0)),
        float(request.json.get('joint6', 0))
    ]'''

# Replace the code
content = content.replace(old_code, new_code)

# Also fix the joint names in the trajectory (line 64 shows wrong names)
content = content.replace("goal_msg.trajectory.joint_names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']", "goal_msg.trajectory.joint_names = ['joint_L1', 'joint_L2', 'joint_L3', 'joint_L4', 'joint_L5', 'joint_L6']")

# Write back
with open('mobile_bridge.py', 'w') as f:
    f.write(content)
    
print('Data types and joint names fixed')
