import math

cmd_rad = 25.0 * math.pi / 180.0
total = 25.0 * math.pi / 180.0 # Motor moves 25 degrees
gear = 10.0

actual = total / gear

print(f"If motor moves 25 degrees, actual_position = {actual*180/math.pi} degrees")
error = cmd_rad - actual
print(f"Error = {error*180/math.pi} degrees")
velocity_command = 2.0 * error
motor_vel = velocity_command * gear
step_freq = (motor_vel * 3200) / (2.0*math.pi)
print(f"Step Freq = {step_freq} Hz")
