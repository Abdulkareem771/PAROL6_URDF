import math

ENCODER_RESOLUTION = 4096.0
PI = math.pi
GEAR_RATIO = 10.0
ENCODER_DIR_SIGN = -1.0
ENCODER_OFFSET = 0.0

last_motor_angle = -1.0
motor_revolutions = 0

def process_counts(virtual_cnt):
    global last_motor_angle, motor_revolutions
    
    # 1. counts to motor angle
    # This is exactly what the Teensy does
    motor_ang = (virtual_cnt / ENCODER_RESOLUTION) * 2.0 * PI
    motor_ang *= ENCODER_DIR_SIGN
    motor_ang += ENCODER_OFFSET
    
    # Normalize [0, 2pi)
    while motor_ang < 0.0: motor_ang += 2.0 * PI
    while motor_ang >= 2.0 * PI: motor_ang -= 2.0 * PI
    
    # Multi-turn tracking
    if last_motor_angle < 0.0:
        last_motor_angle = motor_ang
        
    delta = motor_ang - last_motor_angle
    
    if delta > PI: motor_revolutions -= 1
    if delta < -PI: motor_revolutions += 1
    
    last_motor_angle = motor_ang
    
    # Total
    total = motor_ang + motor_revolutions * 2.0 * PI
    raw_joint_pos = total / GEAR_RATIO
    
    return motor_ang, motor_revolutions, raw_joint_pos

print("--- NEGATIVE ROTATION (Counts UP) ---")
for cnt in range(0, 5000, 500):
    virtual_cnt = cnt % 4096
    ang, revs, pos = process_counts(virtual_cnt)
    print(f"Counts: {virtual_cnt} -> Ang: {ang:5.2f} | Revs: {revs:2d} | Joint: {pos:5.2f}")

print("\n--- POSITIVE ROTATION (Counts DOWN) ---")
last_motor_angle = -1.0
motor_revolutions = 0
for cnt in range(4095, -1000, -500):
    virtual_cnt = cnt % 4096
    ang, revs, pos = process_counts(virtual_cnt)
    print(f"Counts: {virtual_cnt} -> Ang: {ang:5.2f} | Revs: {revs:2d} | Joint: {pos:5.2f}")
