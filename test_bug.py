import math

PI = math.pi
ENCODER_RESOLUTION = 4096.0
GEAR_RATIO = 10.0
ENCODER_DIR_SIGN = -1.0
ENCODER_OFFSET = 0.0

last_motor_angle = -1.0
motor_revolutions = 0

def read_encoder_sim(pw):
    global last_motor_angle, motor_revolutions
    if pw < 2 or pw > 1050: return None
        
    clocks = pw / 0.25
    counts = clocks - 16.0
    if counts < 0.0: counts = 0.0
    if counts >= ENCODER_RESOLUTION: counts = ENCODER_RESOLUTION - 1.0
    
    motor_ang = (counts / ENCODER_RESOLUTION) * 2.0 * PI
    motor_ang *= ENCODER_DIR_SIGN
    motor_ang += ENCODER_OFFSET
    
    motor_ang = math.fmod(motor_ang, 2.0 * PI)
    if motor_ang < 0.0: motor_ang += 2.0 * PI
        
    if last_motor_angle < 0.0:
        last_motor_angle = motor_ang
        total = motor_ang + motor_revolutions * 2.0 * PI
        return total / GEAR_RATIO, motor_ang, 0.0, 0
        
    delta = motor_ang - last_motor_angle
    
    rev_change = 0
    if delta > PI: 
        motor_revolutions -= 1
        rev_change = -1
    if delta < -PI: 
        motor_revolutions += 1
        rev_change = 1
        
    last_motor_angle = motor_ang
    
    total = motor_ang + motor_revolutions * 2.0 * PI
    return total / GEAR_RATIO, motor_ang, delta, rev_change

def generate_true_pw(motor_ang_true):
    ang_norm = math.fmod(motor_ang_true, 2.0*PI)
    if ang_norm < 0.0: ang_norm += 2.0*PI
    counts = -ang_norm / (2.0 * PI) * 4096.0
    counts = math.fmod(counts, 4096.0)
    if counts < 0.0: counts += 4096.0
    clocks = counts + 16.0
    pw = clocks * 0.25
    return pw

positions = []
motor_ang_true = 0.0

# Simulate going from 0 to -17.45 (100 degrees joint * 10 gear)
target = -17.45
for step in range(500):
    motor_ang_true += (target - 0.0) / 500.0
    true_pw = generate_true_pw(motor_ang_true)
    res = read_encoder_sim(true_pw)
    if res is not None:
        raw_joint, mang, delta, rchange = res
        positions.append(raw_joint)
        if rchange != 0:
            print(f"WRAP: Step {step} | True={motor_ang_true:.3f} | PW={true_pw:.1f} | m_ang={mang:.3f} | D={delta:.3f} | Joint={raw_joint:.3f} | Revs={motor_revolutions}")
            
print(f"Final Simulated Joint Position: {positions[-1]:.3f} rad = {positions[-1]*180/PI:.1f} deg")
