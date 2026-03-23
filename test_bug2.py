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
    
    # Safe bounds anti-glitch filter
    if abs(delta) > 1.0 and abs(delta) < 5.2:
        return None
        
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

def generate_true_pw(physical_motor_ang_rad):
    # Physical angle to counts
    # The physical shaft rotates.
    # If it rotates positively, counts decrease (because DIR_SIGN = -1)
    
    counts = -physical_motor_ang_rad / (2.0 * PI) * 4096.0
    counts = math.fmod(counts, 4096.0)
    if counts < 0.0: counts += 4096.0
    clocks = counts + 16.0
    pw = clocks * 0.25
    return pw

print("Simulating Target: 32 deg to -72 deg")
physical_ang = 32.0 * PI / 180.0 * GEAR_RATIO
target_ang = -72.0 * PI / 180.0 * GEAR_RATIO

# Set initial reference
pw = generate_true_pw(physical_ang)
read_encoder_sim(pw)

revs_expected = (physical_ang - target_ang) / (2.0 * PI)
print(f"Initial physical_ang={physical_ang:.3f} rad, target={target_ang:.3f} rad. Reqs {revs_expected:.2f} revs.")

positions = []
for step in range(10000):
    physical_ang += (target_ang - physical_ang) * 0.01  # PID-like decay towards target
    if abs(physical_ang - target_ang) < 0.01:
        break
        
    true_pw = generate_true_pw(physical_ang)
    res = read_encoder_sim(true_pw)
    if res is not None:
        raw_joint, mang, delta, rchange = res
        positions.append(raw_joint)
        if rchange != 0:
            print(f"WRAP: Step {step} | Physical={physical_ang:.3f} | PW={true_pw:.1f} | m_ang={mang:.3f} | D={delta:.3f} | Joint={raw_joint:.3f} | Revs={motor_revolutions}")
            
print(f"Final Simulated Joint Position: {positions[-1]:.3f} rad = {positions[-1]*180/PI:.1f} deg")
print(f"Final Internal Revolutions: {motor_revolutions}")
