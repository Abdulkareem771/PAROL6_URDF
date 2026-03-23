import math

def simulate_teensy_algo(raw_pws):
    # Same logic as control.cpp
    history = [0, 0, 0]
    fill = 0
    
    last_motor_angle = -1.0
    motor_revolutions = 0
    
    results = []
    
    for raw_pw in raw_pws:
        # Hardware sanity check
        if raw_pw < 2 or raw_pw > 1050:
            continue
            
        # Median filter
        slot = fill if fill < 3 else (fill % 3)
        history[slot] = raw_pw
        fill += 1
        if fill > 200: fill = 3
        
        if fill >= 3:
            s = sorted(history[:3])
            pw = s[1]
        else:
            pw = raw_pw
            
        # Calc angle
        clocks = pw / (250.0 / 1000.0)
        counts = clocks - 16.0
        if counts < 0: counts = 0
        if counts >= 4096: counts = 4095
        
        motor_ang = (counts / 4096.0) * 2.0 * math.pi
        
        # Invert sign for J5
        motor_ang *= -1.0
        
        while motor_ang < 0.0: motor_ang += 2.0 * math.pi
        while motor_ang >= 2.0 * math.pi: motor_ang -= 2.0 * math.pi
        
        # Multi-turn
        if last_motor_angle < 0.0:
            last_motor_angle = motor_ang
            
        delta = motor_ang - last_motor_angle
        if delta > math.pi: motor_revolutions -= 1
        if delta < -math.pi: motor_revolutions += 1
        last_motor_angle = motor_ang
        
        total = motor_ang + motor_revolutions * 2.0 * math.pi
        results.append(total)
        
    return results

# generate noisy alternating pattern:
pws = [4] * 10
# add noise pattern repeatedly
for _ in range(100):
   pws.extend([4, 1004, 1004, 4, 1004])

out = simulate_teensy_algo(pws)
print("Start value:", out[0], "End value:", out[-1])
print("Revolutions drifted by", (out[-1] - out[0])/(2*math.pi))
