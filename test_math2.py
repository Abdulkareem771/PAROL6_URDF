import math

PI = math.pi
revs = 0
last_angle = -1.0

# J5 has GEAR ratio 10 and SIGN -1
# Let's see what happens to the math when the encoder counts go from
# 4000 -> 0 -> 4000 (which is a physical rotation in the positive joint direction due to SIGN -1)

# we will feed the raw PWM fractions directly
fractions = [
    4000.0/4096.0,
    100.0/4096.0,
    3900.0/4096.0,
    200.0/4096.0
]

for process in range(10): # simulate multiple rotations
    for frac in fractions:
        
        # 1. Calc motor angle
        motor_ang = frac * 2.0 * PI
        
        # 2. Invert sign
        motor_ang *= -1.0
        
        # 3. Normalize
        while motor_ang < 0.0: motor_ang += 2.0 * PI
        while motor_ang >= 2.0 * PI: motor_ang -= 2.0 * PI
        
        # 4. Multiturn
        if last_angle < 0.0:
            last_angle = motor_ang
            
        delta = motor_ang - last_angle
        # print(f"Raw Delta: {delta:.2f}")
        
        if delta > PI: revs -= 1
        if delta < -PI: revs += 1
        last_angle = motor_ang
        
        # 5. Joint output
        total = motor_ang + revs * 2.0 * PI
        joint = total / 10.0
        
        print(f"Frac: {frac:.3f} | r_ang: {motor_ang:5.2f} | D: {delta:5.2f} | Revs: {revs:2d} | Joint: {joint:5.2f}")
