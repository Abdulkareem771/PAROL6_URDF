#include <stdio.h>
#include <math.h>

int main() {
    float last_motor_angle = -1.0;
    int motor_revolutions = 0;
    float dir_sign = -1.0; // J5 inverted logic
    
    printf("--- J5 POSITIVE ROTATION (Counts DOWN due to invert) ---\n");
    for (int cnt = 4095; cnt >= -4096; cnt -= 500) {
        // Virtualize counts back to 0-4095
        int virtual_cnt = cnt;
        while (virtual_cnt < 0) virtual_cnt += 4096;
        virtual_cnt %= 4096;
        
        // 1. Convert to motor angle
        float motor_ang = (virtual_cnt / 4096.0f) * 2.0f * 3.14159f;
        
        // 2. Apply sign BEFORE normalize
        motor_ang *= dir_sign; 
        
        // 3. Normalize [0, 2pi)
        while (motor_ang < 0.0f) motor_ang += 2.0f * 3.14159f;
        while (motor_ang >= 2.0f * 3.14159f) motor_ang -= 2.0f * 3.14159f;
        
        // 4. Multi-turn tracking
        if (last_motor_angle < 0.0f) {
            last_motor_angle = motor_ang;
        }
        
        float delta = motor_ang - last_motor_angle;
        
        if (delta >  3.14159f) motor_revolutions--;
        if (delta < -3.14159f) motor_revolutions++;
        
        last_motor_angle = motor_ang;
        
        float total = motor_ang + motor_revolutions * 2.0f * 3.14159f;
        float joint_pos = total / 10.0f; // Gear ratio 10
        
        printf("Counts: %4d | Ang: %5.2f | Delta: %5.2f | Revs: %2d | Joint Pos: %5.2f\n", 
               virtual_cnt, motor_ang, delta, motor_revolutions, joint_pos);
    }
    return 0;
}
