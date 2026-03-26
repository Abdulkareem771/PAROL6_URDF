#include <stdio.h>
#include <math.h>

#define PI 3.1415926535f

int main() {
    float last_motor_angle = -1.0;
    int motor_revolutions = 0;
    float dir_sign = -1.0; // J5
    
    printf("--- NEGATIVE ROTATION (Counts UP) ---\n");
    // Simulate counts going down (meaning motor angle goes UP physically)
    for (int cnt = -500; cnt <= 5000; cnt += 200) {
        // Virtualize counts back to 0-4095
        int virtual_cnt = cnt;
        while (virtual_cnt < 0) virtual_cnt += 4096;
        virtual_cnt %= 4096;
        
        float motor_ang = (virtual_cnt / 4096.0f) * 2.0f * PI;
        motor_ang *= dir_sign; 
        
        // Normalize
        while (motor_ang < 0.0f) motor_ang += 2.0f * PI;
        while (motor_ang >= 2.0f * PI) motor_ang -= 2.0f * PI;
        
        if (last_motor_angle < 0.0f) last_motor_angle = motor_ang;
        
        float delta = motor_ang - last_motor_angle;
        if (delta >  PI) motor_revolutions--;
        if (delta < -PI) motor_revolutions++;
        
        last_motor_angle = motor_ang;
        
        float total = motor_ang + motor_revolutions * 2.0f * PI;
        float joint_pos = total / 10.0f; // Gear ratio 10
        
        printf("Ang: %5.2f | Delta: %5.2f | Revs: %2d | Joint Pos: %5.2f\n", motor_ang, delta, motor_revolutions, joint_pos);
    }

    printf("\n--- POSITIVE ROTATION (Counts DOWN) ---\n");
    last_motor_angle = -1.0;
    motor_revolutions = 0;
    for (int cnt = 4500; cnt >= -1000; cnt -= 200) {
        // Virtualize counts back to 0-4095
        int virtual_cnt = cnt;
        while (virtual_cnt < 0) virtual_cnt += 4096;
        virtual_cnt %= 4096;
        
        float motor_ang = (virtual_cnt / 4096.0f) * 2.0f * PI;
        motor_ang *= dir_sign; 
        
        // Normalize
        while (motor_ang < 0.0f) motor_ang += 2.0f * PI;
        while (motor_ang >= 2.0f * PI) motor_ang -= 2.0f * PI;
        
        if (last_motor_angle < 0.0f) last_motor_angle = motor_ang;
        
        float delta = motor_ang - last_motor_angle;
        if (delta >  PI) motor_revolutions--;
        if (delta < -PI) motor_revolutions++;
        
        last_motor_angle = motor_ang;
        
        float total = motor_ang + motor_revolutions * 2.0f * PI;
        float joint_pos = total / 10.0f; // Gear ratio 10
        
        printf("Ang: %5.2f | Delta: %5.2f | Revs: %2d | Joint Pos: %5.2f\n", motor_ang, delta, motor_revolutions, joint_pos);
    }
    return 0;
}
