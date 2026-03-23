#include <iostream>
#include <cmath>

int main() {
    float last_motor_angle = 0.0f;
    int motor_revolutions = 0;
    
    // Simulate weird jumps that might cause a runaway in ONE direction
    // If the angle jumps from 0 to 4 to 0 ?
    float angles[] = {0.0, 4.0, 0.0, 4.0, 0.0, 4.0};
    
    for(float motor_ang : angles) {
        float delta = motor_ang - last_motor_angle;
        if (delta >  M_PI) motor_revolutions--;
        if (delta < -M_PI) motor_revolutions++;
        last_motor_angle = motor_ang;
        
        float total = motor_ang + motor_revolutions * 2.0f * M_PI;
        std::cout << "ang: " << motor_ang << "\t delta: " << delta << "\t revs: " << motor_revolutions << "\t total: " << total << std::endl;
    }
    return 0;
}
