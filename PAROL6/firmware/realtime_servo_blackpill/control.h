/*
 * PAROL6 Control — TIM11 ISR at 500 Hz, polls PWM Input mode registers
 *
 * The control loop reads encoder duty cycles directly from timer CCR registers
 * (no interrupt, no DMA). PWM Input mode handles all edge timing in hardware.
 */

#ifndef CONTROL_H
#define CONTROL_H

#include "config.h"

typedef struct {
    float desired_position;
    float desired_velocity;
    float actual_position;
    float actual_velocity;
    float position_error;
    float velocity_command;
    float total_motor_angle;
    float last_motor_angle;
} JointState;

void controlInit(void);
void controlSetCommand(uint8_t joint_idx, float position, float velocity);
void controlUpdate(void);
const JointState* controlGetState(uint8_t joint_idx);
void controlArm(void);
bool controlIsArmed(void);

#endif // CONTROL_H
