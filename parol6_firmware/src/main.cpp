#include <Arduino.h>

#include "transport/SerialTransport.h"
#include "safety/Supervisor.h"
#include "observer/AlphaBetaFilter.h"
#include "control/Interpolator.h"
#include "control/Interpolator.h"
#include "hal/PwmCaptureEncoder.h"
// -------------------------------------------------------------------------
// Global Architecture Instantiation
// -------------------------------------------------------------------------

static const int NUM_AXES = 6;
#define ISR_PROFILER_PIN 13 // Standard LED Pin on Teensy for oscilloscope hookup

// RTOS/Main Loop Transport
CircularBuffer<RosCommand, 20> rx_queue;
SerialTransport transport;
SafetySupervisor supervisor;

// 1 kHz ISR Mathematical Core
AlphaBetaFilter observer[NUM_AXES] = {
    AlphaBetaFilter(0.85f, 0.05f, 0.001f),
    AlphaBetaFilter(0.85f, 0.05f, 0.001f),
    AlphaBetaFilter(0.85f, 0.05f, 0.001f),
    AlphaBetaFilter(0.85f, 0.05f, 0.001f),
    AlphaBetaFilter(0.85f, 0.05f, 0.001f),
    AlphaBetaFilter(0.85f, 0.05f, 0.001f)
};

LinearInterpolator interpolator[NUM_AXES];

// Hardware Abstraction (Phase 1.5: 6 physical PWM input pins)
PwmCaptureEncoder encoder_hal[NUM_AXES] = {
    PwmCaptureEncoder(2), PwmCaptureEncoder(3), PwmCaptureEncoder(4),
    PwmCaptureEncoder(5), PwmCaptureEncoder(6), PwmCaptureEncoder(7)
};

// Global ISR wrappers for attachInterrupt since class methods can't be passed directly
void isr_enc0() { encoder_hal[0].handle_interrupt(); }
void isr_enc1() { encoder_hal[1].handle_interrupt(); }
void isr_enc2() { encoder_hal[2].handle_interrupt(); }
void isr_enc3() { encoder_hal[3].handle_interrupt(); }
void isr_enc4() { encoder_hal[4].handle_interrupt(); }
void isr_enc5() { encoder_hal[5].handle_interrupt(); }

IntervalTimer controlTimer;
volatile uint32_t system_tick_ms = 0;

// Fake Motor Output HAL for Phase 1 compilation
void set_motor_velocity(int axis, float velocity) {
    // Write physical PWM to motor driver
    // motor_hal[axis].set_pwm(val);
}

// -------------------------------------------------------------------------
// 1 kHz Hardware Timer ISR (Strict Real-Time Execution)
// -------------------------------------------------------------------------
void run_control_loop_isr() {
    digitalWriteFast(ISR_PROFILER_PIN, HIGH);
    
    float current_velocities[NUM_AXES]; 
    float current_positions[NUM_AXES];
    float commanded_velocities[NUM_AXES];
    
    // Strict Control Law: Kp Proportional Gains
    const float Kp[NUM_AXES] = { 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f };

    // 1. Compute Math for All Axes First
    for (int i = 0; i < NUM_AXES; i++) {
        // A. Read Hardware Abstraction Layer
        float raw_pos = encoder_hal[i].read_angle();
        
        // B. Update Observer
        observer[i].update(raw_pos);
        float actual_pos = observer[i].get_position();
        float actual_vel = observer[i].get_velocity();
        current_positions[i] = actual_pos;
        current_velocities[i] = actual_vel;
        
        // C. Get 1ms Interpolated Setpoint
        float cmd_pos, cmd_vel_ff;
        interpolator[i].tick_1ms(cmd_pos, cmd_vel_ff);
        
        // D. Strict Control Law (P + FF)
        float pos_error = cmd_pos - actual_pos;
        float velocity_command = cmd_vel_ff + (Kp[i] * pos_error);
        
        // E. Output Saturation (Clamping)
        const float MAX_VEL_CMD = 15.0f;
        if (velocity_command > MAX_VEL_CMD) velocity_command = MAX_VEL_CMD;
        if (velocity_command < -MAX_VEL_CMD) velocity_command = -MAX_VEL_CMD;
        
        // Cache safe command
        commanded_velocities[i] = velocity_command;
    }
    
    // 2. Safety check uses observer velocities and unified monotonic tick
    system_tick_ms++;
    supervisor.update(system_tick_ms, current_velocities);

    // 3. Motor Hal Output safely governed by Supervisor
    for(int i = 0; i < NUM_AXES; i++) {
        if (supervisor.is_safe()) {
            set_motor_velocity(i, commanded_velocities[i]);
        } else {
            set_motor_velocity(i, 0.0f); // Fast stop on fault
        }
    }
    
    digitalWriteFast(ISR_PROFILER_PIN, LOW);
}

// -------------------------------------------------------------------------
// Main Thread 
// -------------------------------------------------------------------------
void setup() {
    pinMode(ISR_PROFILER_PIN, OUTPUT);
    digitalWriteFast(ISR_PROFILER_PIN, LOW);
    
    transport.init(115200);
    supervisor.init(0);

    for (int i = 0; i < NUM_AXES; i++) {
        encoder_hal[i].init();
        observer[i].set_initial_position(0.0f);
        interpolator[i].reset(0.0f);
    }
    
    // Attach Hardware Interrupts for PWM edge capture
    attachInterrupt(digitalPinToInterrupt(2), isr_enc0, CHANGE);
    attachInterrupt(digitalPinToInterrupt(3), isr_enc1, CHANGE);
    attachInterrupt(digitalPinToInterrupt(4), isr_enc2, CHANGE);
    attachInterrupt(digitalPinToInterrupt(5), isr_enc3, CHANGE);
    attachInterrupt(digitalPinToInterrupt(6), isr_enc4, CHANGE);
    attachInterrupt(digitalPinToInterrupt(7), isr_enc5, CHANGE);
    
    // Start hardware timer at precisely 1000 microseconds (1 kHz)
    controlTimer.begin(run_control_loop_isr, 1000);
}

void loop() {
    // 1. Process UART characters non-blocking
    transport.process_incoming(rx_queue);
    
    // 2. Drain validated commands to interpolators
    while (!rx_queue.isEmpty()) {
        RosCommand cmd;
        noInterrupts();
        cmd = rx_queue.shift();
        interrupts();

        supervisor.feed_watchdog(system_tick_ms);

        // Dynamically compute the duration since the last valid ROS packet
        static uint32_t last_cmd_ts = 0;
        uint32_t delta_ms = 40; // Default fallback (25Hz)
        if (last_cmd_ts != 0 && system_tick_ms > last_cmd_ts) {
            delta_ms = system_tick_ms - last_cmd_ts;
            // Cap the duration to prevent massive interpolation swings if a packet drops
            if (delta_ms > 100) delta_ms = 100;
        }
        last_cmd_ts = system_tick_ms;

        for (int i = 0; i < NUM_AXES; i++) {
            interpolator[i].set_target(cmd.positions[i], cmd.velocities[i], delta_ms); 
        }
    }

    // 3. Provide background status telemetry (10 Hz roughly)
    static uint32_t last_print = 0;
    if (system_tick_ms - last_print > 100) {
        last_print = system_tick_ms;
        // Feedback data gathering requires brief interrupt lock to prevent tearing
        float pos[NUM_AXES], vel[NUM_AXES];
        noInterrupts();
        for (int i=0; i<NUM_AXES; i++) {
            pos[i] = observer[i].get_position();
            vel[i] = observer[i].get_velocity();
        }
        interrupts();
        
        static uint32_t seq = 0;
        transport.send_feedback(seq++, pos, vel);
    }
}
