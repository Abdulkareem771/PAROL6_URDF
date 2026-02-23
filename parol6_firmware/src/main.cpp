#include <Arduino.h>

#include "transport/SerialTransport.h"
#include "safety/Supervisor.h"
#include "observer/AlphaBetaFilter.h"
#include "control/Interpolator.h"
#include "hal/MicrosEncoder.h"

// -------------------------------------------------------------------------
// Global Architecture Instantiation
// -------------------------------------------------------------------------

static const int NUM_AXES = 6;

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

// Hardware Abstraction
MicrosEncoder encoder_hal[NUM_AXES] = {
    MicrosEncoder(0, 0), MicrosEncoder(0, 0), MicrosEncoder(0, 0),
    MicrosEncoder(0, 0), MicrosEncoder(0, 0), MicrosEncoder(0, 0)
};

IntervalTimer controlTimer;

// Fake Motor Output HAL for Phase 1 compilation
void set_motor_velocity(int axis, float velocity) {
    // Write physical PWM to motor driver
    // motor_hal[axis].set_pwm(val);
}

// -------------------------------------------------------------------------
// 1 kHz Hardware Timer ISR (Strict Real-Time Execution)
// -------------------------------------------------------------------------
void run_control_loop_isr() {
    float current_velocities[NUM_AXES]; 
    float current_positions[NUM_AXES];
    
    // Strict Control Law: Kp Proportional Gains
    const float Kp[NUM_AXES] = { 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f };

    for (int i = 0; i < NUM_AXES; i++) {
        // 1. Read Hardware Abstraction Layer
        float raw_pos = encoder_hal[i].read_angle();
        
        // 2. Update Observer
        observer[i].update(raw_pos);
        float actual_pos = observer[i].get_position();
        float actual_vel = observer[i].get_velocity();
        current_positions[i] = actual_pos;
        current_velocities[i] = actual_vel;
        
        // 3. Get 1ms Interpolated Setpoint
        float cmd_pos, cmd_vel_ff;
        interpolator[i].tick_1ms(cmd_pos, cmd_vel_ff);
        
        // 4. Strict Control Law (P + FF)
        float pos_error = cmd_pos - actual_pos;
        float velocity_command = cmd_vel_ff + (Kp[i] * pos_error);
        
        // 5. Motor Hal Output
        if (supervisor.get_state() == SafetySupervisor::NOMINAL) {
            set_motor_velocity(i, velocity_command);
        } else {
            set_motor_velocity(i, 0.0f); // Fast stop on fault
        }
    }
    
    // Safety check uses observer velocities
    supervisor.update(millis(), current_velocities);
}

// -------------------------------------------------------------------------
// Main Thread 
// -------------------------------------------------------------------------
void setup() {
    transport.init(115200);
    supervisor.init(millis());

    for (int i = 0; i < NUM_AXES; i++) {
        encoder_hal[i].init();
        observer[i].set_initial_position(0.0f);
        interpolator[i].reset(0.0f);
    }
    
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

        supervisor.feed_watchdog(millis());

        for (int i = 0; i < NUM_AXES; i++) {
            // Wait expected ROS delta is roughly 40ms (25Hz)
            interpolator[i].set_target(cmd.positions[i], cmd.velocities[i], 40); 
        }
    }

    // 3. Provide background status telemetry (10 Hz roughly)
    static uint32_t last_print = 0;
    if (millis() - last_print > 100) {
        last_print = millis();
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
