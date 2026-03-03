#include <Arduino.h>

// GUI-generated configuration — edit via the Firmware Configurator, not here.
#if __has_include("../generated/config.h")
#  include "../generated/config.h"
#else
// Defaults when no config.h has been generated yet
#  define TRANSPORT_MODE          1   // 1=USB_CDC
#  define CONTROL_LOOP_PERIOD_US  1000
#  define FEEDBACK_RATE_HZ        10
#  define VELOCITY_DEADBAND_RAD_S 0.02f
#endif

#include "transport/SerialTransport.h"
#include "safety/Supervisor.h"
#include "observer/AlphaBetaFilter.h"
#include "control/Interpolator.h"
#include "hal/QuadTimerEncoder.h"
#include "hal/FlexPWMGenerator.h"
#include "hal/ActuatorModel.h"
// -------------------------------------------------------------------------
// Global Architecture Instantiation
// -------------------------------------------------------------------------

#ifndef NUM_AXES
#define NUM_AXES 6
#endif
#define ISR_PROFILER_PIN 13 // Standard LED Pin on Teensy for oscilloscope hookup

// RTOS/Main Loop Transport
CircularBuffer<RosCommand, 20> rx_queue;
SerialTransport transport;
SafetySupervisor supervisor;

// 1 kHz ISR Mathematical Core
// Observer initialized in setup() with per-joint gains from config.h (AB_ALPHA / AB_BETA).
// Placeholder construction here uses dt=0.001 only; gains are overridden in setup().
AlphaBetaFilter observer[NUM_AXES] = {
    AlphaBetaFilter(0.1f, 0.005f, 0.001f),
    AlphaBetaFilter(0.1f, 0.005f, 0.001f),
    AlphaBetaFilter(0.1f, 0.005f, 0.001f),
    AlphaBetaFilter(0.1f, 0.005f, 0.001f),
    AlphaBetaFilter(0.1f, 0.005f, 0.001f),
    AlphaBetaFilter(0.1f, 0.005f, 0.001f)
};

LinearInterpolator interpolator[NUM_AXES];

// Telemetry Export for Main Loop
volatile float telemetry_pos[NUM_AXES];
volatile float telemetry_vel[NUM_AXES];

// Hardware Abstraction (Phase 3: Zero-interrupt QuadTimers)
QuadTimerEncoder encoder_hal[NUM_AXES] = {
    QuadTimerEncoder(10), QuadTimerEncoder(11), QuadTimerEncoder(12),
    QuadTimerEncoder(14), QuadTimerEncoder(15), QuadTimerEncoder(18)
};

// Phase 4 Stage 2: Per-axis stepper drivers (STEP + DIR)
// STEP pins: Zone 2 FlexPWM-capable [2, 6, 7, 8, 4, 5]
// DIR  pins: Zone 3 pure GPIO        [30,31,32,33,34,35]
FlexPWMGenerator stepper[NUM_AXES] = {
    FlexPWMGenerator(STEP_PINS[0], DIR_PINS[0]),  // J1: step=2,  dir=30
    FlexPWMGenerator(STEP_PINS[1], DIR_PINS[1]),  // J2: step=6,  dir=31
    FlexPWMGenerator(STEP_PINS[2], DIR_PINS[2]),  // J3: step=7,  dir=32
    FlexPWMGenerator(STEP_PINS[3], DIR_PINS[3]),  // J4: step=8,  dir=33
    FlexPWMGenerator(STEP_PINS[4], DIR_PINS[4]),  // J5: step=4,  dir=34
    FlexPWMGenerator(STEP_PINS[5], DIR_PINS[5]),  // J6: step=5,  dir=35
};

// Phase 4 Stage 2: Kinematic models — converts rad/s -> step_freq + DIR
// Gear ratios and direction signs sourced from STM32 legacy motor_init.cpp
ActuatorModel actuator[NUM_AXES] = {
    ActuatorModel::create_joint(0),  // J1
    ActuatorModel::create_joint(1),  // J2
    ActuatorModel::create_joint(2),  // J3
    ActuatorModel::create_joint(3),  // J4
    ActuatorModel::create_joint(4),  // J5
    ActuatorModel::create_joint(5),  // J6
};

IntervalTimer controlTimer;
volatile uint32_t system_tick_ms = 0;

// Software Profiler tracking (Phase 1.5)
volatile uint32_t max_isr_time_us = 0;
volatile uint32_t last_isr_time_us = 0;

// Motor Output HAL — converts ControlLaw velocity (rad/s) into STEP frequency + DIR
void set_motor_velocity(int axis, float velocity_rad_s) {
#ifdef JOINT_ENABLED
    // Honour per-joint enable flag from GUI config.h
    if (!JOINT_ENABLED[axis]) {
        stepper[axis].stop();
        return;
    }
#endif
    float freq_hz;
    bool  forward;
    actuator[axis].compute(velocity_rad_s, freq_hz, forward);
    // DIR must be set BEFORE frequency on every call (Servo42C setup time ≥200 ns)
    stepper[axis].set_direction(forward);
    if (actuator[axis].should_stop(freq_hz)) {
        stepper[axis].stop();
    } else {
        stepper[axis].set_frequency(freq_hz);
    }
}

// -------------------------------------------------------------------------
// 1 kHz Hardware Timer ISR (Strict Real-Time Execution)
// -------------------------------------------------------------------------
void run_control_loop_isr() {
    uint32_t start_cycles = ARM_DWT_CYCCNT; // Start cycle profiling
    digitalWriteFast(ISR_PROFILER_PIN, HIGH);
    
    float current_velocities[NUM_AXES]; 
    float current_positions[NUM_AXES];
    float commanded_velocities[NUM_AXES];

#ifdef KP_GAINS
    // Use GUI-configured per-joint gains and limits
    static const float* Kp         = KP_GAINS;
    static const float* MAX_VEL_CMD = MAX_VEL_RAD_S;
#else
    // Fallback defaults if no config.h
    const float Kp[NUM_AXES]          = { 5.0f,  5.0f,  2.0f, 2.0f, 2.0f, 2.0f };
    const float MAX_VEL_CMD[NUM_AXES] = { 3.0f,  3.0f,  6.0f, 6.0f, 6.0f, 6.0f };
#endif

    // 1. Compute Math for All Axes First
    for (int i = 0; i < NUM_AXES; i++) {
        // A. Read Hardware Abstraction Layer
        float raw_pos = encoder_hal[i].read_angle();
        
#if defined(FEATURE_ALPHABETA_FILTER) && FEATURE_ALPHABETA_FILTER == 1
        // B. Update Observer
        observer[i].update(raw_pos);
        float actual_pos = observer[i].get_position();
        float actual_vel = observer[i].get_velocity();
#else
        // Passthrough if observer disabled (noisy!)
        float actual_pos = raw_pos;
        float actual_vel = 0.0f; // Can't reliably derive velocity without a filter
#endif
        current_positions[i] = actual_pos;
        current_velocities[i] = actual_vel;
        
        // --- Gear ratio scaling: convert motor-shaft angle → joint-space angle ---
        // QuadTimerEncoder reads raw PWM duty-cycle → 0..2π per MOTOR revolution.
        // MoveIt and the ROS hardware interface expect JOINT-space radians.
        // Dividing by GEAR_RATIOS[i] maps motor revolutions to joint displacement.
        // This is required for correct bounds checking in parol6_system.cpp and
        // for the control law pos_error to be in the same units as the ROS command.
#ifdef NUM_AXES
        float joint_pos = actual_pos / GEAR_RATIOS[i];
        float joint_vel = actual_vel / GEAR_RATIOS[i];
#else
        static const float fallback_gear[6] = {6.4f, 20.0f, 18.0952381f, 4.0f, 4.0f, 10.0f};
        float joint_pos = actual_pos / fallback_gear[i];
        float joint_vel = actual_vel / fallback_gear[i];
#endif
        // Export JOINT-SPACE values to telemetry buffer
        telemetry_pos[i] = joint_pos;
        telemetry_vel[i] = joint_vel;
        
        // C. Get Interpolated Setpoint (already in joint space from ROS command)
        float cmd_pos, cmd_vel_ff;
        interpolator[i].tick_1ms(cmd_pos, cmd_vel_ff);
        
        // D. Control Law (P + FF) — error computed in joint space (both sides now match)
        float pos_error = cmd_pos - joint_pos;
        float velocity_command = (Kp[i] * pos_error);
#if defined(FEATURE_VEL_FEEDFORWARD) && FEATURE_VEL_FEEDFORWARD == 1
        velocity_command += cmd_vel_ff;
#endif
        
        // E. Per-joint output saturation
        if (velocity_command >  MAX_VEL_CMD[i]) velocity_command =  MAX_VEL_CMD[i];
        if (velocity_command < -MAX_VEL_CMD[i]) velocity_command = -MAX_VEL_CMD[i];

        // F. Velocity deadband (configured in GUI or default)
        if (fabsf(velocity_command) < VELOCITY_DEADBAND_RAD_S) velocity_command = 0.0f;
        
        commanded_velocities[i] = velocity_command;
    }
    
    // 2. Safety check uses observer velocities and unified monotonic tick
    system_tick_ms++;
#if defined(FEATURE_SAFETY_SUPERVISOR) && FEATURE_SAFETY_SUPERVISOR == 1    
    supervisor.update(system_tick_ms, current_velocities);

    // 3. Motor Hal Output safely governed by Supervisor
    for(int i = 0; i < NUM_AXES; i++) {
        if (supervisor.is_safe()) {
            set_motor_velocity(i, commanded_velocities[i]);
        } else {
            set_motor_velocity(i, 0.0f); // Fast stop on fault
        }
    }
#else
    // Supervisor disabled - raw passthrough
    for(int i = 0; i < NUM_AXES; i++) {
        set_motor_velocity(i, commanded_velocities[i]);
    }
#endif
    
    digitalWriteFast(ISR_PROFILER_PIN, LOW);
    
    // Calculate profiling metrics
    uint32_t end_cycles = ARM_DWT_CYCCNT;
    uint32_t cycles_taken = end_cycles - start_cycles;
    uint32_t time_taken_us = cycles_taken / (F_CPU / 1000000); // 600 MHz = 600 cycles/us
    
    last_isr_time_us = time_taken_us;
    if (time_taken_us > max_isr_time_us) {
        max_isr_time_us = time_taken_us;
    }
}

// -------------------------------------------------------------------------
// Main Thread 
// -------------------------------------------------------------------------
void setup() {
    pinMode(ISR_PROFILER_PIN, OUTPUT);
    digitalWriteFast(ISR_PROFILER_PIN, LOW);
    
    // Enable ARM Cycle Counter for precise software profiling
    ARM_DEMCR |= ARM_DEMCR_TRCENA;
    ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;
    
    transport.init(115200);
    supervisor.init(0);
    
    // Initialize FlexPWM stepper drivers BEFORE encoders to prevent CCM clock gating conflicts
    for (int i = 0; i < NUM_AXES; i++) {
        stepper[i].init();    // Configures STEP pin FlexPWM + DIR pin GPIO
        delay(2);             // Allow FlexPWM peripheral to settle before QuadTimer init
    }

    // Hardware Configuration
    for (int i = 0; i < NUM_AXES; i++) {
        encoder_hal[i].init();
        delay(5); // Give encoder time to stabilize
        
        float initial_motor_pos = encoder_hal[i].read_angle();
        observer[i].set_initial_position(initial_motor_pos);
        // Seed interpolator and telemetry in JOINT space, not motor space.
        // The control law computes error as (cmd_pos - joint_pos) where joint_pos = motor_pos / gear.
        // If interpolator is seeded with motor_pos, the first ISR tick produces a massive position
        // error = motor_pos * (1 - 1/gear_ratio), causing a violent startup jerk.
#ifdef NUM_AXES
        float initial_joint_pos = initial_motor_pos / GEAR_RATIOS[i];
#else
        const float fallback_g[6] = {6.4f, 20.0f, 18.0952381f, 4.0f, 4.0f, 10.0f};
        float initial_joint_pos = initial_motor_pos / fallback_g[i];
#endif
        interpolator[i].reset(initial_joint_pos);
        telemetry_pos[i] = initial_joint_pos;
        telemetry_vel[i] = 0.0f;
    }
    
    // No software interrupts needed for Phase 3! (Zero-CPU QuadTimer capture)
    
    // Control loop rate from GUI config (default 1000µs = 1kHz)
    controlTimer.begin(run_control_loop_isr, CONTROL_LOOP_PERIOD_US);
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

        uint32_t current_tick = __atomic_load_n(&system_tick_ms, __ATOMIC_RELAXED);
#if defined(FEATURE_VALIDATE_COMMANDS) && FEATURE_VALIDATE_COMMANDS == 1        
        // Optionally validate command sequence here if feature added later
#endif
#if defined(FEATURE_WATCHDOG) && FEATURE_WATCHDOG == 1        
        supervisor.feed_watchdog(current_tick);
#endif

        // Dynamically compute the duration since the last valid ROS packet
        static uint32_t last_cmd_ts = 0;
        uint32_t delta_ms = 40; // Default fallback (25Hz)
        if (last_cmd_ts != 0 && current_tick > last_cmd_ts) {
            delta_ms = current_tick - last_cmd_ts;
            // Cap the duration to prevent massive interpolation swings if a packet drops
            if (delta_ms > 100) delta_ms = 100;
        }
        last_cmd_ts = current_tick;

        for (int i = 0; i < NUM_AXES; i++) {
            interpolator[i].set_target(cmd.positions[i], cmd.velocities[i], delta_ms); 
        }
    }

    // 3. Provide background status telemetry at FEEDBACK_RATE_HZ
    static const uint32_t FEEDBACK_INTERVAL_MS = 1000 / FEEDBACK_RATE_HZ;
    static uint32_t last_print = 0;
    uint32_t print_tick = __atomic_load_n(&system_tick_ms, __ATOMIC_RELAXED);
    if (print_tick - last_print > FEEDBACK_INTERVAL_MS) {
        last_print = print_tick;
        // Feedback data gathering requires brief interrupt lock to prevent tearing
        float pos[NUM_AXES], vel[NUM_AXES];
        noInterrupts();
        for (int i=0; i<NUM_AXES; i++) {
            pos[i] = telemetry_pos[i];
            vel[i] = telemetry_vel[i];
        }
        interrupts();
        
        
        static uint32_t seq = 0;
        transport.send_feedback(seq++, pos, vel);
        
        // Print Profiling Stats
        // Reset max_isr_time occasionally so we can see if spikes happen continuously or just once
        if (seq % 10 == 0) {
            // Serial.printf("ISR Profiler | Last: %lu us | Max: %lu us\n", last_isr_time_us, max_isr_time_us);
            max_isr_time_us = 0; // Reset peak tracker
        }
    }
}
