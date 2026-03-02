#pragma once

#include <stdint.h>
#include <math.h>

class SafetySupervisor {
public:
    enum State { INIT, NOMINAL, SOFT_ESTOP, FAULT };
    
    SafetySupervisor() : current_state_(INIT), last_cmd_time_ms_(0) {}

    void init(uint32_t current_time_ms) {
        last_cmd_time_ms_ = current_time_ms;
        current_state_ = NOMINAL;
    }

    void update(uint32_t current_time_ms, const float actual_velocities[6]) {
        // 1. Check Watchdog (Command Timeout)
        if (current_time_ms - last_cmd_time_ms_ > COMMAND_TIMEOUT_MS) {
            trigger_fault(SOFT_ESTOP, "Command Timeout");
        }
        
        // 2. Check Kinematic Limits (Runaway Velocity) — per-joint validated limits
        for (int i = 0; i < 6; i++) {
            if (fabs(actual_velocities[i]) > MAX_SAFE_VELOCITY_PER_JOINT[i]) {
                trigger_fault(FAULT, "Runaway Velocity");
            }
        }
    }
    
    void feed_watchdog(uint32_t time_ms) {
        last_cmd_time_ms_ = time_ms;
        // NOTE: Auto-recovery from SOFT_ESTOP is intentionally NOT done here.
        // Recovery is only allowed after a full state reset (explicit operator action),
        // to prevent a reconnecting cable from causing an uncontrolled lurch.
        // Ticket: implement a zero-velocity handshake for safe recovery.
    }

    bool is_safe() const { 
        return current_state_ == NOMINAL; 
    }
    
    State get_state() const { 
        return current_state_; 
    }
    
private:
    State current_state_;
    uint32_t last_cmd_time_ms_;
    const uint32_t COMMAND_TIMEOUT_MS = 200; // 5 missed 25Hz packets
    // Per-joint velocity limits validated from realtime_servo_teensy/config.h
    // {J1, J2, J3, J4, J5, J6} in rad/s
    const float MAX_SAFE_VELOCITY_PER_JOINT[6] = {3.0f, 3.0f, 6.0f, 6.0f, 6.0f, 6.0f};
    // Legacy global limit kept for reference (was incorrectly applied as a single global)
    // const float MAX_SAFE_VELOCITY_RAD_S = 10.0f; 
    
    void trigger_fault(State state, const char* reason) { 
        // Latch hard faults
        if (current_state_ != FAULT) { 
            current_state_ = state;
            // Immediate serial print is bad normally, but allowable for hard ESTOP panic.
            // On a strict implementation, we would queue this string for the transport layer.
            // Keeping it simple for the initial prototype.
        }
    }
};
