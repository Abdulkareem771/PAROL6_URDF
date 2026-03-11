#pragma once

#include <stdint.h>
#include <math.h>

#ifndef COMMAND_TIMEOUT_MS
#define COMMAND_TIMEOUT_MS 200 // Default fallback: 5 missed 25Hz packets
#endif

class SafetySupervisor {
public:
    enum State { INIT, NOMINAL, SOFT_ESTOP, FAULT };
    
    SafetySupervisor() : current_state_(INIT), last_cmd_time_ms_(0) {
        fault_reason_[0] = '\0';
    }

    void init(uint32_t current_time_ms) {
        last_cmd_time_ms_ = current_time_ms;
        current_state_ = NOMINAL;
        fault_reason_[0] = '\0';
    }

    void update(uint32_t current_time_ms, const float joint_velocities[6],
                const float max_safe_joint_velocities[6]) {
        // 1. Check Watchdog (Command Timeout)
        if (current_time_ms - last_cmd_time_ms_ > COMMAND_TIMEOUT_MS) {
            trigger_fault(SOFT_ESTOP, "Command Timeout");
        }
        
        // 2. Check Kinematic Limits (Runaway Velocity) — per-joint validated limits
        for (int i = 0; i < 6; i++) {
            if (fabs(joint_velocities[i]) > max_safe_joint_velocities[i]) {
                trigger_fault(FAULT, "Runaway Velocity");
            }
        }
    }
    
    void feed_watchdog(uint32_t time_ms) {
        last_cmd_time_ms_ = time_ms;
        // NOTE: Auto-recovery from SOFT_ESTOP is intentionally NOT done here.
        // Recovery is only allowed after a full state reset (explicit operator action),
        // to prevent a reconnecting cable from causing an uncontrolled lurch.
    }

    /**
     * Call when a limit switch is asserted OUTSIDE of a homing sequence.
     * Transitions immediately to FAULT to prevent further motion.
     */
    void report_limit_switch(int axis) {
        char buf[32];
        snprintf(buf, sizeof(buf), "Limit: J%d", axis + 1);
        trigger_fault(FAULT, buf);
    }

    /**
     * Call when the HomingFSM reports a FAULT (timeout/no switch found).
     */
    void report_homing_fault(int axis) {
        char buf[32];
        snprintf(buf, sizeof(buf), "Homing timeout J%d", axis + 1);
        trigger_fault(FAULT, buf);
    }

    bool is_safe() const { 
        return current_state_ == NOMINAL; 
    }
    
    State get_state() const { 
        return current_state_; 
    }

    const char* get_fault_reason() const { return fault_reason_; }
    
private:
    State current_state_;
    uint32_t last_cmd_time_ms_;
    char fault_reason_[48];
    void trigger_fault(State state, const char* reason) { 
        // Latch hard faults; SOFT_ESTOP can be overridden by a FAULT.
        if (current_state_ != FAULT) { 
            current_state_ = state;
            strncpy(fault_reason_, reason, sizeof(fault_reason_) - 1);
            fault_reason_[sizeof(fault_reason_) - 1] = '\0';
        }
    }
};
