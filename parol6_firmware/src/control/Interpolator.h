#pragma once

#include <stdint.h>

class LinearInterpolator {
public:
    LinearInterpolator() 
        : current_pos_(0.0f), target_pos_(0.0f), 
          step_delta_(0.0f), feedforward_vel_(0.0f),
          steps_remaining_(0) {}

    // Initialize position directly (e.g. at startup)
    void reset(float initial_pos) {
        current_pos_ = initial_pos;
        target_pos_ = initial_pos;
        step_delta_ = 0.0f;
        feedforward_vel_ = 0.0f;
        steps_remaining_ = 0;
    }

    // Called when a new ROS command is popped from the queue
    // expected_duration_ms should match the inter-arrival time (e.g. 40ms for 25Hz)
    void set_target(float target_pos, float target_vel, uint32_t expected_duration_ms) {
        start_pos_ = current_pos_;
        target_pos_ = target_pos;
        
        // Dynamically compute step delta based on inter-arrival time expectation
        if (expected_duration_ms > 0) {
            step_delta_ = (target_pos_ - start_pos_) / (float)expected_duration_ms;
        } else {
            step_delta_ = 0.0f;
        }
        
        feedforward_vel_ = target_vel; 
        steps_remaining_ = expected_duration_ms;
    }

    // Called precisely every 1ms inside the Control ISR
    void tick_1ms(float& out_pos_cmd, float& out_vel_ff) {
        if (steps_remaining_ > 0) {
            current_pos_ += step_delta_;
            steps_remaining_--;
        } else {
            // Reached target, hold position, zero feedforward velocity
            current_pos_ = target_pos_; 
            feedforward_vel_ = 0.0f;
        }
        
        out_pos_cmd = current_pos_;
        out_vel_ff = feedforward_vel_;
    }
    
private:
    float current_pos_;
    float start_pos_;
    float target_pos_;
    float step_delta_;
    float feedforward_vel_;
    uint32_t steps_remaining_;
};
