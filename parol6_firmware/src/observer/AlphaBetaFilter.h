#pragma once

#include <math.h>

class AlphaBetaFilter {
public:
    AlphaBetaFilter(float alpha, float beta, float dt_s) 
        : alpha_(alpha), beta_(beta), dt_(dt_s) {}

    void set_initial_position(float initial_rad) {
        estimated_pos_ = initial_rad;
        estimated_vel_ = 0.0f;
        last_raw_angle_ = initial_rad;
        turn_offset_ = 0.0f;
    }

    // Called at control frequency with raw, potentially wrapped sensor data
    void update(float raw_encoder_angle_rad) {
        // 1. Unwrap absolute boundary crossing (e.g. 0 -> 2PI)
        float delta = raw_encoder_angle_rad - last_raw_angle_;
        if (delta > M_PI)  turn_offset_ -= 2.0f * M_PI;
        if (delta < -M_PI) turn_offset_ += 2.0f * M_PI;
        last_raw_angle_ = raw_encoder_angle_rad;
        
        float unwrapped_measurement = raw_encoder_angle_rad + turn_offset_;

        // 2. Observer Prediction Step
        estimated_pos_ += estimated_vel_ * dt_;
        
        // 3. Innovation (Measurement Error)
        float residual = unwrapped_measurement - estimated_pos_;
        
        // 4. Observer Update Step
        estimated_pos_ += alpha_ * residual;
        estimated_vel_ += (beta_ / dt_) * residual;
    }
    
    float get_position() const { return estimated_pos_; }
    float get_velocity() const { return estimated_vel_; }

private:
    float estimated_pos_ = 0.0f;
    float estimated_vel_ = 0.0f;
    float last_raw_angle_ = 0.0f;
    float turn_offset_ = 0.0f;
    float alpha_, beta_, dt_;
};
