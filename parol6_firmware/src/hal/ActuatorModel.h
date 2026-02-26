#pragma once

#include <Arduino.h>

/**
 * @brief Phase 4: Stage 2 - Actuator Kinematics Model
 * 
 * This class isolates the physical mechanical realities (gear ratios, 
 * microstepping) from the pure `rad/s` kinematic math of the ControlLaw.
 * 
 * It converts the `velocity_command` from the 1 kHz ISR into exact
 * `steps_per_second` (Hz) and `direction` boolean values required by the 
 * FlexPWM step generator.
 */
class ActuatorModel {
public:
    /**
     * @param gear_ratio The exact, non-integer mechanical reduction ratio (e.g., 18.0952381f for J3).
     * @param microsteps_per_rev The number of microsteps configured on the stepper driver per 1 motor shaft revolution.
     * @param invert_dir Set to true if the physical axis moves opposite to the URDF (+rad convention).
     */
    ActuatorModel(float gear_ratio, uint32_t microsteps_per_rev, bool invert_dir)
        : gear_ratio_(gear_ratio), 
          microsteps_per_rev_(microsteps_per_rev), 
          invert_dir_(invert_dir) 
    {
        // Precompute the conversion constant to avoid division in the 1 kHz ISR.
        // Formula: steps_per_rad = (microsteps_per_rev * gear_ratio) / (2 * PI)
        rad_to_steps_factor_ = ((float)microsteps_per_rev_ * gear_ratio_) / (2.0f * PI);
    }

    /**
     * Converts pure kinematic velocity (rad/s) into physical stepper driver commands.
     * 
     * @param velocity_rad_s The commanded velocity from the ControlLaw output saturation.
     * @param out_frequency_hz [OUTPUT] The absolute pulse train frequency required.
     * @param out_direction [OUTPUT] The physical logic state for the DIR pin.
     */
    void compute_step_command(float velocity_rad_s, uint32_t& out_frequency_hz, bool& out_direction) {
        // 1. Determine Direction
        bool is_positive_vel = (velocity_rad_s >= 0.0f);
        out_direction = invert_dir_ ? !is_positive_vel : is_positive_vel;

        // 2. Determine Absolute Frequency (Hz)
        float abs_vel = abs(velocity_rad_s);
        float steps_per_sec = abs_vel * rad_to_steps_factor_;
        
        // Truncate to integer frequency. The Phase accumulation layer (later) 
        // can handle fractional steps/s if ultra-low speed smoothing is needed.
        out_frequency_hz = (uint32_t)steps_per_sec;
    }

private:
    float gear_ratio_;
    uint32_t microsteps_per_rev_;
    bool invert_dir_;
    
    // Cached conversion factor
    float rad_to_steps_factor_;
};
