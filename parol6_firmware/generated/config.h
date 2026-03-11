#pragma once
#define FEATURE_INTERPOLATOR_LOCK 1
#define FEATURE_ALPHABETA_FILTER  1
#define FEATURE_VEL_FEEDFORWARD   1
#define FEATURE_WATCHDOG          1
#define FEATURE_SAFETY_SUPERVISOR 1
#define FEATURE_ANTI_GLITCH       1
#define FEATURE_VEL_DEADBAND      1
#define FEATURE_ENCODER_TEST_MODE 0
#define FEATURE_SINE_TEST_MODE    0
#define FEATURE_HARDWARE_PWM      1
#define FEATURE_HARDWARE_ENCODER  1
#define FIXED_STEP_FREQ_HZ        0
#define VELOCITY_DEADBAND_RAD_S   0.02f
#define TRANSPORT_MODE            1
#define ROS_COMMAND_RATE_HZ       25
#define FEEDBACK_RATE_HZ          10
#define CONTROL_LOOP_RATE_HZ      1000
#define CONTROL_LOOP_PERIOD_US    1000
#define COMMAND_TIMEOUT_MS        200
#define NUM_AXES                  6
static const int STEP_PINS[6]       = {2, 6, 7, 8, 4, 5};
static const int DIR_PINS[6]        = {30, 31, 32, 33, 34, 35};
static const int ENCODER_PINS[6]    = {10, 11, 12, 14, 15, 18};
static const float GEAR_RATIOS[6]   = {6.4f, 20.0f, 18.0952f, 4.0f, 4.0f, 10.0f};
static const int MICROSTEPS[6]      = {32, 32, 32, 32, 32, 32};
static const bool DIR_INVERT[6]     = {1, 0, 1, 0, 0, 1};
static const bool ROS_DIR_INVERT[6] = {1, 0, 1, 0, 0, 1};
static const bool JOINT_ENABLED[6]  = {1, 1, 1, 1, 1, 1};
static const float MAX_VEL_RAD_S[6] = {3.0f, 3.0f, 6.0f, 6.0f, 6.0f, 6.0f};
static const float STEPS_PER_RAD[6] = {6518.9865f, 20371.8327f, 18431.6194f, 4074.3665f, 4074.3665f, 10185.9164f};
static const float KP_GAINS[6]      = {5.0f, 5.0f, 2.0f, 2.0f, 2.0f, 2.0f};
static const float KI_GAINS[6]      = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static const float MAX_INTEGRAL[6]  = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
static const float AB_ALPHA[6]      = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
static const float AB_BETA[6]       = {0.005f, 0.005f, 0.005f, 0.005f, 0.005f, 0.005f};
static const bool LIMIT_ENABLED[6]      = {0, 0, 0, 0, 0, 0};
static const int LIMIT_PINS[6]          = {20, 21, 22, 23, 24, 25};
static const int LIMIT_POLARITY[6]      = {0, 0, 0, 0, 0, 0};
static const int LIMIT_TYPE[6]          = {2, 2, 2, 2, 2, 2};
static const int LIMIT_PULL[6]          = {1, 1, 1, 1, 1, 1};
static const bool LIMIT_ACTIVE_HIGH[6]  = {0, 0, 0, 0, 0, 0};
static const int HOMING_ORDER[6]        = {3, 4, 5, 0, 1, 2};
static const float HOME_OFFSETS_RAD[6]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static const int HOMING_SPEED[6]        = {500, 500, 500, 500, 500, 500};
static const int HOMED_OFFSET[6]        = {13500, 19588, 23020, -10200, 8900, 15900};
static const int STANDBY_POS[6]         = {10240, -32000, 57905, 0, 0, 32000};
