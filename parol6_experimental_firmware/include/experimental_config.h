#pragma once

#include <Arduino.h>

static constexpr uint32_t SERIAL_BAUD = 115200;
static constexpr uint32_t COMMAND_TIMEOUT_MS = 250;
static constexpr uint32_t CONTROL_PERIOD_US = 5000;
static constexpr uint32_t FEEDBACK_PERIOD_MS = 40;
static constexpr size_t NUM_AXES = 6;

static constexpr uint8_t STEP_PINS[NUM_AXES] = {2, 6, 7, 8, 4, 5};
static constexpr uint8_t DIR_PINS[NUM_AXES] = {30, 31, 32, 33, 34, 35};
static constexpr uint8_t ENCODER_PINS[NUM_AXES] = {10, 11, 12, 14, 15, 18};

static constexpr float GEAR_RATIOS[NUM_AXES] = {
    6.4f, 20.0f, 18.0952381f, 4.0f, 4.0f, 10.0f
};
static constexpr int MICROSTEPS[NUM_AXES] = {32, 32, 32, 32, 32, 32};
static constexpr bool DIR_INVERT[NUM_AXES] = {true, false, true, false, false, true};
static constexpr float KP_GAINS[NUM_AXES] = {5.0f, 5.0f, 2.0f, 2.0f, 2.0f, 2.0f};
static constexpr float MAX_VEL_RAD_S[NUM_AXES] = {3.0f, 3.0f, 6.0f, 6.0f, 6.0f, 6.0f};
static constexpr float HOME_OFFSETS_RAD[NUM_AXES] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static constexpr float DEAD_BAND_RAD_S = 0.02f;

