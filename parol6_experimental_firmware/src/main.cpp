#include <Arduino.h>

#include "experimental_config.h"
#include "FlexPWMGenerator.h"
#include "SoftwareInterruptEncoder.h"

struct CommandFrame {
    uint32_t seq = 0;
    float positions[NUM_AXES] = {};
    float velocities[NUM_AXES] = {};
    bool valid = false;
};

static FlexPWMGenerator steppers[NUM_AXES] = {
    FlexPWMGenerator(STEP_PINS[0], DIR_PINS[0]),
    FlexPWMGenerator(STEP_PINS[1], DIR_PINS[1]),
    FlexPWMGenerator(STEP_PINS[2], DIR_PINS[2]),
    FlexPWMGenerator(STEP_PINS[3], DIR_PINS[3]),
    FlexPWMGenerator(STEP_PINS[4], DIR_PINS[4]),
    FlexPWMGenerator(STEP_PINS[5], DIR_PINS[5]),
};

static SoftwareInterruptEncoder encoders[NUM_AXES] = {
    SoftwareInterruptEncoder(ENCODER_PINS[0], 0),
    SoftwareInterruptEncoder(ENCODER_PINS[1], 1),
    SoftwareInterruptEncoder(ENCODER_PINS[2], 2),
    SoftwareInterruptEncoder(ENCODER_PINS[3], 3),
    SoftwareInterruptEncoder(ENCODER_PINS[4], 4),
    SoftwareInterruptEncoder(ENCODER_PINS[5], 5),
};

static float target_positions[NUM_AXES] = {};
static float target_velocities[NUM_AXES] = {};
static float measured_positions[NUM_AXES] = {};
static float measured_velocities[NUM_AXES] = {};
static float motor_positions[NUM_AXES] = {};
static float previous_raw_angles[NUM_AXES] = {};
static float previous_joint_positions[NUM_AXES] = {};
static bool encoder_initialized[NUM_AXES] = {};

static uint32_t last_command_ms = 0;
static uint32_t last_control_us = 0;
static uint32_t last_feedback_ms = 0;
static uint32_t feedback_seq = 0;
static uint32_t last_command_seq = 0;
static bool command_seen = false;
static bool motors_enabled = false;

static float wrap_delta(float current, float previous) {
    float delta = current - previous;
    if (delta > PI) {
        delta -= 2.0f * PI;
    } else if (delta < -PI) {
        delta += 2.0f * PI;
    }
    return delta;
}

static void send_status(const char* msg) {
    Serial.println(msg);
}

static void stop_all_motors() {
    for (size_t axis = 0; axis < NUM_AXES; ++axis) {
        steppers[axis].stop();
        target_velocities[axis] = 0.0f;
        measured_velocities[axis] = 0.0f;
    }
}

static void zero_current_pose() {
    for (size_t axis = 0; axis < NUM_AXES; ++axis) {
        measured_positions[axis] = HOME_OFFSETS_RAD[axis];
        previous_joint_positions[axis] = measured_positions[axis];
        motor_positions[axis] = measured_positions[axis] * GEAR_RATIOS[axis];
        target_positions[axis] = measured_positions[axis];
        target_velocities[axis] = 0.0f;
    }
}

static bool parse_command(char* line, CommandFrame& out) {
    out = CommandFrame();

    size_t len = strlen(line);
    if (len < 3 || line[0] != '<' || line[len - 1] != '>') {
        return false;
    }

    line[len - 1] = '\0';
    char* body = line + 1;

    if (strcmp(body, "ENABLE") == 0) {
        motors_enabled = true;
        last_command_ms = millis();
        send_status("<ENABLE_ACK>");
        return false;
    }

    if (strcmp(body, "DISABLE") == 0 || strcmp(body, "STOP") == 0) {
        motors_enabled = false;
        stop_all_motors();
        send_status("<DISABLE_ACK>");
        return false;
    }

    if (strcmp(body, "HOME") == 0 || strcmp(body, "ZERO") == 0) {
        zero_current_pose();
        send_status("HOMING_DONE");
        return false;
    }

    char* token = strtok(body, ",");
    if (token == nullptr) {
        return false;
    }

    out.seq = strtoul(token, nullptr, 10);
    for (size_t axis = 0; axis < NUM_AXES; ++axis) {
        token = strtok(nullptr, ",");
        if (token == nullptr) {
            return false;
        }
        out.positions[axis] = atof(token);
    }

    for (size_t axis = 0; axis < NUM_AXES; ++axis) {
        token = strtok(nullptr, ",");
        out.velocities[axis] = token ? atof(token) : 0.0f;
    }

    out.valid = true;
    return true;
}

static void read_serial() {
    static char buffer[256];
    static size_t pos = 0;

    while (Serial.available() > 0) {
        char c = static_cast<char>(Serial.read());
        if (c == '\r' || c == '\n') {
            if (pos == 0) {
                continue;
            }
            buffer[pos] = '\0';
            CommandFrame cmd;
            if (parse_command(buffer, cmd) && cmd.valid) {
                bool is_new = !command_seen || static_cast<int32_t>(cmd.seq - last_command_seq) > 0;
                if (is_new) {
                    command_seen = true;
                    last_command_seq = cmd.seq;
                    last_command_ms = millis();
                    motors_enabled = true;
                    for (size_t axis = 0; axis < NUM_AXES; ++axis) {
                        target_positions[axis] = cmd.positions[axis];
                        target_velocities[axis] = cmd.velocities[axis];
                    }
                } else {
                    send_status("STALE_CMD");
                }
            } else if (buffer[0] == '<') {
                send_status("ERR_PARSE");
            }
            pos = 0;
        } else if (pos < sizeof(buffer) - 1) {
            buffer[pos++] = c;
        }
    }
}

static void update_measurements(float dt_s) {
    for (size_t axis = 0; axis < NUM_AXES; ++axis) {
        float raw = encoders[axis].read_angle();
        if (!encoder_initialized[axis]) {
            previous_raw_angles[axis] = raw;
            motor_positions[axis] = raw;
            measured_positions[axis] = raw / GEAR_RATIOS[axis];
            previous_joint_positions[axis] = measured_positions[axis];
            encoder_initialized[axis] = true;
            continue;
        }

        float delta = wrap_delta(raw, previous_raw_angles[axis]);
        previous_raw_angles[axis] = raw;
        motor_positions[axis] += delta;

        float joint_position = motor_positions[axis] / GEAR_RATIOS[axis];
        measured_velocities[axis] = (joint_position - previous_joint_positions[axis]) / dt_s;
        measured_positions[axis] = joint_position;
        previous_joint_positions[axis] = joint_position;
    }
}

static void apply_motor_command(size_t axis, float joint_velocity_rad_s) {
    if (!motors_enabled) {
        steppers[axis].stop();
        return;
    }

    bool forward = joint_velocity_rad_s >= 0.0f;
    if (DIR_INVERT[axis]) {
        forward = !forward;
    }

    float steps_per_rad = (200.0f * static_cast<float>(MICROSTEPS[axis]) * GEAR_RATIOS[axis]) / (2.0f * PI);
    float frequency_hz = fabsf(joint_velocity_rad_s) * steps_per_rad;

    steppers[axis].set_direction(forward);
    if (frequency_hz < 1.0f) {
        steppers[axis].stop();
    } else {
        steppers[axis].set_frequency(frequency_hz);
    }
}

static void control_step() {
    uint32_t now_us = micros();
    if (last_control_us == 0) {
        last_control_us = now_us;
        return;
    }
    if (now_us - last_control_us < CONTROL_PERIOD_US) {
        return;
    }

    float dt_s = static_cast<float>(now_us - last_control_us) / 1000000.0f;
    last_control_us = now_us;

    if (motors_enabled && millis() - last_command_ms > COMMAND_TIMEOUT_MS) {
        motors_enabled = false;
        stop_all_motors();
    }

    update_measurements(dt_s);

    for (size_t axis = 0; axis < NUM_AXES; ++axis) {
        float error = target_positions[axis] - measured_positions[axis];
        float velocity = (KP_GAINS[axis] * error) + target_velocities[axis];
        if (velocity > MAX_VEL_RAD_S[axis]) {
            velocity = MAX_VEL_RAD_S[axis];
        }
        if (velocity < -MAX_VEL_RAD_S[axis]) {
            velocity = -MAX_VEL_RAD_S[axis];
        }
        if (fabsf(velocity) < DEAD_BAND_RAD_S) {
            velocity = 0.0f;
        }
        apply_motor_command(axis, velocity);
    }
}

static void send_feedback() {
    uint32_t now_ms = millis();
    if (now_ms - last_feedback_ms < FEEDBACK_PERIOD_MS) {
        return;
    }
    last_feedback_ms = now_ms;

    Serial.print("<ACK,");
    Serial.print(feedback_seq++);
    for (size_t axis = 0; axis < NUM_AXES; ++axis) {
        Serial.print(",");
        Serial.print(measured_positions[axis], 4);
    }
    for (size_t axis = 0; axis < NUM_AXES; ++axis) {
        Serial.print(",");
        Serial.print(measured_velocities[axis], 4);
    }
    Serial.print(",0,");
    Serial.print(motors_enabled ? 1 : 0);
    Serial.println(">");
}

void setup() {
    Serial.begin(SERIAL_BAUD);

    for (size_t axis = 0; axis < NUM_AXES; ++axis) {
        steppers[axis].init();
        encoders[axis].init();
        target_positions[axis] = HOME_OFFSETS_RAD[axis];
        previous_joint_positions[axis] = HOME_OFFSETS_RAD[axis];
    }

    delay(100);
    send_status("EXPERIMENTAL_PAROL6_READY");
}

void loop() {
    read_serial();
    control_step();
    send_feedback();
}

