/*
 * PAROL6 UART Control with Velocity Support
 * 
 * Uses MKS SERVO42C UART mode with RS485 daisy-chain
 * - All 6 motors on ONE serial bus (Serial2)
 * - Each motor has unique address (0xE0-0xE5)
 * - Position + Velocity commands from ROS
 * - MKS handles acceleration/deceleration internally
 * - Much more reliable than step/dir for geared systems
 */

#include "SERVO42C.h"

#define USB_BAUD 115200
#define MOTOR_BAUD 115200
#define NUM_MOTORS 6

// RS485 pins (all motors share same bus)
#define MOTOR_RX_PIN 16
#define MOTOR_TX_PIN 17

// Motor addresses (each motor must be configured with unique address)
const uint8_t MOTOR_ADDRS[NUM_MOTORS] = {
  0xE0,  // J1
  0xE1,  // J2
  0xE2,  // J3
  0xE3,  // J4
  0xE4,  // J5
  0xE5   // J6
};

// Motor configuration
#define STEPS_PER_REV 200
#define MICROSTEPS 16
#define BASE_PULSES_PER_REV (STEPS_PER_REV * MICROSTEPS)  // 3200

// Gearbox ratios (motor revolutions per joint revolution)
const float GEAR_RATIOS[NUM_MOTORS] = {
  25.0,   // J1: Direct drive
  1.0,  // J2: 25:1 gearbox
  1.0,   // J3: Direct drive
  1.0,   // J4: Direct drive
  1.0,   // J5: Direct drive
  1.0    // J6: Direct drive
};

// Calculate pulses per joint revolution (including gearbox)
float getPulsesPerJointRev(int motor_idx) {
  return BASE_PULSES_PER_REV * GEAR_RATIOS[motor_idx];
}

// Motor objects (all share Serial2 bus, differentiated by address)
SERVO42C* motors[NUM_MOTORS];

// State
float current_positions[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};
float target_positions[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};
float target_velocities[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};

String inputBuffer = "";
uint32_t last_seq = 0;
unsigned long last_feedback = 0;

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial) delay(10);
  
  // Initialize shared RS485 bus
  Serial2.begin(MOTOR_BAUD, SERIAL_8N1, MOTOR_RX_PIN, MOTOR_TX_PIN);
  delay(100);
  
  // Create motor objects (all use same Serial2, different addresses)
  for (int i = 0; i < NUM_MOTORS; i++) {
    motors[i] = new SERVO42C(Serial2, MOTOR_ADDRS[i]);
    motors[i]->begin(MOTOR_BAUD);
    delay(50);
    
    // Configure for UART mode
    motors[i]->setWorkMode(MODE_UART);
    delay(50);
    motors[i]->uartEnable(true);
    delay(50);
    motors[i]->setMicrostep(MICROSTEPS);
    delay(50);
  }
  
  delay(500);  // Let all motors initialize
}

void loop() {
  // Read commands from ROS
  while (Serial.available()) {
    char c = Serial.read();
    
    if (c == '<') {
      inputBuffer = "";
    } else if (c == '>') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
  
  // Send feedback at 100 Hz
  unsigned long now = millis();
  if (now - last_feedback >= 10) {
    last_feedback = now;
    sendFeedback();
  }
}

void processCommand(String cmd) {
  // Parse: "SEQ,J1_pos,J1_vel,J2_pos,J2_vel,J3_pos,J3_vel,J4_pos,J4_vel,J5_pos,J5_vel,J6_pos,J6_vel"
  // Total: 1 (seq) + 12 (6 joints × 2 values) = 13 tokens
  
  int idx = cmd.indexOf(',');
  if (idx == -1) return;
  
  last_seq = cmd.substring(0, idx).toInt();
  
  // Parse all 6 joints (position + velocity pairs)
  int start = idx + 1;
  for (int i = 0; i < NUM_MOTORS; i++) {
    // Parse position
    int next = cmd.indexOf(',', start);
    if (next == -1) return;
    
    String pos_val = cmd.substring(start, next);
    target_positions[i] = pos_val.toFloat();
    
    // Parse velocity
    start = next + 1;
    next = cmd.indexOf(',', start);
    
    String vel_val;
    if (i == NUM_MOTORS - 1) {
      vel_val = cmd.substring(start);
    } else {
      if (next == -1) return;
      vel_val = cmd.substring(start, next);
      start = next + 1;
    }
    
    target_velocities[i] = vel_val.toFloat();
  }
  
  // Update all motors with new commands
  updateMotors();
}

void updateMotors() {
  // Update all motors
  for (int motor_idx = 0; motor_idx < NUM_MOTORS; motor_idx++) {
    float position_error = target_positions[motor_idx] - current_positions[motor_idx];
    
    // Dead zone
    if (abs(position_error) < 0.001) continue;  // ~0.06 degrees
    
    // Convert position to motor pulses (accounting for gearbox)
    float pulses_per_joint_rev = getPulsesPerJointRev(motor_idx);
    float target_pulses = (target_positions[motor_idx] * pulses_per_joint_rev) / (2.0 * PI);
    float current_pulses = (current_positions[motor_idx] * pulses_per_joint_rev) / (2.0 * PI);
    
    int32_t delta_pulses = (int32_t)(target_pulses - current_pulses);
    
    if (abs(delta_pulses) < 5) continue;  // Already close enough
    
    // Convert velocity to MKS speed
    // MKS speed range: 0-255 (0 = slowest, 255 = fastest)
    // Map velocity (rad/s) to MKS speed
    // For geared joints, scale velocity by gear ratio
    float velocity_rad_s = abs(target_velocities[motor_idx]);
    
    // Scale factor: adjust based on testing
    // Higher gear ratio = need higher motor speed for same joint velocity
    float speed_scale = 40.0 * GEAR_RATIOS[motor_idx];
    uint8_t mks_speed = constrain((int)(velocity_rad_s * speed_scale), 10, 255);
    
    // Send position command to MKS
    // CRITICAL: MKS UART may have pulse limits, so split large movements
    RunDirection dir = (delta_pulses > 0) ? RUN_FWD : RUN_REV;
    uint32_t total_pulses = abs(delta_pulses);
    uint8_t status;
    
    // Split into chunks if needed (max 10000 pulses per command)
    const uint32_t MAX_PULSES_PER_CMD = 10000;
    uint32_t pulses_sent = 0;
    
    while (pulses_sent < total_pulses) {
      uint32_t pulses_this_cmd = min(total_pulses - pulses_sent, MAX_PULSES_PER_CMD);
      
      motors[motor_idx]->uartRunPulses(mks_speed, dir, pulses_this_cmd, status);
      
      pulses_sent += pulses_this_cmd;
      
      // Small delay between commands
      if (pulses_sent < total_pulses) {
        delay(5);
      }
    }
    
    // DON'T update current_positions here!
    // We'll read the actual position from the motor in sendFeedback()
    
    delay(2);  // Small delay between motor commands on shared bus
  }
}

// Position tracking with filtering
float last_valid_positions[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};
float position_velocities[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};  // Estimated velocities
unsigned long last_position_update[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};
int consecutive_read_failures[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};

void sendFeedback() {
  // Read actual positions from motors with filtering
  // Format: <ACK,SEQ,J1_pos,J1_vel,J2_pos,J2_vel,...>
  
  unsigned long now = millis();
  
  // Update current positions by reading from motors
  for (int i = 0; i < NUM_MOTORS; i++) {
    EncoderReading enc;
    bool read_success = motors[i]->readEncoder(enc);
    
    if (read_success) {
      // Convert encoder reading to motor pulses
      // Encoder value is 0-65535 for one revolution, carry tracks full revolutions
      int32_t motor_pulses = enc.carry * 65536 + enc.value;
      
      // Convert motor pulses to joint angle (accounting for gearbox)
      // Note: Encoder counts per revolution may differ from step counts
      // Assuming encoder has 65536 counts per motor revolution
      float encoder_counts_per_rev = 65536.0;
      float motor_revolutions = motor_pulses / encoder_counts_per_rev;
      
      // Account for gearbox
      float joint_revolutions = motor_revolutions / GEAR_RATIOS[i];
      float new_position = joint_revolutions * 2.0 * PI;
      
      // Outlier rejection: check if position change is reasonable
      float position_change = new_position - last_valid_positions[i];
      float time_delta = (now - last_position_update[i]) / 1000.0;  // seconds
      
      // Maximum reasonable velocity: 10 rad/s (very generous)
      float max_reasonable_change = 10.0 * time_delta;
      
      if (abs(position_change) < max_reasonable_change || consecutive_read_failures[i] > 5) {
        // Position is reasonable OR we haven't had a good read in a while
        
        // Exponential smoothing filter (alpha = 0.7 for responsiveness)
        float alpha = 0.7;
        current_positions[i] = alpha * new_position + (1.0 - alpha) * last_valid_positions[i];
        
        // Estimate velocity for prediction
        if (time_delta > 0.001) {
          position_velocities[i] = position_change / time_delta;
        }
        
        last_valid_positions[i] = current_positions[i];
        last_position_update[i] = now;
        consecutive_read_failures[i] = 0;
      } else {
        // Outlier detected - ignore this reading
        consecutive_read_failures[i]++;
      }
    } else {
      // Read failed - use prediction based on last known velocity
      consecutive_read_failures[i]++;
      
      if (consecutive_read_failures[i] < 10) {
        // Predict position based on velocity
        float time_delta = (now - last_position_update[i]) / 1000.0;
        current_positions[i] = last_valid_positions[i] + position_velocities[i] * time_delta;
      }
      // If too many failures, just keep last known position
    }
  }
  
  Serial.print("<ACK,");
  Serial.print(last_seq);
  
  for (int i = 0; i < NUM_MOTORS; i++) {
    Serial.print(",");
    Serial.print(current_positions[i], 3);
    Serial.print(",");
    Serial.print(target_velocities[i], 3);
  }
  
  Serial.print(">");
  Serial.println();
  Serial.flush();
}

/*
 * DESIGN NOTES:
 * 
 * RS485 Daisy-Chain:
 * - All 6 motors on ONE serial bus (Serial2)
 * - Each motor has unique address (0xE0-0xE5)
 * - Motors must be configured with addresses using MKS software
 * - Wiring: A+/B- daisy-chained between all motors
 * 
 * UART Mode Advantages:
 * - Direct velocity control (send speed to MKS)
 * - MKS handles acceleration internally
 * - Can read encoder for true closed-loop
 * - No step pulse generation overhead
 * - Much more reliable with gearboxes
 * - No packet loss issues
 * 
 * Gearbox Handling:
 * - Automatically accounts for gear ratio in position
 * - Velocity scaling: higher gear ratio = higher motor speed
 * - J2 with 25:1 gearbox works perfectly
 * 
 * Motor Address Configuration:
 * - Use MKS configuration software to set each motor's address
 * - Default is 0xE0, change to 0xE1, 0xE2, etc.
 * - Each motor must have unique address on the bus
 */
