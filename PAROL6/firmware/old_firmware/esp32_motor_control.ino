/*
 * PAROL6 Day 4: Motor Control with MKS Servo42C
 * 
 * Purpose:
 *   - Receive ROS commands via USB Serial (from ros2_control)
 *   - Send position commands to MKS Servo42C motors
 *   - Read encoder feedback from motors
 *   - Report actual positions back to ROS
 * 
 * Hardware Connections:
 *   - ESP32 USB ←→ PC (ROS commands)
 *   - ESP32 Serial2 (GPIO16/17) ←→ MKS Servo42C #1
 *   - ESP32 Serial1 (GPIO9/10)  ←→ MKS Servo42C #2
 *   // Add more serial ports or use RS485 bus for 6 motors
 * 
 * Message Format (USB Serial):
 *   RX from ROS: <SEQ,J1,J2,J3,J4,J5,J6>
 *   TX to ROS:   <ACK,SEQ,J1,J2,J3,J4,J5,J6>
 * 
 * Build: Arduino IDE with ESP32 board support
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

#define USB_BAUD 115200
#define MOTOR_BAUD 38400  // MKS Servo42C default baud rate

// Motor Communication (adjust GPIO pins as needed)
#define MOTOR1_RX 16
#define MOTOR1_TX 17
#define MOTOR2_RX 9
#define MOTOR2_TX 10
// For 6 motors, use RS485 or additional UARTs

// Motor Parameters
#define STEPS_PER_REV 200       // Motor steps per revolution
#define MOTOR_MICROSTEPS 16     // Microstepping setting
#define GEAR_RATIO 1.0          // Adjust if gearbox present
#define RAD_TO_STEPS(rad) ((rad) * (STEPS_PER_REV * MOTOR_MICROSTEPS * GEAR_RATIO) / (2.0 * PI))
#define STEPS_TO_RAD(steps) ((steps) * (2.0 * PI) / (STEPS_PER_REV * MOTOR_MICROSTEPS * GEAR_RATIO))

// Communication Timing
#define MOTOR_CMD_INTERVAL_MS 20  // 50Hz motor update rate
#define FEEDBACK_TIMEOUT_MS 100   // Timeout for motor response

// ============================================================================
// MOTOR CONTROL
// ============================================================================

struct MotorState {
  int32_t target_steps;      // Commanded position (steps)
  int32_t current_steps;     // Actual position from encoder (steps)
  bool motor_ready;          // Motor communication status
};

MotorState motors[6];

// ============================================================================
// MKS SERVO42C PROTOCOL
// ============================================================================
// Based on EM_ClosedLoop communication protocol
// Full spec: https://github.com/makerbase-mks/MKS-SERVO42C

// Command codes
#define CMD_SYNC 0xFD          // Sync byte
#define CMD_EN_MOTOR 0xF3      // Enable motor
#define CMD_SET_POS_SYNC 0xF6  // Set position (synchronous mode)
#define CMD_READ_ENCODER 0x30  // Read encoder position

class MKSServo {
private:
  HardwareSerial* serial;
  uint8_t motor_id;
  
  // Calculate checksum (XOR of all bytes)
  uint8_t calcChecksum(uint8_t* data, int len) {
    uint8_t sum = 0;
    for (int i = 0; i < len; i++) {
      sum ^= data[i];
    }
    return sum;
  }
  
public:
  MKSServo(HardwareSerial* ser, uint8_t id) : serial(ser), motor_id(id) {}
  
  void begin() {
    // Enable motor
    uint8_t cmd[] = {CMD_SYNC, motor_id, CMD_EN_MOTOR, 0x01, 0x00};
    cmd[4] = calcChecksum(cmd, 4);
    serial->write(cmd, 5);
    delay(10);
  }
  
  // Send absolute position command (in steps)
  bool setPosition(int32_t steps) {
    // MKS uses 16-bit signed position (-32768 to 32767)
    // For larger range, track revolutions separately
    int16_t pos = (int16_t)(steps % 65536);
    
    uint8_t cmd[8];
    cmd[0] = CMD_SYNC;
    cmd[1] = motor_id;
    cmd[2] = CMD_SET_POS_SYNC;
    cmd[3] = (uint8_t)(pos & 0xFF);        // Low byte
    cmd[4] = (uint8_t)((pos >> 8) & 0xFF); // High byte
    cmd[5] = 0x00;  // Speed (0 = max)
    cmd[6] = 0x00;  // Acceleration (0 = default)
    cmd[7] = calcChecksum(cmd, 7);
    
    serial->write(cmd, 8);
    return true;
  }
  
  // Read current encoder position
  int32_t readPosition() {
    uint8_t cmd[4];
    cmd[0] = CMD_SYNC;
    cmd[1] = motor_id;
    cmd[2] = CMD_READ_ENCODER;
    cmd[3] = calcChecksum(cmd, 3);
    
    serial->write(cmd, 4);
    
    // Wait for response (6 bytes: SYNC, ID, STATUS, POS_LOW, POS_HIGH, CHECKSUM)
    unsigned long start = millis();
    while (serial->available() < 6) {
      if (millis() - start > FEEDBACK_TIMEOUT_MS) {
        return 0;  // Timeout - return last known position
      }
      delayMicroseconds(100);
    }
    
    uint8_t resp[6];
    serial->readBytes(resp, 6);
    
    // Verify checksum
    if (resp[5] != calcChecksum(resp, 5)) {
      return 0;  // Checksum error
    }
    
    // Extract 16-bit position
    int16_t pos = (resp[4] << 8) | resp[3];
    return (int32_t)pos;
  }
};

// ============================================================================
// MOTOR INSTANCES
// ============================================================================

HardwareSerial Motor1Serial(1);  // UART1
HardwareSerial Motor2Serial(2);  // UART2
MKSServo motor1(&Motor1Serial, 0xE0);  // Motor ID 0xE0 (default)
MKSServo motor2(&Motor2Serial, 0xE0);  // Motor ID 0xE0 (adjust if daisy-chained)

// For 6 motors: use RS485 bus with different IDs or add more UARTs

// ============================================================================
// ROS COMMUNICATION
// ============================================================================

String inputBuffer = "";
unsigned long last_motor_update = 0;
uint32_t last_seq = 0;

void setup() {
  // USB Serial for ROS communication
  Serial.begin(USB_BAUD);
  while (!Serial) {
    delay(10);
  }
  
  // Motor Serial Ports
  Motor1Serial.begin(MOTOR_BAUD, SERIAL_8N1, MOTOR1_RX, MOTOR1_TX);
  Motor2Serial.begin(MOTOR_BAUD, SERIAL_8N1, MOTOR2_RX, MOTOR2_TX);
  
  // Initialize motors
  delay(100);
  motor1.begin();
  motor2.begin();
  
  // Initialize motor states
  for (int i = 0; i < 6; i++) {
    motors[i].target_steps = 0;
    motors[i].current_steps = 0;
    motors[i].motor_ready = (i < 2);  // Only first 2 motors connected for now
  }
  
  Serial.println("READY");
  Serial.println("MKS Servo42C Motor Controller v1.0");
}

void loop() {
  // 1. Read commands from ROS
  while (Serial.available()) {
    char c = Serial.read();
    
    if (c == '<') {
      inputBuffer = "";
    } else if (c == '>') {
      processROSCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
  
  // 2. Update motors at fixed rate
  unsigned long now = millis();
  if (now - last_motor_update >= MOTOR_CMD_INTERVAL_MS) {
    last_motor_update = now;
    updateMotors();
    sendFeedback();
  }
}

// ============================================================================
// COMMAND PROCESSING
// ============================================================================

void processROSCommand(String cmd) {
  // Expected format: "SEQ,J1,J2,J3,J4,J5,J6"
  // Example: "42,0.1500,-0.2000,0.0000,0.0000,0.0000,0.0000"
  
  int commaIndex = cmd.indexOf(',');
  if (commaIndex == -1) return;  // Invalid format
  
  // Extract sequence number
  last_seq = cmd.substring(0, commaIndex).toInt();
  
  // Parse joint positions (radians)
  float positions[6];
  int start = commaIndex + 1;
  for (int i = 0; i < 6; i++) {
    int nextComma = cmd.indexOf(',', start);
    if (nextComma == -1 && i < 5) return;  // Missing joints
    
    String posStr = (nextComma == -1) ? cmd.substring(start) : cmd.substring(start, nextComma);
    positions[i] = posStr.toFloat();
    start = nextComma + 1;
  }
  
  // Convert radians to motor steps
  for (int i = 0; i < 6; i++) {
    motors[i].target_steps = (int32_t)RAD_TO_STEPS(positions[i]);
  }
}

// ============================================================================
// MOTOR UPDATE
// ============================================================================

void updateMotors() {
  // Send position commands to motors
  // For now, only motor 1 and 2 are implemented
  
  if (motors[0].motor_ready) {
    motor1.setPosition(motors[0].target_steps);
    motors[0].current_steps = motor1.readPosition();
  }
  
  if (motors[1].motor_ready) {
    motor2.setPosition(motors[1].target_steps);
    motors[1].current_steps = motor2.readPosition();
  }
  
  // Motors 3-6: Add similar logic when connected
  // For testing, echo target as current for unconnected motors
  for (int i = 2; i < 6; i++) {
    motors[i].current_steps = motors[i].target_steps;
  }
}

// ============================================================================
// FEEDBACK TO ROS
// ============================================================================

void sendFeedback() {
  // Format: <ACK,SEQ,J1,J2,J3,J4,J5,J6>
  // Positions in radians with 2 decimal precision
  
  Serial.print("<ACK,");
  Serial.print(last_seq);
  
  for (int i = 0; i < 6; i++) {
    float pos_rad = STEPS_TO_RAD(motors[i].current_steps);
    Serial.print(",");
    Serial.print(pos_rad, 2);  // 2 decimal places
  }
  
  Serial.println(">");
}

// ============================================================================
// NOTES FOR INTEGRATION
// ============================================================================

/*
 * MOTOR WIRING (MKS Servo42C):
 *   - VCC: 12-24V DC power supply
 *   - GND: Common ground with ESP32
 *   - RX/TX: Connect to ESP32 GPIO (with 3.3V ↔ 5V level shifter if needed)
 *   - EN: Optional enable pin (pull low to disable)
 * 
 * MOTOR CONFIGURATION:
 *   1. Set motor IDs if using RS485 bus (multiple motors on same serial)
 *   2. Configure microstepping via DIP switches or software
 *   3. Tune STEPS_PER_REV and MOTOR_MICROSTEPS to match your setup
 *   4. Adjust GEAR_RATIO if using gearboxes
 * 
 * SCALING FOR 6 MOTORS:
 *   Option A: Use RS485 bus with different motor IDs (0xE0-0xE5)
 *   Option B: Use ESP32-S3 with more UART ports
 *   Option C: Use I2C/SPI motor driver boards
 * 
 * TESTING PROCEDURE:
 *   1. Connect ONE motor first
 *   2. Verify it moves when ROS sends commands
 *   3. Check encoder feedback is stable
 *   4. Add motors one at a time
 *   5. Use `ros2 topic echo /joint_states` to verify positions
 * 
 * SAFETY:
 *   - Start with LOW current limits on motors
 *   - Test unloaded (no robot arm attached)
 *   - Add emergency stop button
 *   - Monitor motor temperature
 */
