/*
 * PAROL6 ESP32 Feedback Firmware (Arduino Version)
 * 
 * Compatible with: Arduino IDE, PlatformIO
 * Target: ESP32 (tested on ESP32-D0WDQ6-V3)
 * 
 * PROTOCOL:
 * Receives: <SEQ,J1,J2,J3,J4,J5,J6>
 * Sends:    <ACK,SEQ,J1,J2,J3,J4,J5,J6>
 * 
 * This firmware echoes position commands from ROS2 and tracks
 * joint positions. Ready for motor integration (Day 4).
 */

// Current joint positions (radians) - updated from commands
float joint_positions[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
uint32_t last_seq = 0;

// Buffer for incoming serial data
const int BUFFER_SIZE = 128;
char line_buffer[BUFFER_SIZE];
int line_idx = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("READY: ESP32_FEEDBACK_ARDUINO_V1");
}

void loop() {
  // Process incoming serial data
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '<') {
      // Start of new command
      line_idx = 0;
      line_buffer[line_idx++] = c;
      
    } else if (c == '>') {
      // End of command - parse and respond
      line_buffer[line_idx++] = c;
      line_buffer[line_idx] = '\0';
      
      parseAndRespond(line_buffer, line_idx);
      line_idx = 0;
      
    } else if (line_idx > 0 && line_idx < BUFFER_SIZE - 1) {
      // Middle of command
      line_buffer[line_idx++] = c;
    }
  }
}

void parseAndRespond(const char* cmd, int len) {
  // Parse: <SEQ,J1,J2,J3,J4,J5,J6>
  
  // Skip '<' and find tokens
  char buffer[BUFFER_SIZE];
  strncpy(buffer, cmd, len);
  buffer[len] = '\0';
  
  // Remove '<' and '>'
  char *start = buffer;
  if (*start == '<') start++;
  char *end = strchr(start, '>');
  if (end) *end = '\0';
  
  // Parse sequence number (first token)
  char *token = strtok(start, ",");
  if (token == NULL) return;
  
  uint32_t seq = (uint32_t)atol(token);
  last_seq = seq;
  
  // Parse 6 joint positions
  int joint_idx = 0;
  while ((token = strtok(NULL, ",")) != NULL && joint_idx < 6) {
    joint_positions[joint_idx] = atof(token);
    joint_idx++;
  }
  
  // Only send ACK if we got all 6 joints
  if (joint_idx == 6) {
    sendFeedback(seq);
  }
}

void sendFeedback(uint32_t seq) {
  // Format: <ACK,SEQ,J1,J2,J3,J4,J5,J6>
  char response[128];
  snprintf(response, sizeof(response),
           "<ACK,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f>\n",
           seq,
           joint_positions[0], joint_positions[1], joint_positions[2],
           joint_positions[3], joint_positions[4], joint_positions[5]);
  
  Serial.print(response);
}
