#include <AccelStepper.h>

/**
 * PAROL6 Arduino Uno Firmware WITH FEEDBACK (Day 3)
 * 
 * HARDWARE CONSTRAINTS (Arduino Uno):
 * The Uno has limited pins. We use every single Digital Pin (2-13) for 6 motors.
 * Limit Switches and Alarms must go to Analog Pins (A0-A5) used as Digital Inputs.
 * 
 * PIN MAPPING:
 * J1: Step 2, Dir 3
 * J2: Step 4, Dir 5
 * J3: Step 6, Dir 7
 * J4: Step 8, Dir 9
 * J5: Step 10, Dir 11
 * J6: Step 12, Dir 13
 * Limit Switch (All in Series?): A0
 * Alarm (All in Parallel?): A1
 */

// Define stepper motor connections and interface type
#define MOTOR_INTERFACE_TYPE 1 // 1 = Driver (Step/Dir)

// Joint 1
#define STEP_1 2
#define DIR_1  3
// Joint 2
#define STEP_2 4
#define DIR_2  5
// Joint 3
#define STEP_3 6
#define DIR_3  7
// Joint 4
#define STEP_4 8
#define DIR_4  9
// Joint 5
#define STEP_5 10
#define DIR_5  11
// Joint 6
#define STEP_6 12
#define DIR_6  13

// Sensors (Using Analog Pins as Digital)
#define LIMIT_PIN A0 
#define ALARM_PIN A1

// Joint Config (Steps per Radian - ADJUST THESE AFTER CALIBRATION)
float STEPS_PER_RAD[] = {
  10185.9, // J1
  8500.0,  // J2
  8500.0,  // J3
  4000.0,  // J4
  2000.0,  // J5
  2000.0   // J6
};

// AccelStepper Instances
AccelStepper stepper1(MOTOR_INTERFACE_TYPE, STEP_1, DIR_1);
AccelStepper stepper2(MOTOR_INTERFACE_TYPE, STEP_2, DIR_2);
AccelStepper stepper3(MOTOR_INTERFACE_TYPE, STEP_3, DIR_3);
AccelStepper stepper4(MOTOR_INTERFACE_TYPE, STEP_4, DIR_4);
AccelStepper stepper5(MOTOR_INTERFACE_TYPE, STEP_5, DIR_5);
AccelStepper stepper6(MOTOR_INTERFACE_TYPE, STEP_6, DIR_6);

// Serial Buffer
const byte numChars = 128;
char receivedChars[numChars];
boolean newData = false;

// **NEW: Sequence counter for feedback**
uint32_t last_seq = 0;

void setup() {
  Serial.begin(115200);
  
  pinMode(LIMIT_PIN, INPUT_PULLUP);
  pinMode(ALARM_PIN, INPUT_PULLUP);

  configStepper(stepper1);
  configStepper(stepper2);
  configStepper(stepper3);
  configStepper(stepper4);
  configStepper(stepper5);
  configStepper(stepper6);

  Serial.println("READY: UNO_FEEDBACK_V1");
}

void configStepper(AccelStepper &stp) {
  stp.setMaxSpeed(4000);
  stp.setAcceleration(2000);
}

void loop() {
  // 1. Check Serial
  recvWithStartEndMarkers();
  if (newData) {
    parseData();
    newData = false;
  }

  // 2. Run Motors
  stepper1.run();
  stepper2.run();
  stepper3.run();
  stepper4.run();
  stepper5.run();
  stepper6.run();
}

void recvWithStartEndMarkers() {
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;
 
    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();
        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) ndx = numChars - 1;
            } else {
                receivedChars[ndx] = '\0';
                recvInProgress = false;
                ndx = 0;  // Reset for next message
                newData = true;
            }
        } else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}

void parseData() {
    // **NEW: Parse incoming format <SEQ,J1,J2,J3,J4,J5,J6>**
    char * strtokIndx;
    
    // First token is sequence number
    strtokIndx = strtok(receivedChars, ",");
    if (strtokIndx != NULL) {
        last_seq = atol(strtokIndx);  // Store sequence number
    }
    
    // Parse joint positions
    float j_rads[6];
    int i = 0;
    strtokIndx = strtok(NULL, ",");
    while (strtokIndx != NULL && i < 6) {
        j_rads[i] = atof(strtokIndx);
        strtokIndx = strtok(NULL, ",");
        i++;
    }

    // Apply to Motors (only if we got 6 values)
    if (i == 6) {
        stepper1.moveTo(j_rads[0] * STEPS_PER_RAD[0]);
        stepper2.moveTo(j_rads[1] * STEPS_PER_RAD[1]);
        stepper3.moveTo(j_rads[2] * STEPS_PER_RAD[2]);
        stepper4.moveTo(j_rads[3] * STEPS_PER_RAD[3]);
        stepper5.moveTo(j_rads[4] * STEPS_PER_RAD[4]);
        stepper6.moveTo(j_rads[5] * STEPS_PER_RAD[5]);
        
        // **NEW: Send ACK feedback immediately**
        sendFeedback(j_rads);
    }
}

// **NEW FUNCTION: Send feedback to ROS**
void sendFeedback(float j_rads[6]) {
    // Calculate current positions in radians
    float pos[6];
    pos[0] = stepper1.currentPosition() / STEPS_PER_RAD[0];
    pos[1] = stepper2.currentPosition() / STEPS_PER_RAD[1];
    pos[2] = stepper3.currentPosition() / STEPS_PER_RAD[2];
    pos[3] = stepper4.currentPosition() / STEPS_PER_RAD[3];
    pos[4] = stepper5.currentPosition() / STEPS_PER_RAD[4];
    pos[5] = stepper6.currentPosition() / STEPS_PER_RAD[5];
    
    // Format: <ACK,SEQ,J1,J2,J3,J4,J5,J6>
    char buffer[128];
    snprintf(buffer, sizeof(buffer),
             "<ACK,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f>\n",
             last_seq,
             pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]);
    
    Serial.print(buffer);
}
