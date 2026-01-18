/**
 * PAROL6 ESP32 Benchmark Firmware
 * 
 * PURPOSE: Test ROS-ESP32 communication integrity BEFORE connecting motors
 * 
 * FEATURES:
 * - Receives trajectory waypoints (6 joint positions + velocities + accelerations)
 * - Logs received data with microsecond timestamps
 * - Sends ACK back to laptop for latency measurement
 * - Generates log file on SD card (optional) AND serial output
 * - Detects packet loss by sequence numbers
 * 
 * PROTOCOL:
 * Incoming: <SEQ,J1,J2,J3,J4,J5,J6,V1,V2,V3,V4,V5,V6,A1,A2,A3,A4,A5,A6>
 * Response: <ACK,SEQ,TIMESTAMP_US>
 */

// Configuration
#define USE_SD_LOGGING false  // Set true if SD card attached
#define BAUD_RATE 115200
#define LOG_TO_SERIAL true

// Includes
#if USE_SD_LOGGING
  #include <SD.h>
  #include <SPI.h>
  #define SD_CS_PIN 5
  File logFile;
#endif

// Stats tracking
struct CommStats {
  unsigned long packetsReceived = 0;
  unsigned long packetsLost = 0;
  unsigned long lastSeqNum = 0;
  unsigned long totalLatencyUs = 0;
  unsigned long maxLatencyUs = 0;
  unsigned long minLatencyUs = 999999;
};

CommStats stats;

// Buffer for serial
const int BUFFER_SIZE = 256;
char rxBuffer[BUFFER_SIZE];
int bufferIndex = 0;
bool messageComplete = false;

void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial) { delay(10); }
  
  #if USE_SD_LOGGING
    if (!SD.begin(SD_CS_PIN)) {
      Serial.println("ERROR: SD Card Mount Failed!");
    } else {
      logFile = SD.open("/comm_test.log", FILE_WRITE);
      if (logFile) {
        logFile.println("Timestamp_us,Seq,J1,J2,J3,J4,J5,J6,V1,V2,V3,V4,V5,V6,A1,A2,A3,A4,A5,A6");
        logFile.close();
      }
    }
  #endif
  
  Serial.println("READY: ESP32_BENCHMARK_V1");
  Serial.println("Waiting for commands...");
}

void loop() {
  // Read serial input
  while (Serial.available() > 0) {
    char inChar = Serial.read();
    
    if (inChar == '<') {
      // Start of message
      bufferIndex = 0;
      messageComplete = false;
    } else if (inChar == '>') {
      // End of message
      rxBuffer[bufferIndex] = '\0';
      messageComplete = true;
      processMessage();
    } else if (bufferIndex < BUFFER_SIZE - 1) {
      rxBuffer[bufferIndex++] = inChar;
    }
  }
}

void processMessage() {
  unsigned long receiveTime = micros();
  
  // Parse message: SEQ,J1,J2,J3,J4,J5,J6,V1,V2,V3,V4,V5,V6,A1,A2,A3,A4,A5,A6
  float values[19];  // 1 seq + 6 pos + 6 vel + 6 acc
  int valueCount = 0;
  
  char* token = strtok(rxBuffer, ",");
  while (token != NULL && valueCount < 19) {
    values[valueCount++] = atof(token);
    token = strtok(NULL, ",");
  }
  
  if (valueCount == 19) {
    unsigned long seqNum = (unsigned long)values[0];
    
    // Detect packet loss
    if (stats.packetsReceived > 0) {
      unsigned long expected = stats.lastSeqNum + 1;
      if (seqNum != expected) {
        stats.packetsLost += (seqNum - expected);
      }
    }
    
    stats.lastSeqNum = seqNum;
    stats.packetsReceived++;
    
    // Send ACK immediately
    Serial.print("<ACK,");
    Serial.print(seqNum);
    Serial.print(",");
    Serial.print(receiveTime);
    Serial.println(">");
    
    // Log to SD card
    #if USE_SD_LOGGING
      logFile = SD.open("/comm_test.log", FILE_APPEND);
      if (logFile) {
        logFile.print(receiveTime);
        for (int i = 0; i < valueCount; i++) {
          logFile.print(",");
          logFile.print(values[i], 4);
        }
        logFile.println();
        logFile.close();
      }
    #endif
    
    // Log to serial (optional)
    #if LOG_TO_SERIAL
      Serial.print("LOG,");
      Serial.print(receiveTime);
      Serial.print(",SEQ:");
      Serial.print(seqNum);
      Serial.print(",J:[");
      for (int i = 1; i <= 6; i++) {
        Serial.print(values[i], 3);
        if (i < 6) Serial.print(",");
      }
      Serial.println("]");
    #endif
    
  } else {
    Serial.println("ERROR: Invalid packet format");
  }
}

// Print stats on request
void printStats() {
  Serial.println("\n=== Communication Statistics ===");
  Serial.print("Packets Received: "); Serial.println(stats.packetsReceived);
  Serial.print("Packets Lost: "); Serial.println(stats.packetsLost);
  if (stats.packetsReceived > 0) {
    float lossRate = (stats.packetsLost * 100.0) / (stats.packetsReceived + stats.packetsLost);
    Serial.print("Loss Rate: "); Serial.print(lossRate, 2); Serial.println("%");
  }
  Serial.println("================================\n");
}
