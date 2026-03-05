#pragma once

#include <Arduino.h>
#include <CircularBuffer.h>

struct RosCommand {
    uint32_t seq;
    float positions[6];
    float velocities[6];
    uint32_t timestamp_us;
};

class SerialTransport {
public:
    void init(uint32_t baud) {
        Serial.begin(baud);
    }

    // Called in main loop (Non-blocking)
    void process_incoming(CircularBuffer<RosCommand, 20>& cmd_queue) {
        while (Serial.available()) {
            char c = Serial.read();
            if (c == '\n' || c == '\r') {
                if (rx_pos_ > 0) {
                    rx_buf_[rx_pos_] = '\0';
                    RosCommand new_cmd;
                    if (parse_string(rx_buf_, new_cmd)) {
                        cmd_queue.push(new_cmd); 
                    }
                    rx_pos_ = 0;
                }
            } else if (rx_pos_ < MAX_BUF - 1) {
                rx_buf_[rx_pos_++] = c;
            }
        }
    }

    void send_feedback(uint32_t seq, const float current_pos[6], const float current_vel[6]) {
        // Format: <ACK,seq,p1,p2,p3,p4,p5,p6,v1,v2,v3,v4,v5,v6>
        Serial.print("<ACK,");
        Serial.print(seq);
        for (int i = 0; i < 6; i++) {
            Serial.print(",");
            Serial.print(current_pos[i], 4);
        }
        for (int i = 0; i < 6; i++) {
            Serial.print(",");
            Serial.print(current_vel[i], 4);
        }
        Serial.println(">");
    }

private:
    static const int MAX_BUF = 256;
    char rx_buf_[MAX_BUF];
    int rx_pos_ = 0;
    
    // Quick parse: <SEQ,p1,p2,p3,p4,p5,p6,v1,v2,v3,v4,v5,v6>
    bool parse_string(char* str, RosCommand& cmd) {
        if (str[0] != '<' || str[strlen(str) - 1] != '>') return false;
        
        // Remove brackets
        str[strlen(str) - 1] = '\0';
        char* pt = str + 1;
        
        char* token = strtok(pt, ",");
        if (!token) return false;
        
        // Actually, matching the original format: <SEQ, J1, J2...>
        cmd.seq = strtoul(token, NULL, 10);
        
        for (int i = 0; i < 6; i++) {
            token = strtok(NULL, ",");
            if (!token) return false;
            cmd.positions[i] = atof(token);
        }

        // Optional velocities (backward compatible)
        for (int i = 0; i < 6; i++) {
            token = strtok(NULL, ",");
            if (token) {
                cmd.velocities[i] = atof(token);
            } else {
                cmd.velocities[i] = 0.0f;
            }
        }
        
        cmd.timestamp_us = micros();
        return true;
    }
};
