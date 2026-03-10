#pragma once

#include <Arduino.h>
#include <CircularBuffer.h>

// Select physical transport based on GUI config.h setting.
// TRANSPORT_MODE: 0 = UART (Serial1 pins 0/1), 1 = USB CDC HS (Native Serial), 2 = Ethernet (UDP)
#if defined(TRANSPORT_MODE) && TRANSPORT_MODE == 0
#  define SERIAL_DEV Serial1
#elif defined(TRANSPORT_MODE) && TRANSPORT_MODE == 1
#  define SERIAL_DEV Serial
#else
#  define SERIAL_DEV Serial
#endif

struct RosCommand {
    uint32_t seq;
    float positions[6];
    float velocities[6];
    uint32_t timestamp_us;
    bool is_home_cmd;
    bool is_enable_cmd;
};

class SerialTransport {
public:
    void init(uint32_t baud) {
        SERIAL_DEV.begin(baud);
    }


    // Called in main loop (Non-blocking)
    void process_incoming(CircularBuffer<RosCommand, 20>& cmd_queue) {
        while (SERIAL_DEV.available()) {
            char c = SERIAL_DEV.read();
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

    void send_feedback(uint32_t seq, const float current_pos[6], const float current_vel[6],
                       uint8_t lim_state = 0) {
        // Format: <ACK,seq,p1..p6,v1..v6,lim_state>
        SERIAL_DEV.print("<ACK,");
        SERIAL_DEV.print(seq);
        for (int i = 0; i < 6; i++) {
            SERIAL_DEV.print(",");
            SERIAL_DEV.print(current_pos[i], 4);
        }
        for (int i = 0; i < 6; i++) {
            SERIAL_DEV.print(",");
            SERIAL_DEV.print(current_vel[i], 4);
        }
        SERIAL_DEV.print(",");
        SERIAL_DEV.print(lim_state);  // bitmask: bit0=J1...bit5=J6
        SERIAL_DEV.println(">");
    }

    /** Send a raw string (e.g., "HOMING_DONE\n"). */
    void send_string(const char* msg) {
        SERIAL_DEV.print(msg);
    }

private:
    static const int MAX_BUF = 256;
    char rx_buf_[MAX_BUF];
    int rx_pos_ = 0;
    
    // Quick parse: <SEQ,p1,p2,p3,p4,p5,p6,v1,v2,v3,v4,v5,v6>
    // Or special command: <HOME>
    bool parse_string(char* str, RosCommand& cmd) {
        if (str[0] != '<' || str[strlen(str) - 1] != '>') return false;
        
        cmd.is_home_cmd = false;
        cmd.is_enable_cmd = false;
        
        // Remove brackets
        str[strlen(str) - 1] = '\0';
        char* pt = str + 1;
        
        if (strcmp(pt, "HOME") == 0) {
            cmd.is_home_cmd = true;
            return true;
        }
        if (strcmp(pt, "ENABLE") == 0) {
            cmd.is_enable_cmd = true;
            return true;
        }
        
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
