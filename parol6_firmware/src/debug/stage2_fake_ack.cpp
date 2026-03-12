/**
 * stage2_fake_ack.cpp — Stage 2: Fake ACK Packets (hardcoded zeros)
 *
 * WHAT THIS TESTS:
 *   - Full ACK packet format is parseable by the GUI Serial tab
 *   - ros2_control parol6_system.cpp can parse the packets
 *   - No encoder or stepper needed yet
 *
 * SUCCESS LOOKS LIKE:
 *   - GUI Serial tab shows "<ACK,N,0.0,...>" packets at ~25 Hz
 *   - All 6 joints show 0.000 in the Jog tab encoder readouts
 *   - RViz shows the robot frozen at all-zeros (home) position
 *   - Sending "<ENABLE>" returns "<ENABLE_ACK>"
 *
 * NO physical hardware needed for this stage.
 */

#include <Arduino.h>

static uint32_t seq      = 0;
static bool     enabled  = false;
static uint32_t last_fb  = 0;

// Parse a line: returns true if it was a known command
static bool handle_line(const String& line) {
    if (line == "<ENABLE>") {
        enabled = true;
        Serial.println("<ENABLE_ACK>");
        return true;
    }
    if (line == "<HOME>") {
        Serial.println("<HOMING_DONE>");
        return true;
    }
    // Any position command: just acknowledge
    if (line.startsWith("<") && line.endsWith(">")) {
        // Silently accept — no motion at this stage
        return true;
    }
    return false;
}

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(115200);
    delay(500);
    Serial.println("<STAGE2_READY>");
    Serial.println("# Sending fake ACK packets at 25 Hz. Positions are all 0.");
    Serial.println("# Send <ENABLE> to flip state_byte from 3 -> 1.");
}

void loop() {
    // Collect incoming lines
    static String buf;
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            buf.trim();
            if (buf.length() > 0) handle_line(buf);
            buf = "";
        } else {
            buf += c;
        }
    }

    // Fake ACK at 25 Hz
    uint32_t now = millis();
    if (now - last_fb >= 40) {
        last_fb = now;
        digitalToggleFast(LED_BUILTIN);

        uint8_t state_byte = enabled ? 1 : 3;  // 1=NOMINAL, 3=DISABLED
        // Format: <ACK,seq,p1..p6,v1..v6,lim_state,state_byte>
        Serial.print("<ACK,"); Serial.print(seq++);
        Serial.print(",0.0000,0.0000,0.0000,0.0000,0.0000,0.0000");  // positions
        Serial.print(",0.0000,0.0000,0.0000,0.0000,0.0000,0.0000");  // velocities
        Serial.print(",0,");
        Serial.print(state_byte);
        Serial.println(">");
    }
}
