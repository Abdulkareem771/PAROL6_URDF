#ifndef SERVO42C_H
#define SERVO42C_H

#include <Arduino.h>

// ----------------------------------------------------------
// ENUMS FOR USER-FRIENDLY CODE
// ----------------------------------------------------------

enum Servo42C_MotorType { MOTOR_0_9_DEG = 0, MOTOR_1_8_DEG = 1 };
enum Servo42C_WorkMode  { MODE_OPEN = 0, MODE_VFOC = 1, MODE_UART = 2 };
enum Servo42C_EnablePin { EN_LOW = 0, EN_HIGH = 1, EN_ALWAYS = 2 };
enum Servo42C_Direction { DIR_CW = 0, DIR_CCW = 1 };
enum Servo42C_BaudRate  { B9600=1, B19200, B25000, B38400, B57600, B115200 };
enum ZeroMode           { ZERO_DISABLE=0, ZERO_DIRMODE, ZERO_NEARMODE };

// UART control run-speed direction bit
enum RunDirection { RUN_FWD=0, RUN_REV=1 };

// ----------------------------------------------------------
// DATA STRUCTS
// ----------------------------------------------------------

struct EncoderReading {
    int32_t carry;
    uint16_t value;
};

class SERVO42C {
public:
    SERVO42C(HardwareSerial& port, uint8_t addr = 0xE0);

    // Core
    bool begin(uint32_t baud);

    // -----------------------
    // READ COMMANDS
    // -----------------------
    bool readEncoder(EncoderReading &enc);
    bool readPulses(int32_t &pulses);
    bool readAngleError(int16_t &err);
    bool readEnablePin(uint8_t &state);
    bool readProtectState(uint8_t &state);

    // -----------------------
    // SET PARAMETER COMMANDS
    // -----------------------
    bool calibrate();
    bool setMotorType(Servo42C_MotorType type);
    bool setWorkMode(Servo42C_WorkMode mode);
    bool setCurrent(uint8_t ma);
    bool setMicrostep(uint8_t steps);
    bool setEnablePin(Servo42C_EnablePin en);
    bool setMotorDir(Servo42C_Direction dir);
    bool setAutoScreen(bool enable);
    bool setProtect(bool enable);
    bool setInterpolator(bool enable);
    bool setUartBaud(Servo42C_BaudRate baud);
    bool setAddress(uint8_t newAddr);
    bool restoreDefaults();

    // -----------------------
    // ZERO MODE COMMANDS
    // -----------------------
    bool zeroModeSet(ZeroMode mode);
    bool zeroModeSetZero();
    bool zeroModeSetSpeed(uint8_t spd);
    bool zeroModeSetDir(Servo42C_Direction dir);
    bool zeroModeGo();

    // -----------------------
    // PID / TORQUE / ACC
    // -----------------------
    bool setKp(uint16_t kp);
    bool setKi(uint16_t ki);
    bool setKd(uint16_t kd);
    bool setACC(uint16_t acc);
    bool setMaxTorque(uint16_t maxT);

    // -----------------------
    // UART RUN COMMANDS
    // -----------------------
    bool uartEnable(bool enable);
    bool uartRunSpeed(uint8_t speed, RunDirection dir);
    bool uartStop();
    bool uartSaveRunState();
    bool uartClearRunState();
    bool uartRunPulses(uint8_t speed, RunDirection dir, uint32_t pulses, uint8_t &statusOut);

private:
    HardwareSerial* serial;
    uint8_t addr;

    void flushInput();
    uint8_t calcCRC(uint8_t *data, uint8_t len);
    bool sendCmd(uint8_t func);
    bool sendCmd1(uint8_t func, uint8_t d0);
    bool sendCmd2(uint8_t func, uint16_t val);
    bool sendCmd4(uint8_t func, uint32_t val);

    bool readReply(uint8_t expectedAddr, uint8_t expectedLen, uint8_t *buf);
};

#endif
