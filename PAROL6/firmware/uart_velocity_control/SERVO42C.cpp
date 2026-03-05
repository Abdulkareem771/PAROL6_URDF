#include "SERVO42C.h"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
SERVO42C::SERVO42C(HardwareSerial& port, uint8_t addr)
{
    this->serial = &port;
    this->addr = addr;
}

// ---------------------------------------------------------------------------
// Begin UART
// ---------------------------------------------------------------------------
bool SERVO42C::begin(uint32_t baud)
{
    serial->begin(baud);
    delay(100);
    flushInput();
    return true;
}

// ---------------------------------------------------------------------------
// Flush serial input
// ---------------------------------------------------------------------------
void SERVO42C::flushInput()
{
    while (serial->available())
        serial->read();
}

// ---------------------------------------------------------------------------
// CRC8 = simple sum of bytes (8-bit)
// ---------------------------------------------------------------------------
uint8_t SERVO42C::calcCRC(uint8_t* data, uint8_t len)
{
    uint16_t sum = 0;
    for (uint8_t i = 0; i < len; i++)
        sum += data[i];
    return (uint8_t)(sum & 0xFF);
}

// ---------------------------------------------------------------------------
// Send Short Commands
// ---------------------------------------------------------------------------
bool SERVO42C::sendCmd(uint8_t func)
{
    uint8_t buf[3] = { addr, func, (uint8_t)((addr + func) & 0xFF) };
    serial->write(buf, 3);
    return true;
}

bool SERVO42C::sendCmd1(uint8_t func, uint8_t d0)
{
    uint8_t buf[4] = { addr, func, d0 };
    buf[3] = calcCRC(buf, 3);
    serial->write(buf, 4);
    return true;
}

bool SERVO42C::sendCmd2(uint8_t func, uint16_t val)
{
    uint8_t buf[5] = {
        addr, func,
        (uint8_t)(val >> 8), (uint8_t)(val & 0xFF)
    };
    buf[4] = calcCRC(buf, 4);
    serial->write(buf, 5);
    return true;
}

bool SERVO42C::sendCmd4(uint8_t func, uint32_t val)
{
    uint8_t buf[7] = {
        addr, func,
        (uint8_t)(val >> 24), (uint8_t)(val >> 16),
        (uint8_t)(val >> 8), (uint8_t)(val)
    };
    buf[6] = calcCRC(buf, 6);
    serial->write(buf, 7);
    return true;
}

// ---------------------------------------------------------------------------
// readReply() – simple, per-call implementation
// We assume:
//   - RX is flushed just before the command (see readEncoder/readPulses/...)
//   - The servo replies with exactly `expectedLen` bytes starting with addr
// This avoids any cross-call state and greatly reduces chance of "missed" frames.
// ---------------------------------------------------------------------------
bool SERVO42C::readReply(uint8_t expectedAddr, uint8_t expectedLen, uint8_t* outFrame)
{
    const uint32_t timeout = 5;          // total timeout for a full frame
    uint32_t start = millis();

    // 1) Wait for first byte == expectedAddr
    while (millis() - start < timeout)
    {
        if (!serial->available())
            continue;

        uint8_t b = serial->read();
        if (b != expectedAddr)
            continue;                      // ignore garbage until we see address

        outFrame[0] = b;
        break;
    }

    // If we still haven't got the first byte, fail
    if (outFrame[0] != expectedAddr)
        return false;

    // 2) Read remaining bytes of this frame
    for (uint8_t i = 1; i < expectedLen; i++)
    {
        while (!serial->available())
        {
            if (millis() - start >= timeout)
                return false;              // timeout while waiting for rest of frame
        }
        outFrame[i] = serial->read();
    }

    // 3) CRC check (last byte is CRC)
    uint8_t crcCalc = calcCRC(outFrame, expectedLen - 1);
    if (crcCalc != outFrame[expectedLen - 1])
        return false;

    return true;
}

// ---------------------------------------------------------------------------
// READ COMMANDS
// ---------------------------------------------------------------------------
bool SERVO42C::readEncoder(EncoderReading& enc)
{
    uint8_t buf[8];

    // No flush here — preserve pipeline
    sendCmd(0x30);

    if (!readReply(addr, 8, buf))
        return false;

    enc.carry =
        ((int32_t)buf[1] << 24) |
        ((int32_t)buf[2] << 16) |
        ((int32_t)buf[3] << 8) |
        buf[4];

    enc.value = (uint16_t)((buf[5] << 8) | buf[6]);
    return true;
}

bool SERVO42C::readAngleError(int16_t& err)
{   flushInput(); 
    sendCmd(0x39);
    uint8_t buf[4];

    if (!readReply(addr, 4, buf))
        return false;

    err = (int16_t)((buf[1] << 8) | buf[2]);
    return true;
}

bool SERVO42C::readEnablePin(uint8_t& state)
{   flushInput(); 
    sendCmd(0x3A);
    uint8_t buf[3];

    if (!readReply(addr, 3, buf))
        return false;

    state = buf[1];
    return true;
}

bool SERVO42C::readProtectState(uint8_t& state)
{
    sendCmd(0x3E);
    uint8_t buf[3];

    if (!readReply(addr, 3, buf))
        return false;

    state = buf[1];
    return true;
}

// ---------------------------------------------------------------------------
// ZERO MODE / SETTINGS / PID / UART CONTROL
// (same as earlier OOP code — unchanged except they now use bulletproof readReply)
// ---------------------------------------------------------------------------

bool SERVO42C::calibrate() { sendCmd1(0x80, 0); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setMotorType(Servo42C_MotorType t) { sendCmd1(0x81, t); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setWorkMode(Servo42C_WorkMode m) { sendCmd1(0x82, m); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setCurrent(uint8_t ma) { sendCmd1(0x83, ma); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setMicrostep(uint8_t s) { sendCmd1(0x84, s); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setEnablePin(Servo42C_EnablePin e) { sendCmd1(0x85, e); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setMotorDir(Servo42C_Direction d) { sendCmd1(0x86, d); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setAutoScreen(bool en) { sendCmd1(0x87, en?1:0); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setProtect(bool en) { sendCmd1(0x88, en?1:0); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setInterpolator(bool en) { sendCmd1(0x89, en?1:0); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setUartBaud(Servo42C_BaudRate b) { sendCmd1(0x8A, b); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setAddress(uint8_t newAddr) { sendCmd1(0x8B, newAddr); uint8_t v; return readEnablePin(v); }
bool SERVO42C::restoreDefaults() { sendCmd(0x3F); uint8_t v; return readEnablePin(v); }

bool SERVO42C::zeroModeSet(ZeroMode m) { sendCmd1(0x90, m); uint8_t v; return readEnablePin(v); }
bool SERVO42C::zeroModeSetZero() { sendCmd1(0x91, 0); uint8_t v; return readEnablePin(v); }
bool SERVO42C::zeroModeSetSpeed(uint8_t spd) { sendCmd1(0x92, spd); uint8_t v; return readEnablePin(v); }
bool SERVO42C::zeroModeSetDir(Servo42C_Direction d) { sendCmd1(0x93, d); uint8_t v; return readEnablePin(v); }
bool SERVO42C::zeroModeGo() { sendCmd1(0x94, 0); uint8_t v; return readEnablePin(v); }

bool SERVO42C::setKp(uint16_t kp) { sendCmd2(0xA1, kp); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setKi(uint16_t ki) { sendCmd2(0xA2, ki); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setKd(uint16_t kd) { sendCmd2(0xA3, kd); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setACC(uint16_t acc) { sendCmd2(0xA4, acc); uint8_t v; return readEnablePin(v); }
bool SERVO42C::setMaxTorque(uint16_t maxT) { sendCmd2(0xA5, maxT); uint8_t v; return readEnablePin(v); }

bool SERVO42C::uartEnable(bool e) { sendCmd1(0xF3, e?1:0); uint8_t v; return readEnablePin(v); }

bool SERVO42C::uartRunSpeed(uint8_t speed, RunDirection dir)
{
    uint8_t val = (dir ? 0x80 : 0x00) | (speed & 0x7F);
    sendCmd1(0xF6, val);
    uint8_t v;
    return readEnablePin(v);
}

bool SERVO42C::uartStop()
{
    sendCmd(0xF7);
    uint8_t v;
    return readEnablePin(v);
}

bool SERVO42C::uartSaveRunState()
{
    sendCmd1(0xFF, 0xC8);
    uint8_t v;
    return readEnablePin(v);
}

bool SERVO42C::uartClearRunState()
{
    sendCmd1(0xFF, 0xCA);
    uint8_t v;
    return readEnablePin(v);
}



bool SERVO42C::uartRunPulses(
    uint8_t speed,
    RunDirection dir,
    uint32_t pulses,
    uint8_t &statusOut
)
{
    uint8_t val = (dir ? 0x80 : 0x00) | (speed & 0x7F);

    uint8_t buf[8];
    buf[0] = addr;
    buf[1] = 0xFD;
    buf[2] = val;
    buf[3] = (pulses >> 24) & 0xFF;
    buf[4] = (pulses >> 16) & 0xFF;
    buf[5] = (pulses >> 8)  & 0xFF;
    buf[6] = pulses & 0xFF;
    buf[7] = calcCRC(buf, 7);

    serial->write(buf, 8);

    uint8_t reply[3];
    if (!readReply(addr, 3, reply))
        return false;

    statusOut = reply[1];  // 0=fail, 1=running, 2=complete
    return true;
}
