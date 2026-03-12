#pragma once

class EncoderHAL {
public:
    virtual ~EncoderHAL() = default;
    virtual void init() = 0;
    virtual float read_angle() = 0;
};

