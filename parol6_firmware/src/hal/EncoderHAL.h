#pragma once

class EncoderHAL {
public:
    virtual ~EncoderHAL() = default;
    
    // Abstract hardware configuration setup
    virtual void init() = 0;
    
    // Returns raw, absolute rotor angle in radians [-PI, PI] or [0, 2PI]
    // The AlphaBeta observer will handle wrapping.
    virtual float read_angle() = 0;
};
