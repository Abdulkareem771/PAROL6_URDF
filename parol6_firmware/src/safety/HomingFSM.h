#pragma once
/**
 * HomingFSM.h — Homing state machine for PAROL6 Teensy 4.1
 *
 * Sequence for one axis:
 *   IDLE  → call begin()
 *   SEEKING  → increments a step counter; calls set_motor_velocity() at slow homing speed.
 *              When limit switch fires, latches position and transitions to BACKOFF.
 *   BACKOFF  → moves HOMED_OFFSET_STEPS in the reverse direction to release the switch.
 *   ZEROING  → zeros the encoder & interpolator (caller must perform this via set_zero callbacks).
 *   DONE     → reports success to caller; stops motor.
 *   FAULT    → timeout or max-travel exceeded before switch found.
 *
 * The FSM is designed for non-blocking use:
 *   - tick_1ms() is called from the ISR (or main loop at ≥1kHz) once per millisecond.
 *   - It calls the set_velocity and zero_encoder callbacks you provide.
 *
 * Usage:
 *   HomingFSM fsm;
 *   fsm.configure(axis=0, homing_vel_rad_s=0.3f, backoff_steps=500,
 *                 max_travel_steps=200000, &limit_switch);
 *   fsm.set_velocity_cb([axis](float v){ set_motor_velocity(axis, v); });
 *   fsm.set_zero_cb([axis](){ zero_encoder_and_interpolator(axis); });
 *   fsm.begin();
 *   // Each ms: fsm.tick_1ms();
 *   // Check: fsm.is_done(), fsm.has_faulted()
 */

#include <stdint.h>
#include <functional>

class HomingFSM {
public:
    enum State { IDLE, SEEKING, BACKOFF, ZEROING, DONE, FAULT };

    HomingFSM() : _state(IDLE), _steps(0), _backoff_steps(0), _max_travel(0),
                  _homing_vel(0.3f), _axis(0), _limit_sw(nullptr) {}

    // Types for callbacks — avoids coupling to global functions
    using VelCB  = std::function<void(float)>;   // set velocity (rad/s)
    using ZeroCB = std::function<void()>;          // zero encoder + interpolator

    void configure(int axis, float homing_vel_rad_s, int backoff_steps,
                   int max_travel_steps, void* limit_sw_ptr) {
        _axis         = axis;
        _homing_vel   = homing_vel_rad_s;
        _backoff_steps = backoff_steps;
        _max_travel   = max_travel_steps;
        _limit_sw     = limit_sw_ptr;  // Stored as void* to avoid header cycle; cast in tick
        reset();
    }

    void set_velocity_cb(VelCB cb) { _vel_cb  = cb; }
    void set_zero_cb(ZeroCB cb)    { _zero_cb = cb; }

    void begin() {
        if (_state != IDLE) return;
        _steps = 0;
        _state = SEEKING;
    }

    /**
     * Drive the FSM one tick. Call exactly once per millisecond.
     * Returns current state.
     */
    State tick_1ms(uint32_t now_ms, bool limit_triggered) {
        switch (_state) {
            case SEEKING:
                if (limit_triggered) {
                    _state = BACKOFF;
                    _steps = 0;
                    if (_vel_cb) _vel_cb(0.0f);
                } else if (_steps > _max_travel) {
                    _state = FAULT;
                    if (_vel_cb) _vel_cb(0.0f);
                } else {
                    if (_vel_cb) _vel_cb(-_homing_vel);  // Negative = toward limit
                    _steps++;
                }
                break;

            case BACKOFF:
                if (_steps >= _backoff_steps) {
                    _state = ZEROING;
                    if (_vel_cb) _vel_cb(0.0f);
                } else {
                    if (_vel_cb) _vel_cb(_homing_vel);   // Positive = away from limit
                    _steps++;
                }
                break;

            case ZEROING:
                // Zero the encoder — one-shot
                if (_zero_cb) _zero_cb();
                _zero_cb = nullptr;  // Prevent double-call
                _state = DONE;
                break;

            case DONE:
            case IDLE:
            case FAULT:
            default:
                if (_vel_cb) _vel_cb(0.0f);
                break;
        }
        return _state;
    }

    bool is_done()    const { return _state == DONE; }
    bool has_faulted() const { return _state == FAULT; }
    bool is_active()  const { return _state == SEEKING || _state == BACKOFF || _state == ZEROING; }
    State state()     const { return _state; }

    void reset() {
        _state = IDLE;
        _steps = 0;
    }

private:
    State    _state;
    int      _steps;
    int      _backoff_steps;
    int      _max_travel;
    float    _homing_vel;
    int      _axis;
    void*    _limit_sw;
    VelCB    _vel_cb;
    ZeroCB   _zero_cb;
};


/**
 * HomingSequencer — runs a HomingFSM per axis in HOMING_ORDER sequence.
 * Call tick_1ms() from the main thread at ≥1 kHz.
 */
class HomingSequencer {
public:
    enum SeqState { IDLE, HOMING, DONE, FAULT };

    HomingSequencer() : _seq_state(IDLE), _current_idx(0) {}

    /**
     * Initialise all FSMs from config arrays.
     * @param order        Array of axis indices in homing order (length = num_axes)
     * @param num_axes     Number of axes
     * @param homing_vel   Homing velocity (rad/s) per axis
     * @param backoff      Backoff steps per axis
     * @param max_travel   Max travel steps per axis before FAULT
     * @param vel_cb       Callback: set_motor_velocity(axis, vel)
     * @param zero_cb      Callback: zero_encoder(axis)
     */
    void configure(const int* order, int num_axes,
                   const float* homing_vel,
                   const int* backoff,
                   const int* max_travel,
                   std::function<void(int,float)> vel_cb,
                   std::function<void(int)> zero_cb) {
        _num_axes = num_axes;
        for (int i = 0; i < num_axes; i++) _order[i] = order[i];

        for (int a = 0; a < num_axes; a++) {
            int ax = order[a];
            _fsm[a].configure(ax, homing_vel[ax], backoff[ax], max_travel[ax], nullptr);
            _fsm[a].set_velocity_cb([ax, vel_cb](float v){ vel_cb(ax, v); });
            _fsm[a].set_zero_cb([ax, zero_cb](){ zero_cb(ax); });
        }
    }

    void begin() {
        if (_seq_state != IDLE) return;
        _current_idx = 0;
        _seq_state = HOMING;
        _fsm[0].begin();
    }

    SeqState tick_1ms(uint32_t now_ms, const bool* limit_states) {
        if (_seq_state != HOMING) return _seq_state;

        int ax    = _order[_current_idx];
        bool trig = limit_states[ax];
        HomingFSM::State fs = _fsm[_current_idx].tick_1ms(now_ms, trig);

        if (fs == HomingFSM::FAULT) {
            _seq_state = FAULT;
        } else if (fs == HomingFSM::DONE) {
            _current_idx++;
            if (_current_idx >= _num_axes) {
                _seq_state = DONE;
            } else {
                _fsm[_current_idx].begin();
            }
        }
        return _seq_state;
    }

    bool is_done()    const { return _seq_state == DONE; }
    bool has_faulted() const { return _seq_state == FAULT; }
    bool is_active()  const { return _seq_state == HOMING; }

    void reset() {
        _seq_state   = IDLE;
        _current_idx = 0;
        for (int i = 0; i < _num_axes; i++) _fsm[i].reset();
    }

private:
    static constexpr int MAX_AXES = 6;
    SeqState      _seq_state;
    int           _current_idx;
    int           _num_axes = 6;
    int           _order[MAX_AXES] = {};
    HomingFSM     _fsm[MAX_AXES];
};
