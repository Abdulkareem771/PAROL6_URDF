#pragma once
/**
 * LimitSwitch.h — Debounced limit switch HAL for PAROL6 Teensy 4.1
 *
 * Inductive NPN (open-collector, through optocoupler):
 *   Normal: Teensy pin HIGH (INPUT_PULLUP).  Triggered: pin pulled LOW → active_high = false.
 *
 * Inductive PNP (sourcing, through optocoupler):
 *   Normal: Teensy pin LOW (INPUT_PULLDOWN). Triggered: pin driven HIGH → active_high = true.
 *
 * Mechanical NC (NO optocoupler):
 *   Normal: pin HIGH (INPUT_PULLUP). Triggered: pin LOW → active_high = false.
 *
 * Debounce: requires 3 consecutive samples spaced ≥1 ms apart to confirm a transition.
 * This prevents stepper-motor EMI from ghost-triggering the supervisor.
 */

#include <Arduino.h>

class LimitSwitch {
public:
    /**
     * @param pin        Teensy GPIO pin number
     * @param pull_mode  Arduino pin mode: INPUT, INPUT_PULLUP, or INPUT_PULLDOWN
     * @param active_high  true if HIGH = triggered, false if LOW = triggered
     */
    LimitSwitch() : _pin(0), _pull_mode(INPUT_PULLUP), _active_high(false),
                    _enabled(false), _confirmed(false), _sample_count(0), _last_sample_ms(0) {}

    void configure(int pin, int pull_mode, bool active_high, bool enabled) {
        _pin         = pin;
        _pull_mode   = pull_mode;
        _active_high = active_high;
        _enabled     = enabled;
        _confirmed   = false;
        _sample_count = 0;
        _last_sample_ms = 0;
    }

    void init() {
        if (!_enabled) return;
        pinMode(_pin, _pull_mode);
    }

    /**
     * Call every millisecond from the main ISR or main loop.
     * Returns true once 3 consecutive reads confirm a trigger.
     * The confirmation latches until cleared with reset().
     */
    bool update(uint32_t now_ms) {
        if (!_enabled) return false;
        if (_confirmed) return true;  // Latch until reset

        if (now_ms - _last_sample_ms < 1) return false;
        _last_sample_ms = now_ms;

        bool raw = (digitalReadFast(_pin) == HIGH) == _active_high;

        if (raw) {
            _sample_count++;
            if (_sample_count >= DEBOUNCE_SAMPLES) {
                _confirmed = true;
            }
        } else {
            _sample_count = 0; // Reset on any LOW (glitch rejection)
        }
        return _confirmed;
    }

    /** Unlatch after homing completes or ESTOP is cleared. */
    void reset() {
        _confirmed    = false;
        _sample_count = 0;
    }

    bool is_triggered() const { return _confirmed; }
    bool is_enabled()   const { return _enabled; }
    int  pin()          const { return _pin; }

private:
    static constexpr int DEBOUNCE_SAMPLES = 3;

    int      _pin;
    int      _pull_mode;
    bool     _active_high;
    bool     _enabled;
    bool     _confirmed;
    int      _sample_count;
    uint32_t _last_sample_ms;
};
