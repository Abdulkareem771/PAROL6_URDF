/*
 * PAROL6 Real-Time Servo Control — Main (Teensy 4.1)
 *
 * Architecture:
 *   IntervalTimer (hardware)  →  controlUpdate()   @ 500 Hz (2 ms)
 *   Main loop()               →  serial I/O        @ 50 Hz  (20 ms)
 *
 * Why IntervalTimer instead of FreeRTOS:
 *   - Teensy 4.1 has 4 hardware timers with sub-microsecond precision
 *   - No FreeRTOS overhead (context switching, stack allocation, FPU save)
 *   - controlUpdate() runs in ISR context → guaranteed timing
 *   - Serial I/O runs in main loop (can't do USB I/O in ISR)
 *
 * Teensy 4.1 @ 600 MHz:
 *   - controlUpdate() takes ~30 µs (6 motors × encoder decode + control law)
 *   - 2000 µs period = 1.5% CPU utilisation for control
 */

#include "config.h"
#include "motor.h"
#include "control.h"
#include "serial_comm.h"

// ============================================================================
// HARDWARE TIMER
// ============================================================================

IntervalTimer controlTimer;

// ============================================================================
// SERIAL TIMING (software, in main loop)
// ============================================================================

static elapsedMicros serialElapsed;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  // Initialize subsystems
  serialCommInit();
  motorsInit();
  controlInit();

  // Start control loop — hardware timer, ISR context, guaranteed 500 Hz
  controlTimer.begin(controlUpdate, CONTROL_PERIOD_US);

  // Priority: ensure encoder ISRs (GPIO, priority 128) run BEFORE control ISR.
  // If control ISR preempts encoder ISR, pulse width measurement is corrupted
  // by ~30µs, causing multi-turn tracking to fail completely.
  // Priority 144 = lower than GPIO default 128 = encoder ISR runs first.
  controlTimer.priority(144);  // MUST be > 128 (GPIO default)

  // Reset serial timer
  serialElapsed = 0;
}

// ============================================================================
// MAIN LOOP — serial I/O at 50 Hz
// ============================================================================

void loop() {
  // Process incoming commands (non-blocking)
  serialCommProcessIncoming();

  // Send feedback at 50 Hz
  if (serialElapsed >= FEEDBACK_PERIOD_US) {
    serialElapsed -= FEEDBACK_PERIOD_US;
    serialCommSendFeedback();
  }
}
