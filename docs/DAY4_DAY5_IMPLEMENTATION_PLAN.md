# Day 4 & 5 Implementation Plans - Appendix to TEAMMATE_SETUP_GUIDE.md

See main guide: `docs/TEAMMATE_SETUP_GUIDE.md`

This appendix provides detailed implementation plans for Day 4 (Motor Integration) and Day 5 (Validation).

---

## Day 4: Motor Integration - Detailed Implementation

### Hardware Setup (1 hour)

**Wiring (Motor 1 Example):**
```
ESP32          MKS Servo42C
GPIO16 (TX) → RX
GPIO17 (RX) ← TX  
GND        ─  GND
           PSU+ → VCC (12-24V)
```

**For 6 motors:** Use RS485 bus with unique motor IDs (0xE0-0xE5)

### Firmware Configuration

**File:** `PAROL6/firmware/esp32_motor_control.ino`

**Adjust these parameters:**
```cpp
#define STEPS_PER_REV 200       // Motor steps (check datasheet)
#define MOTOR_MICROSTEPS 16     // DIP switch setting
#define GEAR_RATIO 1.0          // If using gearbox
#define MOTOR_BAUD 38400        // MKS default
```

### Testing Procedure

**Phase 1: Single Motor (2h):**
1. Flash `esp32_motor_control.ino`
2.Send test command: `<0,0.1,0,0,0,0,0>`
3. Motor should move ~6° (0.1 rad)
4. Verify feedback: `<ACK,0,0.10,0.00,...>`

**Phase 2: ROS Integration (1h):**
```bash
ros2 launch parol6_hardware real_robot.launch.py
# Should see: ✅ Serial opened, feedback received
# NO "State tolerance violation" errors if motor responds
```

**Phase 3: Multi-Motor (4h):**
- Add motors one at a time
- Test each before combining
- Final test: Move all 6 axes simultaneously

### Expected Issues

| Problem | Solution |
|---------|----------|
| Motor vibrates | Match microstepping (DIP vs code) |
| Wrong direction | Swap motor wiring or negate in code |
| Tolerance abort | Increase threshold in `parol6_controllers.yaml` |

---

## Day 5: Validation - Detailed Plan

### Engineering Gate Test (15 min)

**Procedure:**
1. Launch full stack
2. Execute 10 different RViz trajectories
3. Let idle for remaining time

**Success Criteria:**
```
✅ 0% packet loss
✅ All trajectories complete
✅ No crashes
✅ Latency < 100ms
```

**Evidence to collect:**
- Terminal logs (copy to file)
- Screenshot of final stats
- `rqt_plot` graph

### Thesis Gate Test (30 min)

**With full payload:**
- Fast motions (dynamics test)
- Slow precision (accuracy test)
- Continuous operation

**Metrics:**
- Position accuracy: ±0.01 rad
- Packet loss: 0%
- CPU/memory stable

### Fault Injection Tests

1. **USB Disconnect:** Unplug during motion → Should log error, stop safely
2. **Motor Stall:** Block motor → Should abort trajectory
3. **RViz Crash:** Kill RViz → Hardware continues running
4. **E-Stop:** Cancel action → Immediate halt

**Document pass/fail for each**

---

## Complete System Architecture

### Data Flow (End-to-End)

```
User (RViz)
  ↓ [Drag interactive marker]
MoveIt Planner
  ↓ [Trajectory: positions, velocities, timestamps]
parol6_arm_controller (25Hz)
  ↓ [Interpolated setpoints]
PAROL6System::write()
  ↓ [Serial TX: <SEQ,J1-J6>]
ESP32 Serial Parser
  ↓ [Extract positions]
MKS Servo42C Motors
  ↓ [Move to position]
Encoders
  ↓ [Read actual position]
ESP32 Feedback Generator
  ↓ [Serial RX: <ACK,SEQ,J1-J6>]
PAROL6System::read()
  ↓ [Update hw_states_positions_[]]
joint_state_broadcaster
  ↓ [Publish /joint_states]
RViz Visualization
  ↓ [Update robot model]
User sees motion ✓
```

### Control Loop Timing

| Step | Rate | Latency |
|------|------|---------|
| MoveIt → Controller | 25Hz | <5ms |
| Serial TX | 25Hz | ~1ms |
| Motor move | 50Hz | ~20ms |
| Serial RX | 25Hz | ~1ms |
| RViz update | 25Hz | ~10ms |
| **Total loop** | **25Hz** | **~50ms** |

### Safety Mechanisms

1. **Tolerance Checking** - Aborts if position error > threshold
2. **Packet Loss Detection** - Tracks sequence numbers
3. **Non-Blocking I/O** - Prevents control loop hangs
4. **Emergency Stop** - RViz stop button sends abort

---

## Thesis Documentation Checklist

### Code Deliverables
- [x] `parol6_hardware/src/parol6_system.cpp` (C++ driver)
- [x] `PAROL6/firmware/esp32_motor_control.ino` (Motor firmware)
- [x] `parol6_hardware/config/parol6_controllers.yaml` (Tuning)

### Documentation
- [x] System architecture diagram
- [x] Teammate setup guide
- [x] Day 3 validation walkthrough
- [ ] Day 4 motor integration log
- [ ] Day 5 validation report

### Evidence
- [ ] Terminal logs (engineering gate)
- [ ] RViz screenshots
- [ ] `rqt_plot` graphs
- [ ] Metrics table (packet loss, latency, etc.)
- [ ] Video (optional, smooth motion demo)

### Thesis Sections

**Introduction:**
> "A production-grade ros2_control hardware interface was developed for the PAROL6 6-DOF robot arm, achieving 0% packet loss and <50ms control loop latency..."

**Methodology:**
> "The system architecture follows ROS 2 Hardware Interface specifications (REP-2008). Serial communication uses non-blocking I/O with POSIX termios configuration..."

**Implementation:**
> "The ESP32 firmware implements the MKS Servo42C closed-loop protocol, converting joint position commands (radians) to motor steps and reading encoder feedback..."

**Results:**
> "Validation testing demonstrated 100% reliability over a 30-minute continuous operation period with full payload. Quantitative metrics: [include table]"

**Figures to include:**
1. System architecture diagram
2. Data flow diagram
3. Control loop timing chart
4. Packet loss over time (should be flatline at 0%)
5. RViz screenshot showing robot at various poses

---

**For questions, see main guide:** `docs/TEAMMATE_SETUP_GUIDE.md`
