# Day 2: Serial TX Implementation Plan

**Goal:** Implement non-blocking serial transmission to ESP32

**Status:** Ready to start  
**Prerequisites:** Day 1 SIL validation complete ✅

---

## 🎯 Objectives

1. **Open serial port** in `on_configure()` lifecycle method
2. **Send position commands** via `write()` at 25Hz
3. **Validate non-blocking behavior** (jitter < 5ms)
4. **Test with ESP32** (no motors connected)

---

## 📋 Implementation Checklist

### Part 1: Serial Port Opening
- [ ] Add `#include <serial/serial.h>` to header
- [ ] Add `std::unique_ptr<serial::Serial> serial_` member
- [ ] Implement `on_configure()`:
  - [ ] Read serial port from hardware_info
  - [ ] Create Serial object
  - [ ] Try-catch for connection errors
  - [ ] Log connection status
- [ ] Test: Launch should not crash with serial port unplugged

### Part 2: Command Transmission
- [ ] Implement `write()` method:
  - [ ] Format command string: `<seq,pos1,pos2,pos3,pos4,pos5,pos6>\n`
  - [ ] Use `%.2f` precision for positions
  - [ ] Increment sequence number
  - [ ] Call `serial_->write(cmd_string)`
  - [ ] Add timing guard (warn if > 5ms)
- [ ] Test: ESP32 receives valid commands at 25Hz

### Part 3: Validation
- [ ] Monitor timing with `ros2 topic hz /joint_states`
- [ ] Verify jitter < 5ms
- [ ] Test with ESP32 echoing commands back
- [ ] 15-minute stability test

---

## 🔧 Code Changes Required

### File: `include/parol6_hardware/parol6_system.hpp`

Add to includes:
```cpp
#include <serial/serial.h>
```

Add to private members:
```cpp
// Serial communication (Day 2)
std::unique_ptr<serial::Serial> serial_;
uint32_t write_count_{0};
```

### File: `src/parol6_system.cpp`

Replace `on_configure()` stub:
```cpp
CallbackReturn PAROL6System::on_configure(const rclcpp_lifecycle::State & previous_state)
{
  RCLCPP_INFO(logger_, "🔧 on_configure() - Opening serial port");

  // Read parameters from URDF
  std::string port = info_.hardware_parameters["serial_port"];
  int baud = std::stoi(info_.hardware_parameters["baud_rate"]);
  
  RCLCPP_INFO(logger_, "  Port: %s, Baud: %d", port.c_str(), baud);

  try {
    serial_ = std::make_unique<serial::Serial>(
      port,
      baud,
      serial::Timeout::simpleTimeout(5)  // 5ms timeout (non-blocking)
    );
    
    if (serial_->isOpen()) {
      RCLCPP_INFO(logger_, "✅ Serial port opened successfully");
    } else {
      RCLCPP_ERROR(logger_, "❌ Failed to open serial port");
      return CallbackReturn::ERROR;
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(logger_, "❌ Serial exception: %s", e.what());
    return CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}
```

Replace `write()` stub:
```cpp
return_type PAROL6System::write(const rclcpp::Time & time, const rclcpp::Duration & period)
{
  // Day 2: Send commands via serial
  if (!serial_ || !serial_->isOpen()) {
    return return_type::ERROR;
  }

  // Format: <seq,pos1,pos2,pos3,pos4,pos5,pos6>\n
  char cmd_buffer[128];
  snprintf(cmd_buffer, sizeof(cmd_buffer),
    "<%u,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f>\n",
    write_count_++,
    hw_command_positions_[0],
    hw_command_positions_[1],
    hw_command_positions_[2],
    hw_command_positions_[3],
    hw_command_positions_[4],
    hw_command_positions_[5]
  );

  auto start = std::chrono::high_resolution_clock::now();
  
  serial_->write(cmd_buffer);
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  if (duration.count() > 5000) {  // Warn if > 5ms
    RCLCPP_WARN(logger_, "write() took %ld μs (> 5ms target)", duration.count());
  }

  if (write_count_ % 100 == 0) {
    RCLCPP_DEBUG(logger_, "Sent %u commands", write_count_);
  }

  return return_type::OK;
}
```

---

## 🧪 Testing Procedure

### Test 1: Serial Port Opening
```bash
# Without ESP32 connected - should fail gracefully
ros2 launch parol6_hardware real_robot.launch.py

# Expected:
# [parol6_hardware.system]: ❌ Failed to open serial port
# Launch should exit cleanly
```

### Test 2: With ESP32 (Echo Mode)
**ESP32 firmware:** Simple echo (receive, print, send back)

```bash
# Connect ESP32 to /dev/ttyUSB0
ros2 launch parol6_hardware real_robot.launch.py

# In ESP32 serial monitor (115200 baud):
# Should see:
# <0,0.00,0.00,0.00,0.00,0.00,0.00>
# <1,0.00,0.00,0.00,0.00,0.00,0.00>
# <2,0.00,0.00,0.00,0.00,0.00,0.00>
# ...
```

### Test 3: Timing Validation
```bash
# Monitor topic rate
ros2 topic hz /joint_states

# Expected:
# average rate: 25.000 Hz
# std dev: < 0.001s (< 1ms jitter)
```

### Test 4: Send Trajectory
```bash
ros2 action send_goal /parol6_arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "{
    trajectory: {
      joint_names: [joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6],
      points: [
        {positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 0}},
        {positions: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 2}}
      ]
    }
  }"

# ESP32 should show changing position values:
# <150,0.00,0.00,0.00,0.00,0.00,0.00>
# <151,0.01,0.00,0.00,0.00,0.00,0.00>
# <152,0.02,0.00,0.00,0.00,0.00,0.00>
# ...
```

---

## ✅ Success Criteria

- [ ] Serial port opens without crashing controller_manager
- [ ] Commands sent at exactly 25Hz
- [ ] Jitter remains < 5ms
- [ ] ESP32 receives valid formatted data
- [ ] 15-minute stable operation without errors
- [ ] Clean shutdown closes serial port

---

## 🔐 Data Integrity and Ordering Strategy

To prevent loss-of-order or stale command execution:

### Sequence Number

Each command frame includes a monotonically increasing sequence counter:
```
<seq, pos1, pos2, pos3, pos4, pos5, pos6>
```

**ESP32 validates:**
- Sequence monotonicity (no backwards jumps)
- Missing packets (gap detection)
- Repeated frames (duplicate suppression)

### Lost Packet Handling

If ESP32 detects a gap in sequence numbers:

1.  **Hold last valid command** - Do NOT extrapolate
2.  **Do NOT jump to new position** - Prevents uncontrolled motion
3.  **Report fault status** (Day 3+ via feedback)

This prevents the robot from making unsafe jumps.

### Host-Side Recovery

If acknowledgements stall for > N cycles (Day 3+):

1.  Trigger warning log
2.  Optionally transition controller to `inactive`
3.  Require manual restart

### Packet Format Guarantees

- **Fixed width:** `<seq,p1,p2,p3,p4,p5,p6>\n` (consistent parsing)
- **Terminator:** `\n` ensures frame boundaries
- **Precision:** `%.2f` balances resolution vs bandwidth
- **No checksums (Day 2):** Serial UART has hardware CRC

**Day 3+:** Add CRC16 for end-to-end integrity validation.

---

## 🐛 Common Issues

### Issue: "Permission denied: /dev/ttyUSB0"
**Fix:** Add user to dialout group (requires container rebuild or chmod)
```bash
sudo chmod 666 /dev/ttyUSB0
```

### Issue: "Device not found"
**Check:**
```bash
ls -l /dev/ttyUSB* /dev/ttyACM*
# Verify device exists and is accessible from container
```

### Issue: Jitter increases
**Causes:**
- Blocking serial write
- Buffer overflow
- CPU throttling

**Debug:**
```bash
# Add timing logs to write()
RCLCPP_INFO_THROTTLE(logger_, *get_clock(), 1000, "write() took %ld μs", duration.count());
```

---

## 📊 Expected Timeline

- **Implementation:** 1-2 hours
- **Testing:** 1 hour
- **Validation:** 15 minutes (stability test)
- **Total:** ~3 hours

---

## 🚀 Next: Day 3

Once Day 2 is complete:
- Implement `read()` to parse ESP32 feedback
- Close the control loop
- Add sequence number validation

---

**Status:** Ready to implement  
**Confidence:** High (clear plan, Day 1 validated)
