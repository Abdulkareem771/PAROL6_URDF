// Copyright (c) 2026 PAROL6 Team
//
// ros2_control Hardware Interface for PAROL6 6-DOF Robot
// Day 1: SIL (Software-in-the-Loop) Validation
//
// This is a MINIMAL stub implementation for validating ROS plumbing:
// - No serial communication
// - No ESP32 connection
// - Hardware states hardcoded to zero
// - Validates: plugin loading, lifecycle, controller activation
//
// Next steps (Day 2): Add serial communication

#include "parol6_hardware/parol6_system.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <cinttypes>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace parol6_hardware
{

using hardware_interface::CallbackReturn;
using hardware_interface::return_type;

// ============================================================================
// LIFECYCLE METHOD: on_init
// ============================================================================
// Called once when the hardware component is loaded
// Purpose: Read configuration from URDF, allocate resources
//
// For Day 1 (SIL):
// - Read joint names from URDF
// - Initialize state/command vectors
// - No hardware configuration yet
// ============================================================================

CallbackReturn PAROL6System::on_init(const hardware_interface::HardwareInfo & info)
{
  // Call base class init
  if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS)
  {
    return CallbackReturn::ERROR;
  }

  RCLCPP_INFO(logger_, "🚀 Day 1: SIL Validation - Initializing PAROL6 Hardware Interface");

  // Read parameters
  try {
    serial_port_ = info_.hardware_parameters.at("serial_port");
    baud_rate_ = std::stoi(info_.hardware_parameters.at("baud_rate"));
    auto spoof_it = info_.hardware_parameters.find("allow_spoofing");
    if (spoof_it != info_.hardware_parameters.end()) {
      allow_spoofing_ = (spoof_it->second == "true" || spoof_it->second == "1");
    }
  } catch (const std::out_of_range & e) {
    RCLCPP_ERROR(logger_, "❌ Missing required hardware parameter: %s", e.what());
    return CallbackReturn::ERROR;
  }

  RCLCPP_INFO(logger_, "📝 Config: Port=%s, Baud=%d, allow_spoofing=%s",
              serial_port_.c_str(), baud_rate_, allow_spoofing_ ? "true" : "false");

  // Read joint names from URDF
  joint_names_.clear();
  for (const auto & joint : info_.joints)
  {
    joint_names_.push_back(joint.name);
    RCLCPP_INFO(logger_, "  ✓ Joint: %s", joint.name.c_str());
  }

  const size_t num_joints = joint_names_.size();
  
  if (num_joints != 6)
  {
    RCLCPP_ERROR(logger_, "❌ Expected 6 joints, got %zu", num_joints);
    return CallbackReturn::ERROR;
  }

  // Load per-joint kinematic sign correction from xacro <param name="ros_invert"> tags.
  // Expected values: "true" = invert (-1.0), "false" or absent = normal (+1.0).
  // Fallback: J1, J3, J6 default to -1.0 (from STM32 legacy motor_init.cpp).
  //
  // To change these: edit parol6.ros2_control.xacro OR use the GUI Joints tab "ROS Inv" checkbox,
  // then update the xacro to match.
  const bool legacy_invert_fallback[6] = {true, false, true, false, false, true};
  dir_signs_.resize(num_joints);
  for (size_t i = 0; i < num_joints; ++i) {
    auto it = info_.joints[i].parameters.find("ros_invert");
    bool should_invert;
    if (it != info_.joints[i].parameters.end()) {
      should_invert = (it->second == "true" || it->second == "1");
    } else {
      should_invert = legacy_invert_fallback[i];
      RCLCPP_WARN(logger_,
        "Joint '%s' missing 'ros_invert' xacro param — falling back to legacy default (%s)",
        info_.joints[i].name.c_str(), should_invert ? "true" : "false");
    }
    dir_signs_[i] = should_invert ? -1.0 : 1.0;
    RCLCPP_INFO(logger_, "  ✓ Joint %s: ros_invert=%s (sign=%.1f)",
      info_.joints[i].name.c_str(), should_invert ? "true" : "false", dir_signs_[i]);
  }

  hw_state_positions_.resize(num_joints, 0.0);
  hw_state_velocities_.resize(num_joints, 0.0);
  hw_command_positions_.resize(num_joints, 0.0);
  hw_command_velocities_.resize(num_joints, 0.0);  // NEW: velocity commands

  RCLCPP_INFO(logger_, "✅ on_init() complete - %zu joints configured", num_joints);
  
  return CallbackReturn::SUCCESS;
}

// ============================================================================
// LIFECYCLE METHOD: on_configure  
// ============================================================================
// Called when transitioning from UNCONFIGURED to INACTIVE
// Purpose: Setup hardware connection (serial port, etc.)
//
// For Day 1 (SIL):
// - Just log success - no hardware to configure
// - Day 2 will add: serial port opening
// ============================================================================

CallbackReturn PAROL6System::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(logger_, "🔧 on_configure() - Opening serial port %s...", serial_port_.c_str());
  serial_ok_ = false;  // Assume no hardware until proven otherwise

  try {
    serial_.Open(serial_port_);
    
    // Set Baud Rate
    using LibSerial::BaudRate;
    BaudRate baud;
    switch (baud_rate_) {
      case 9600: baud = BaudRate::BAUD_9600; break;
      case 19200: baud = BaudRate::BAUD_19200; break;
      case 38400: baud = BaudRate::BAUD_38400; break;
      case 57600: baud = BaudRate::BAUD_57600; break;
      case 115200: baud = BaudRate::BAUD_115200; break;
      default:
        RCLCPP_WARN(logger_, "⚠️ Unsupported baud rate %d, defaulting to 115200", baud_rate_);
        baud = LibSerial::BaudRate::BAUD_115200;
    }
    serial_.SetBaudRate(baud);
    serial_.SetCharacterSize(LibSerial::CharacterSize::CHAR_SIZE_8);
    serial_.SetParity(LibSerial::Parity::PARITY_NONE);
    serial_.SetStopBits(LibSerial::StopBits::STOP_BITS_1);
    serial_.SetFlowControl(LibSerial::FlowControl::FLOW_CONTROL_NONE);

    serial_.SetDTR(true);
    serial_.SetRTS(true);
    
    // Non-blocking with timeout protection
    // VTIME is in deciseconds (1 = 100 ms)
    serial_.SetVTime(1);   // 100 ms timeout
    serial_.SetVMin(0);    // Non-blocking

    // Flush any stale data
    serial_.FlushIOBuffers();

    if (!serial_.IsOpen()) {
      if (!allow_spoofing_) {
        RCLCPP_ERROR(logger_, "❌ Serial port %s not open after Open() and spoofing is disabled",
                     serial_port_.c_str());
        return CallbackReturn::ERROR;
      }
      RCLCPP_WARN(logger_, "⚠️ Serial port %s not open after Open() — running in explicit SPOOF mode",
                  serial_port_.c_str());
    } else {
      serial_ok_ = true;
      RCLCPP_INFO(logger_, "✅ Serial opened: %s @ %d baud (100 ms timeout)",
                  serial_port_.c_str(), baud_rate_);
    }

  } catch (const std::exception &e) {
    if (!allow_spoofing_) {
      RCLCPP_ERROR(logger_,
        "❌ Serial port '%s' unavailable (%s) and spoofing is disabled.",
        serial_port_.c_str(), e.what());
      return CallbackReturn::ERROR;
    }
    RCLCPP_WARN(logger_,
      "⚠️ Serial port '%s' unavailable (%s) — running in explicit SPOOF mode.",
      serial_port_.c_str(), e.what());
    serial_ok_ = false;
  }

  return CallbackReturn::SUCCESS;
}

// ============================================================================
// LIFECYCLE METHOD: on_activate
// ============================================================================
// Called when transitioning from INACTIVE to ACTIVE
// Purpose: Start communication, enable motors
//
// For Day 1 (SIL):
// - Just log success - controllers will start calling read()/write()
// - Day 4 will add: motor enabling logic
// ============================================================================

CallbackReturn PAROL6System::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(logger_, "⚡ on_activate() - Controllers will now call read()/write()");

  // Do NOT initialize commands here — wait until first real feedback arrives in read().
  // Initializing to zeros now would cause write() to command all joints to position 0,
  // arming the motors and driving them toward zero before we know where they actually are.
  // Commands will be properly seeded in read() when first_feedback_received_ is set.
  // Fill with NaN so any premature write() calls are detectable and safe.
  std::fill(hw_command_positions_.begin(), hw_command_positions_.end(), std::numeric_limits<double>::quiet_NaN());
  std::fill(hw_command_velocities_.begin(), hw_command_velocities_.end(), 0.0);

  RCLCPP_INFO(logger_, "✅ on_activate() complete - System ACTIVE (waiting for first encoder feedback before sending commands)");
  return CallbackReturn::SUCCESS;
}

// ============================================================================
// LIFECYCLE METHOD: on_deactivate
// ============================================================================
// Called when transitioning from ACTIVE to INACTIVE
// Purpose: Stop communication, disable motors safely
//
// For Day 1 (SIL):
// - Just log - no resources to clean up yet
// - Day 2+ will add: stop any threads, close serial port
// ============================================================================

CallbackReturn PAROL6System::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(logger_, "🛑 on_deactivate() - Stopping hardware interface");
  
  // Day 2+ TODO: Clean up resources
  // - Stop serial communication thread (if using threading)
  // - Don't close serial port (that's for on_cleanup)
  
  RCLCPP_INFO(logger_, "✅ on_deactivate() complete");
  return CallbackReturn::SUCCESS;
}

// ============================================================================
// EXPORT STATE INTERFACES
// ============================================================================
// Tell ros2_control what state data we provide
// For PAROL6: position and velocity for each joint
// ============================================================================

std::vector<hardware_interface::StateInterface> PAROL6System::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    // Position state
    state_interfaces.emplace_back(hardware_interface::StateInterface(
      joint_names_[i], hardware_interface::HW_IF_POSITION, &hw_state_positions_[i]));

    // Velocity state
    state_interfaces.emplace_back(hardware_interface::StateInterface(
      joint_names_[i], hardware_interface::HW_IF_VELOCITY, &hw_state_velocities_[i]));
  }

  RCLCPP_INFO(logger_, "📤 Exported %zu state interfaces", state_interfaces.size());
  return state_interfaces;
}

// ============================================================================
// EXPORT COMMAND INTERFACES
// ============================================================================
// Tell ros2_control what commands we accept
// For PAROL6: position commands for each joint
// (Velocity and acceleration added later if needed)
// ============================================================================

std::vector<hardware_interface::CommandInterface> PAROL6System::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    // Position command
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
      joint_names_[i], hardware_interface::HW_IF_POSITION, &hw_command_positions_[i]));
    
    // Velocity command (NEW)
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
      joint_names_[i], hardware_interface::HW_IF_VELOCITY, &hw_command_velocities_[i]));
  }

  RCLCPP_INFO(logger_, "📥 Exported %zu command interfaces", command_interfaces.size());
  return command_interfaces;
}

// ============================================================================
// READ FROM HARDWARE
// ============================================================================
// Called at controller update rate (25 Hz) to read joint states from hardware
//
// For Day 1 (SIL):
// - Just return OK - states remain at zero
// - Day 3 will add: read serial feedback from ESP32, parse positions/velocities
// ============================================================================

return_type PAROL6System::read(
  const rclcpp::Time & time, const rclcpp::Duration & /*period*/)
{
  // Vector safety check (prevent segfault on misconfiguration)
  if (hw_state_positions_.size() < 6) {
    RCLCPP_ERROR(logger_, "❌ State vector size invalid! Expected 6, got %zu", 
                 hw_state_positions_.size());
    return return_type::ERROR;
  }
  
  // If serial port never opened (no Teensy connected), run in pure spoof mode
  if (!serial_ok_) {
    if (!allow_spoofing_) {
      RCLCPP_ERROR_THROTTLE(logger_, clock_, 1000,
        "❌ read() called without an active serial connection and spoofing is disabled");
      return return_type::ERROR;
    }
    goto spoof_states;
  }

  // Check if data is available (non-blocking)
  if (!serial_.IsDataAvailable()) {
    // Check if we are starved (e.g. 5 seconds without data) -> assume full HIL spoof mode
    auto now = time;
    if (std::abs(now.seconds() - last_rx_time_.seconds()) > 5.0) {
        goto spoof_states;
    }
    return return_type::OK;  // No data yet, not an error
  }
  
  try {
    // Read line (blocking with 2ms timeout from serial configuration)
    std::string response;
    serial_.ReadLine(response, '\n');

    // DEBUG: Log raw feedback for validation (throttled manually to prevent crash)
    static int log_counter = 0;
    if (++log_counter % 50 == 0) { // Log every ~2 seconds (50 * 40ms)
      // Check for non-empty response to avoid printing empty lines
      if (!response.empty()) {
        RCLCPP_INFO(logger_, "📥 Raw feedback: %s", response.c_str());
      }
    }
    
    // Parse: <ACK,SEQ,J1,J2,J3,J4,J5,J6>
    if (response.empty() || response[0] != '<') {
      parse_errors_++;
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
        "Invalid feedback format (missing '<')");
      if (!allow_spoofing_) {
        return return_type::ERROR;
      }
      goto spoof_states;
    }
    
    // Find closing bracket
    size_t end_pos = response.find('>');
    if (end_pos == std::string::npos) {
      parse_errors_++;
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
        "Invalid feedback format (missing '>')");
      if (!allow_spoofing_) {
        return return_type::ERROR;
      }
      goto spoof_states;
    }
    
    // Extract content between < and >
    std::string content = response.substr(1, end_pos - 1);
    
    // Tokenize by comma
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t comma_pos;
    while ((comma_pos = content.find(',', start)) != std::string::npos) {
      tokens.push_back(content.substr(start, comma_pos - start));
      start = comma_pos + 1;
    }
    tokens.push_back(content.substr(start));  // Last token
    
    // Validate: ACK + SEQ + 6 positions + 6 velocities + lim_state + state_byte = 16 tokens
    // Firmware format: <ACK, SEQ, p1..p6, v1..v6, lim_state, state_byte>
    // tokens[0]=ACK, [1]=SEQ, [2..7]=positions, [8..13]=velocities, [14]=lim_state, [15]=state_byte
    // Backward-compat: also accept 14 tokens (firmware without limit switch support)
    if (tokens.size() < 14 || tokens.size() > 18) {
      parse_errors_++;
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
        "Invalid feedback: expected 14-18 tokens, got %zu", tokens.size());
      if (!allow_spoofing_) {
        return return_type::ERROR;
      }
      goto spoof_states;
    }
    
    // Validate ACK
    if (tokens[0] != "ACK") {
      parse_errors_++;
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
        "Invalid feedback: expected ACK, got '%s'", tokens[0].c_str());
      if (!allow_spoofing_) {
        return return_type::ERROR;
      }
      goto spoof_states;
    }
    
    // Parse sequence number
    uint32_t received_seq = std::stoul(tokens[1]);
    
    // Capture state BEFORE we update first_feedback_received_
    bool is_first_packet = !first_feedback_received_;
    
    // Wraparound-safe packet loss detection
    if (first_feedback_received_) {
      uint32_t expected_seq = last_received_seq_ + 1;
      
      // Handle uint32_t wraparound (UINT32_MAX -> 0)
      bool wraparound = (last_received_seq_ == UINT32_MAX && received_seq == 0);
      bool sequence_ok = (received_seq == expected_seq) || wraparound;
      
      if (!sequence_ok) {
        // Prevent underflow: only count forward jumps as loss
        uint32_t lost_count = 0;
        if (received_seq > expected_seq) {
          lost_count = received_seq - expected_seq;
        } else {
          lost_count = 1;  // Conservative: assume 1 packet lost on backward jump
        }
        
        packets_lost_ += lost_count;
        RCLCPP_WARN(logger_, 
          "⚠️ PACKET LOSS DETECTED! Expected seq %u, got %u (lost %u packets)",
          expected_seq, received_seq, lost_count);
      }
      
      // Track inter-packet timing (thesis latency evidence)
      auto now = time;
      double dt_ms = std::abs(now.seconds() - last_rx_time_.seconds()) * 1000.0;
      max_rx_period_ms_ = std::max(max_rx_period_ms_, dt_ms);
      last_rx_time_ = now;
      
    } else {
      first_feedback_received_ = true;
      last_rx_time_ = time;  // Initialize timing
      RCLCPP_INFO(logger_, "✅ First feedback received (seq %u)", received_seq);
    }
    

    last_received_seq_ = received_seq;
    packets_received_++;
    
    // REAL FEEDBACK: Dual-format parser
    const bool USE_REAL_FEEDBACK = true; 
    if (USE_REAL_FEEDBACK) {
      // URDF joint position limits — used to reject garbage fake-encoder values
      const double joint_lower[6] = {-3.14159, -0.98,   -2.0, -3.14159, -3.14159, -3.14159};
      const double joint_upper[6] = { 3.14159,  1.0,    1.3,  3.14159,  3.14159,  3.14159};

      double candidate_pos[6], candidate_vel[6];

      // Detect packet format by token count:
      //
      //   FORMAT A — realtime_servo_blackpill (interleaved):
      //     <ACK, SEQ, p0,v0, p1,v1, p2,v2, p3,v3, p4,v4, p5,v5>
      //     tokens: [0]=ACK [1]=SEQ [2]=p0 [3]=v0 [4]=p1 [5]=v1 ... [12]=p5 [13]=v5   → size == 14
      //
      //   FORMAT B — parol6_firmware / Teensy (flat, grouped):
      //     <ACK, SEQ, p0,p1,p2,p3,p4,p5, v0,v1,v2,v3,v4,v5 [,lim_state [,state_byte]]>
      //     tokens: [0]=ACK [1]=SEQ [2..7]=pos [8..13]=vel [14]=lim [15]=state  → size >= 14
      //
      // Heuristic to distinguish them when both have 14 tokens:
      //   In Format A (interleaved), tokens[3] is v0 (a velocity, typically small << positions).
      //   In Format B (flat), tokens[3] is p1 (a position, similar magnitude to tokens[2]).
      //   We detect Format A when tokens.size() == 14 (BlackPill never sends trailing lim/state).
      //   We assume Format B when tokens.size() > 14.

      const bool is_interleaved = (tokens.size() == 14);

      if (is_interleaved) {
        // FORMAT A — interleaved: p0,v0, p1,v1, ...
        // tokens[2 + i*2] = pos_i,  tokens[3 + i*2] = vel_i
        for (size_t i = 0; i < 6; ++i) {
          candidate_pos[i] = std::stod(tokens[2 + i * 2]) * dir_signs_[i];
          candidate_vel[i] = std::stod(tokens[3 + i * 2]) * dir_signs_[i];
        }
        RCLCPP_DEBUG(logger_, "📦 Parsing interleaved BlackPill ACK (14 tokens)");
      } else {
        // FORMAT B — flat: p0..p5 then v0..v5
        // tokens[2 + i] = pos_i,  tokens[8 + i] = vel_i
        for (size_t i = 0; i < 6; ++i) {
          candidate_pos[i] = std::stod(tokens[2 + i]) * dir_signs_[i];
          candidate_vel[i] = std::stod(tokens[8 + i]) * dir_signs_[i];
        }
        RCLCPP_DEBUG(logger_, "📦 Parsing flat Teensy ACK (%zu tokens)", tokens.size());
      }

      // Soft bounds: log a warning but ALWAYS apply real encoder feedback.
      // Hard rejection caused a runaway spiral: reject → spoof → echo commands → MoveIt
      // sends bigger positions → encoder accumulates → reject again → infinite loop.
      for (size_t i = 0; i < 6; ++i) {
        double range = joint_upper[i] - joint_lower[i];
        if (candidate_pos[i] < joint_lower[i] - 0.1 * range ||
            candidate_pos[i] > joint_upper[i] + 0.1 * range) {
          RCLCPP_WARN_THROTTLE(logger_, clock_, 2000,
            "⚠️ Joint J%zu position %.4f outside URDF limits [%.2f, %.2f] — clamping to limits",
            i + 1, candidate_pos[i], joint_lower[i], joint_upper[i]);
          candidate_pos[i] = std::clamp(candidate_pos[i], joint_lower[i], joint_upper[i]);
        }
      }

      for (size_t i = 0; i < 6; ++i) {
        hw_state_positions_[i] = candidate_pos[i];
        hw_state_velocities_[i] = candidate_vel[i];
      }

      // On the very first packet: seed command with real encoder positions so
      // write() sends current state (not zeros/NaN) when the firmware first arms.
      if (is_first_packet) {
        for (size_t i = 0; i < 6; ++i) {
          hw_command_positions_[i] = candidate_pos[i];
          hw_command_velocities_[i] = 0.0;
        }
        RCLCPP_INFO(logger_, "📌 Commands seeded from first encoder feedback: [ %.3f %.3f %.3f %.3f %.3f %.3f ] — motors safe",
                    candidate_pos[0], candidate_pos[1], candidate_pos[2], 
                    candidate_pos[3], candidate_pos[4], candidate_pos[5]);
      }

      return return_type::OK;
    }
    
    // Log statistics every 5 minutes (thesis validation data)
    RCLCPP_INFO_THROTTLE(logger_, clock_, 300000,
      "📊 Stats: RX=%lu LOST=%lu ERR=%lu Loss%%=%.4f MAX_DT=%.2f ms",
      packets_received_, packets_lost_, parse_errors_,
      packets_received_ > 0 ? (100.0 * packets_lost_ / packets_received_) : 0.0,
      max_rx_period_ms_);
      
  } catch (const std::exception& e) {
    parse_errors_++;
    RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
      "Error reading feedback: %s", e.what());
    if (!allow_spoofing_) {
      return return_type::ERROR;
    }
  }

spoof_states:
  // HIL SPOOFING: Echo outbound commands directly back as current state ALWAYS
  // This tells MoveIt the robot is perfectly following the trajectory,
  // preventing "Controller is taking too long... TIMED_OUT" errors.
  for (size_t i = 0; i < 6; ++i) {
    if (!std::isnan(hw_command_positions_[i])) {
      hw_state_positions_[i] = hw_command_positions_[i];
    }
    if (!std::isnan(hw_command_velocities_[i])) {
      hw_state_velocities_[i] = hw_command_velocities_[i];
    }
  }

  return return_type::OK;
}

// ============================================================================
// WRITE TO HARDWARE
// ============================================================================
// Called at controller update rate (25 Hz) to send commands to hardware
//
// For Day 1 (SIL):
// - Just return OK - no hardware to command
// - Day 2 will add: format and send serial command to ESP32
// ============================================================================

return_type PAROL6System::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // Safety guard: do not send any commands until we have real encoder feedback.
  // Without this guard, write() would blast position=NaN or position=0 to all
  // joints immediately on controller activation, arming the motors and driving
  // them toward zero before we know where they actually are.
  if (!first_feedback_received_) {
    RCLCPP_INFO_THROTTLE(logger_, clock_, 2000,
      "⏳ write() suppressed — waiting for first encoder feedback before commanding motors");
    return return_type::OK;
  }

  // Format: <SEQ,J1_p,J2_p,J3_p,J4_p,J5_p,J6_p,J1_v,J2_v,J3_v,J4_v,J5_v,J6_v>
  char buffer[512];
  
  // Safety: don't write if commands haven't been seeded yet (NaN from on_activate)
  for (size_t i = 0; i < 6; ++i) {
    if (std::isnan(hw_command_positions_[i])) {
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, "⚠️ write() skipped — command position NaN for joint %zu", i);
      return return_type::OK;
    }
  }

  int written = snprintf(buffer, sizeof(buffer),
           "<%" PRIu32 ",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f>\n",
           seq_counter_++,
           hw_command_positions_[0] * dir_signs_[0],
           hw_command_positions_[1] * dir_signs_[1],
           hw_command_positions_[2] * dir_signs_[2],
           hw_command_positions_[3] * dir_signs_[3],
           hw_command_positions_[4] * dir_signs_[4],
           hw_command_positions_[5] * dir_signs_[5],
           hw_command_velocities_[0] * dir_signs_[0],
           hw_command_velocities_[1] * dir_signs_[1],
           hw_command_velocities_[2] * dir_signs_[2],
           hw_command_velocities_[3] * dir_signs_[3],
           hw_command_velocities_[4] * dir_signs_[4],
           hw_command_velocities_[5] * dir_signs_[5]);

  if (written < 0 || written >= (int)sizeof(buffer)) {
      RCLCPP_ERROR(logger_, "Command buffer overflow! written=%d", written);
      return return_type::ERROR;
  }

  try {
    if (!serial_ok_) {
      if (allow_spoofing_) {
        return return_type::OK;
      }
      RCLCPP_ERROR_THROTTLE(logger_, clock_, 1000,
        "❌ write() called without an active serial connection");
      return return_type::ERROR;
    }
    serial_.Write(buffer);
  } catch (const std::exception &e) {
    RCLCPP_WARN_THROTTLE(logger_, clock_, 1000,
                         "⚠️ Serial TX timeout/error: %s", e.what());
    return return_type::ERROR;
  }
  
  return return_type::OK;
}

// ============================================================================
// COMMAND MODE SWITCHING
// ============================================================================
// Required by ROS2 Control Humble to allow multiple command interfaces
// (e.g. position and velocity) per joint.
// ============================================================================

hardware_interface::return_type PAROL6System::prepare_command_mode_switch(
  const std::vector<std::string>& /*start_interfaces*/,
  const std::vector<std::string>& /*stop_interfaces*/)
{
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type PAROL6System::perform_command_mode_switch(
  const std::vector<std::string>& /*start_interfaces*/,
  const std::vector<std::string>& /*stop_interfaces*/)
{
  return hardware_interface::return_type::OK;
}

}  // namespace parol6_hardware

// Export plugin
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  parol6_hardware::PAROL6System, hardware_interface::SystemInterface)
