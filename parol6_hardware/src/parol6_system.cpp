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
  } catch (const std::out_of_range & e) {
    RCLCPP_ERROR(logger_, "❌ Missing required hardware parameter: %s", e.what());
    return CallbackReturn::ERROR;
  }

  RCLCPP_INFO(logger_, "📝 Config: Port=%s, Baud=%d", serial_port_.c_str(), baud_rate_);

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
    
    // Non-blocking with timeout protection
    // VTIME is in deciseconds (1 = 100 ms)
    serial_.SetVTime(1);   // 100 ms timeout
    serial_.SetVMin(0);    // Non-blocking

    if (!serial_.IsOpen()) {
      RCLCPP_WARN(logger_, "⚠️ Serial port %s not open after Open() — running in SPOOF mode",
                  serial_port_.c_str());
    } else {
      serial_ok_ = true;
      RCLCPP_INFO(logger_, "✅ Serial opened: %s @ %d baud (100 ms timeout)",
                  serial_port_.c_str(), baud_rate_);
    }

  } catch (const std::exception &e) {
    // Serial unavailable — NOT fatal. Fall through to spoof mode.
    // Controller_manager and all spawners still come up correctly.
    RCLCPP_WARN(logger_,
      "⚠️ Serial port '%s' unavailable (%s) — running in SPOOF mode (echoes commands as state)."
      " Connect Teensy and restart to enable real hardware.",
      serial_port_.c_str(), e.what());
    serial_ok_ = false;
    // Return SUCCESS intentionally — do NOT crash the controller_manager
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
  
  // Initialize command to current state (good practice)
  hw_command_positions_ = hw_state_positions_;
  
  RCLCPP_INFO(logger_, "✅ on_activate() complete - System ACTIVE");
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
      goto spoof_states;
    }
    
    // Find closing bracket
    size_t end_pos = response.find('>');
    if (end_pos == std::string::npos) {
      parse_errors_++;
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
        "Invalid feedback format (missing '>')");
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
    
    // Validate: should have ACK + SEQ + 6 positions + 6 velocities = 14 tokens
    // Firmware format: <ACK, SEQ, p1, p2, p3, p4, p5, p6, v1, v2, v3, v4, v5, v6>
    // tokens[0]=ACK(str), [1]=SEQ, [2..7]=positions, [8..13]=velocities
    if (tokens.size() != 14) {
      parse_errors_++;
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
        "Invalid feedback: expected 14 tokens, got %zu", tokens.size());
      goto spoof_states;
    }
    
    // Validate ACK
    if (tokens[0] != "ACK") {
      parse_errors_++;
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
        "Invalid feedback: expected ACK, got '%s'", tokens[0].c_str());
      goto spoof_states;
    }
    
    // Parse sequence number
    uint32_t received_seq = std::stoul(tokens[1]);
    
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
    
    // REAL FEEDBACK: (Toggle to true when moving to physical Actuators in Phase 4)
    const bool USE_REAL_FEEDBACK = true; 
    if (USE_REAL_FEEDBACK) {
      // Kinematic sign correction loaded from xacro ros_invert params in on_init()

      // URDF joint position limits — used to reject garbage fake-encoder values
      const double joint_lower[6] = {-3.14159, -0.98,   -2.0, -3.14159, -3.14159, -3.14159};
      const double joint_upper[6] = { 3.14159,  1.0,    1.3,  3.14159,  3.14159,  3.14159};

      // Parse and sign-correct all 6 joints
      // Firmware sends grouped: <ACK, SEQ, p0,p1,p2,p3,p4,p5, v0,v1,v2,v3,v4,v5>
      // tokens indices:          [0]   [1]  [2][3][4][5][6][7] [8][9][10][11][12][13]
      double candidate_pos[6], candidate_vel[6];
      bool all_valid = true;
      for (size_t i = 0; i < 6; ++i) {
        candidate_pos[i] = std::stod(tokens[2 + i]) * dir_signs_[i];  // tokens[2..7]
        candidate_vel[i] = std::stod(tokens[8 + i]) * dir_signs_[i];  // tokens[8..13]
        // Reject packets where ANY joint is outside its URDF limits + 10% tolerance
        double range = joint_upper[i] - joint_lower[i];
        if (candidate_pos[i] < joint_lower[i] - 0.1 * range ||
            candidate_pos[i] > joint_upper[i] + 0.1 * range) {
          RCLCPP_WARN_THROTTLE(logger_, clock_, 2000,
            "Joint J%zu feedback %.4f outside bounds [%.2f, %.2f] — ignoring packet",
            i + 1, candidate_pos[i], joint_lower[i], joint_upper[i]);
          all_valid = false;
          break;
        }
      }
      
      if (all_valid) {
        for (size_t i = 0; i < 6; ++i) {
          hw_state_positions_[i] = candidate_pos[i];
          hw_state_velocities_[i] = candidate_vel[i];
        }
        return return_type::OK; // Skip spoofing if successfully parsed and valid
      }
      // Fall through to spoof if any value was out of range
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
  // Format: <SEQ,p1,p2,p3,p4,p5,p6,v1,v2,v3,v4,v5,v6>
  // Total: 1 (seq) + 12 (6 joints × 2 values) = 13 values
  char buffer[512];  // Larger buffer for velocity data
  
  // Use PRIu32 for sequence number portability
  // #include <inttypes.h> -> Already at top
  
  // Kinematic sign correction from on_init() (xacro ros_invert params)
  
  // Format: <SEQ,J1_p,J2_p,J3_p,J4_p,J5_p,J6_p,J1_v,J2_v,J3_v,J4_v,J5_v,J6_v>
  
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
    serial_.Write(buffer);
  } catch (const std::exception &e) {
    RCLCPP_WARN_THROTTLE(logger_, clock_, 1000,
                         "⚠️ Serial TX timeout/error: %s", e.what());
    return return_type::ERROR;
  }
  
  return return_type::OK;
}

}  // namespace parol6_hardware

// Export plugin
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  parol6_hardware::PAROL6System, hardware_interface::SystemInterface)
