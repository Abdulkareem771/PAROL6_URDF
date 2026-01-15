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

  // Allocate state and command storage
  hw_state_positions_.resize(num_joints, 0.0);
  hw_state_velocities_.resize(num_joints, 0.0);
  hw_command_positions_.resize(num_joints, 0.0);

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
  RCLCPP_INFO(logger_, "🔧 on_configure() - Opening serial port...");

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
        baud = BaudRate::BAUD_115200;
    }
    serial_.SetBaudRate(baud);

    if (!serial_.IsOpen()) {
      RCLCPP_ERROR(logger_, "❌ Failed to open serial port: %s", serial_port_.c_str());
      return CallbackReturn::ERROR;
    }

    RCLCPP_INFO(logger_, "✅ Serial opened successfully: %s @ %d", serial_port_.c_str(), baud_rate_);

  } catch (const std::exception &e) {
    RCLCPP_ERROR(logger_, "❌ Serial exception during configure: %s", e.what());
    return CallbackReturn::ERROR;
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
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // Check if data is available (non-blocking)
  if (!serial_.IsDataAvailable()) {
    return return_type::OK;  // No data yet, not an error
  }
  
  try {
    // Read line (blocking with timeout from serial port configuration)
    std::string response;
    serial_.ReadLine(response, '\n');
    
    // Parse: <ACK,SEQ,J1,J2,J3,J4,J5,J6>
    if (response.empty() || response[0] != '<') {
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
        "Invalid feedback format (missing '<')");
      return return_type::OK;
    }
    
    // Find closing bracket
    size_t end_pos = response.find('>');
    if (end_pos == std::string::npos) {
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
        "Invalid feedback format (missing '>')");
      return return_type::OK;
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
    
    // Validate: should have ACK + SEQ + 6 joints = 8 tokens
    if (tokens.size() != 8) {
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
        "Invalid feedback: expected 8 tokens, got %zu", tokens.size());
      return return_type::OK;
    }
    
    // Validate ACK
    if (tokens[0] != "ACK") {
      RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
        "Invalid feedback: expected ACK, got '%s'", tokens[0].c_str());
      return return_type::OK;
    }
    
    // Parse sequence number
    uint32_t received_seq = std::stoul(tokens[1]);
    
    // Check for packet loss (after first packet)
    if (first_feedback_received_) {
      uint32_t expected_seq = last_received_seq_ + 1;
      if (received_seq != expected_seq) {
        RCLCPP_WARN(logger_, 
          "⚠️ PACKET LOSS DETECTED! Expected seq %u, got %u (lost %u packets)",
          expected_seq, received_seq, received_seq - expected_seq);
      }
    } else {
      first_feedback_received_ = true;
      RCLCPP_INFO(logger_, "✅ First feedback received (seq %u)", received_seq);
    }
    
    last_received_seq_ = received_seq;
    
    // Parse joint positions (tokens 2-7)
    for (size_t i = 0; i < 6; ++i) {
      hw_state_positions_[i] = std::stod(tokens[i + 2]);
    }
    
    // Success!
    return return_type::OK;
    
  } catch (const std::exception& e) {
    RCLCPP_WARN_THROTTLE(logger_, clock_, 1000, 
      "Error reading feedback: %s", e.what());
    return return_type::OK;  // Don't fail the controller
  }
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
  // Format: <SEQ,J1,J2,J3,J4,J5,J6>
  // Precision: %.2f (sufficient for 0.05 rad resolution)
  char buffer[256];
  
  // Use PRIu32 for sequence number portability
  // #include <inttypes.h> -> Already at top
  
  int written = snprintf(buffer, sizeof(buffer),
           "<%" PRIu32 ",%.2f,%.2f,%.2f,%.2f,%.2f,%.2f>\n",
           seq_counter_++,
           hw_command_positions_[0],
           hw_command_positions_[1],
           hw_command_positions_[2],
           hw_command_positions_[3],
           hw_command_positions_[4],
           hw_command_positions_[5]);

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
