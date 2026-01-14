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
  RCLCPP_INFO(logger_, "🔧 on_configure() - Day 1: SIL (no hardware)");
  
  // Day 2 TODO: Open serial port here
  // Example:
  // try {
  //   serial_ = std::make_unique<serial::Serial>(
  //     serial_port_, baud_rate_,
  //     serial::Timeout::simpleTimeout(5));  // 5ms timeout
  //   
  //   if (!serial_->isOpen()) {
  //     RCLCPP_ERROR(logger_, "❌ Failed to open serial port");
  //     return CallbackReturn::ERROR;
  //   }
  //   
  //   RCLCPP_INFO(logger_, "✓ Serial port opened: %s", serial_port_.c_str());
  // } catch (std::exception& e) {
  //   RCLCPP_ERROR(logger_, "❌ Serial error: %s", e.what());
  //   return CallbackReturn::ERROR;
  // }

  RCLCPP_INFO(logger_, "✅ on_configure() complete");
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
  // Day 1: No-op (states stay at zero)
  // This is OK for SIL validation - just proving the plumbing works
  
  // Day 3 TODO: Read from ESP32
  // if (serial_->available()) {
  //   std::string response = serial_->readline();
  //   parse_feedback(response);  // Update hw_state_positions_, hw_state_velocities_
  // }
  
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
  // Day 1: No-op (just log occasionally for debugging)
  static size_t write_count = 0;
  if (++write_count % 250 == 0)  // Log every 10 seconds @ 25Hz
  {
    RCLCPP_DEBUG(logger_, "write() called %zu times (Day 1: no hardware)", write_count);
  }
  
  // Day 2 TODO: Send command to ESP32
  // Format command with %.2f precision (from implementation plan):
  // char buffer[256];
  // snprintf(buffer, sizeof(buffer),
  //          "<%" PRIu32 ",%.2f,%.2f,%.2f,%.2f,%.2f,%.2f>\n",
  //          seq_++,
  //          hw_command_positions_[0],
  //          hw_command_positions_[1],
  //          hw_command_positions_[2],
  //          hw_command_positions_[3],
  //          hw_command_positions_[4],
  //          hw_command_positions_[5]);
  //
  // try {
  //   serial_->write(buffer);
  // } catch (serial::SerialException& e) {
  //   RCLCPP_WARN_THROTTLE(logger_, *clock_, 1000, "Serial write timeout");
  //   return return_type::ERROR;
  // }
  
  return return_type::OK;
}

}  // namespace parol6_hardware

// Export plugin
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  parol6_hardware::PAROL6System, hardware_interface::SystemInterface)
