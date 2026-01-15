// Copyright (c) 2026 PAROL6 Team
// 
// ros2_control Hardware Interface for PAROL6 6-DOF Robot
// Day 1: SIL (Software-in-the-Loop) - Minimal stub implementation

#ifndef PAROL6_HARDWARE__PAROL6_SYSTEM_HPP_
#define PAROL6_HARDWARE__PAROL6_SYSTEM_HPP_

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include <rclcpp_lifecycle/state.hpp>
#include <libserial/SerialPort.h>

namespace parol6_hardware
{

class PAROL6System : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(PAROL6System)

  // Lifecycle interface methods
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  // Export interface descriptions
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  // Communication with hardware
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // Joint names (from URDF)
  std::vector<std::string> joint_names_;

  // State interfaces (positions, velocities)
  std::vector<double> hw_state_positions_;
  std::vector<double> hw_state_velocities_;

  // Command interfaces (positions)
  std::vector<double> hw_command_positions_;

  // Logger (namespace-qualified for better filtering)
  rclcpp::Logger logger_{rclcpp::get_logger("parol6_hardware.system")};

  // Serial Communication
  LibSerial::SerialPort serial_;
  std::string serial_port_;
  int baud_rate_;
  uint32_t seq_counter_ = 0;  // TX sequence counter
  
  // Feedback tracking (RX)
  uint32_t last_received_seq_ = 0;
  bool first_feedback_received_ = false;
  
  // Performance metrics (for thesis validation)
  uint64_t packets_received_ = 0;
  uint64_t packets_lost_ = 0;
  uint64_t parse_errors_ = 0;
  
  // Latency tracking (thesis evidence)
  rclcpp::Time last_rx_time_;
  double max_rx_period_ms_ = 0.0;
  
  // Clock for throttling logs
  rclcpp::Clock clock_;
};

}  // namespace parol6_hardware

#endif  // PAROL6_HARDWARE__PAROL6_SYSTEM_HPP_
