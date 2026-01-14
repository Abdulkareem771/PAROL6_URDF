---
name: 🤖 Day 4 - First Motion
about: Safe hardware integration and first trajectory execution
title: '[DAY-4] First Motion with Hardware'
labels: 'phase-4, hardware, motors, safety-critical'
assignees: ''
---

## 🎯 Objective
Safely activate motors and execute first trajectory with hardware.

## ⚠️ SAFETY REQUIREMENTS
- [ ] Current limits set to MINIMUM
- [ ] Velocity limits reduced (50% max)
- [ ] Emergency stop accessible
- [ ] Physical power cutoff available
- [ ] No-load test first (motors unconnected from robot)
- [ ] Team member present for first motion

## ✅ Tasks
- [ ] Connect motors with low current limit
- [ ] Test single joint first
- [ ] Execute simple trajectory (0.1 rad movement)
- [ ] Validate smooth motion (no jerking)
- [ ] Test all 6 joints individually
- [ ] Test coordinated motion
- [ ] Document motor parameters

## 📊 Success Criteria
- [ ] Motors respond to commands
- [ ] Smooth motion without vibration
- [ ] Position feedback matches commands
- [ ] No overheating
- [ ] Emergency stop functional

## 🔧 Configuration
- Update motor current limits in ESP32 firmware
- Set conservative velocity limits in controller config

## 📚 Documentation
- Create Day 4 validation report with video
- Document motor parameters and limits
- Safety incident log (if any)

## ⚠️ Blockers
- Requires Day 3 feedback loop complete
- ESP32 motor control firmware ready
- Hardware setup complete
