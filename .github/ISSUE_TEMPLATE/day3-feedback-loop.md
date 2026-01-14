---
name: 🔄 Day 3 - Feedback Loop
about: Implement ESP32 feedback parsing and closed-loop control
title: '[DAY-3] Feedback Loop Implementation'
labels: 'phase-3, serial, feedback'
assignees: ''
---

## 🎯 Objective
Close the control loop by reading ESP32 feedback and validating data integrity.

## ✅ Tasks
- [ ] Implement `read()` to parse ESP32 responses
- [ ] Add sequence number validation
- [ ] Detect and handle packet loss
- [ ] Update state interfaces with feedback
- [ ] Test with ESP32 sending mock positions
- [ ] 15-minute 0% packet loss validation

## 📊 Success Criteria
- [ ] Feedback parsed correctly
- [ ] Sequence numbers validated
- [ ] 0% packet loss over 15 minutes
- [ ] State interfaces reflect ESP32 data
- [ ] Clean error handling for corrupt packets

## 🔧 Implementation Files
- `src/parol6_system.cpp` - Implement read()
- Add packet validation logic

## 📚 Documentation
- Update Hardware Interface Guide with read() details
- Create Day 3 validation report

## ⚠️ Blockers
- Requires Day 2 Serial TX complete
- ESP32 firmware must send feedback
