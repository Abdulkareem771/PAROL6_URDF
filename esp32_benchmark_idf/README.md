# ESP32 Firmware - Documentation Index

**Welcome! This is your starting point for ESP32 firmware development and testing.**

---

## 🎯 Quick Start - Which Guide Do I Need?

Choose based on what you want to do:

| I want to... | Guide to use | Time needed |
|--------------|--------------|-------------|
| **Flash ESP32 and test** | [QUICK_START.md](QUICK_START.md) | 10 mins |
| **Test full ROS pipeline** | [TESTING_WITH_ROS.md](TESTING_WITH_ROS.md) | 20 mins |
| **Understand the firmware code** | [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | 30 mins |
| **Understand ROS system** | [ROS_SYSTEM_ARCHITECTURE.md](ROS_SYSTEM_ARCHITECTURE.md) | 30 mins |
| **Add motor control** | [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#adapting-for-motor-control) | Varies |

---

## 📚 Documentation Structure

```
esp32_benchmark_idf/
├── README.md (you are here)          ← Navigation guide
├── QUICK_START.md                    ← Fast testing (standalone)
├── TESTING_WITH_ROS.md               ← Full ROS pipeline testing
├── DEVELOPER_GUIDE.md                ← Code walkthrough + motor integration
├── ROS_SYSTEM_ARCHITECTURE.md        ← How RViz → ESP32 works
└── ALTERNATIVE_FLASH_METHODS.md      ← Other flashing options
```

---

## 🚀 Complete Workflow (Recommended Path)

### For New Team Members:

**1. Environment Setup** (One-time, 5 minutes)
   - Start container: `./start_container.sh`
   - Fix Python: `./fix_python_env.sh`
   - See: [Main GET_STARTED.md](../GET_STARTED.md)

**2. Flash & Test ESP32** (10 minutes)
   - Quick test without ROS
   - Verify communication works
   - See: **[QUICK_START.md](QUICK_START.md)** ✅ **Start here!**

**3. Test ROS Pipeline** (20 minutes)
   - Full RViz → MoveIt → Driver → ESP32
   - Measure latency, analyze logs
   - See: **[TESTING_WITH_ROS.md](TESTING_WITH_ROS.md)**

**4. Understand the Code** (30 minutes)
   - How firmware works
   - Prepare for motor integration
   - See: **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**

**5. Motor Integration** (Your thesis work!)
   - Replace benchmark code with motor control
   - See: [DEVELOPER_GUIDE.md - Motor Integration](DEVELOPER_GUIDE.md#adapting-for-motor-control)

---

## 📖 Guide Descriptions

### [QUICK_START.md](QUICK_START.md)
**Best for:** First-time users, quick verification  
**You'll learn:** How to flash ESP32 and test communication  
**Prerequisites:** ESP32 plugged in, Docker container running  
**Output:** Verified 0% packet loss, ~30ms latency

**Covers:**
- One-command flashing
- Standalone communication test
- Success criteria

---

### [TESTING_WITH_ROS.md](TESTING_WITH_ROS.md)
**Best for:** Testing full robot pipeline  
**You'll learn:** RViz → ESP32 data flow, message formats  
**Prerequisites:** QUICK_START.md completed  
**Output:** Working robot control, CSV logs with trajectory data

**Covers:**
- Full ROS integration
- Message format explained (positions only, not vel/acc)
- Log analysis
- Latency measurement
- What ESP32 receives vs what's logged

---

### [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
**Best for:** Understanding/modifying firmware  
**You'll learn:** ESP-IDF project structure, code walkthrough  
**Prerequisites:** Basic C programming, some embedded experience  
**Output:** Understanding to add motor control

**Covers:**
- ESP-IDF concepts (FreeRTOS, UART, logging)
- Line-by-line code explanation
- Motor integration examples
- Debugging tips
- Best practices

---

### [ROS_SYSTEM_ARCHITECTURE.md](ROS_SYSTEM_ARCHITECTURE.md)
**Best for:** Understanding ROS pipeline, adding custom planners  
**You'll learn:** How ROS components connect, data flow  
**Prerequisites:** Basic ROS 2 knowledge  
**Output:** Ability to modify/extend ROS system

**Covers:**
- RViz, MoveIt, Driver explained
- ROS 2 concepts (topics, actions, parameters)
- Step-by-step data flow
- How to interact with system
- Adding custom functionality

---

### [ALTERNATIVE_FLASH_METHODS.md](ALTERNATIVE_FLASH_METHODS.md)
**Best for:** Advanced users, alternatives to Docker  
**You'll learn:** Different flashing approaches  
**Prerequisites:** Some embedded development experience

**Covers:**
- Using esptool.py directly
- Installing ESP-IDF locally
- PlatformIO option

---

## 🎓 Learning Paths

### Path 1: "I just need to test communication"
1. [QUICK_START.md](QUICK_START.md) ✓ Done!

### Path 2: "I'm integrating with ROS"
1. [QUICK_START.md](QUICK_START.md)
2. [TESTING_WITH_ROS.md](TESTING_WITH_ROS.md)
3. [ROS_SYSTEM_ARCHITECTURE.md](ROS_SYSTEM_ARCHITECTURE.md) (for custom development)

### Path 3: "I'm implementing motor control"
1. [QUICK_START.md](QUICK_START.md)
2. [TESTING_WITH_ROS.md](TESTING_WITH_ROS.md)
3. [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) ← Focus here
4. [ROS_SYSTEM_ARCHITECTURE.md](ROS_SYSTEM_ARCHITECTURE.md) (for higher-level control)

### Path 4: "I'm adding custom ROS planning"
1. [TESTING_WITH_ROS.md](TESTING_WITH_ROS.md)
2. [ROS_SYSTEM_ARCHITECTURE.md](ROS_SYSTEM_ARCHITECTURE.md) ← Focus here
3. [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) (understand what ESP32 expects)

---

## 🛠️ Common Tasks - Quick Reference

| Task | Command | Guide |
|------|---------|-------|
| Flash ESP32 | `./flash.sh /dev/ttyUSB0` | [QUICK_START.md](QUICK_START.md) |
| Test communication | `python3 scripts/test_driver_communication.py` | [QUICK_START.md](QUICK_START.md) |
| Launch ROS pipeline | `./start_real_robot.sh` | [TESTING_WITH_ROS.md](TESTING_WITH_ROS.md) |
| Analyze logs | `python3 scripts/quick_log_analysis.py` | [TESTING_WITH_ROS.md](TESTING_WITH_ROS.md) |
| Monitor ESP32 | `idf.py -p /dev/ttyUSB0 monitor` | Any guide |
| Build firmware | `idf.py build` | [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) |

---

## 📁 Project Structure

```
esp32_benchmark_idf/
├── README.md                      ← You are here (navigation)
├── QUICK_START.md                 ← Start here for first test
├── TESTING_WITH_ROS.md            ← Full ROS pipeline
├── DEVELOPER_GUIDE.md             ← Code explanation
├── ROS_SYSTEM_ARCHITECTURE.md     ← ROS system details
├── ALTERNATIVE_FLASH_METHODS.md   ← Other options
│
├── main/
│   ├── benchmark_main.c           ← Firmware source code
│   └── CMakeLists.txt
├── CMakeLists.txt
├── flash.sh                       ← Quick flash script
└── COLCON_IGNORE                  ← Keeps ROS from building this
```

---

## ❓ FAQ

**Q: Which guide should I read first?**  
A: [QUICK_START.md](QUICK_START.md) - it's the fastest way to verify everything works.

**Q: I want to understand the code, where do I start?**  
A: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - complete code walkthrough with explanations.

**Q: How do I test with ROS?**  
A: [TESTING_WITH_ROS.md](TESTING_WITH_ROS.md) - full pipeline from RViz to ESP32.

**Q: I want to add custom motion planning, which guide?**  
A: [ROS_SYSTEM_ARCHITECTURE.md](ROS_SYSTEM_ARCHITECTURE.md) - shows how to interact with the system.

**Q: Where are the message formats explained?**  
A: [TESTING_WITH_ROS.md](TESTING_WITH_ROS.md#important-what-esp32-actually-receives) - detailed breakdown.

**Q: How do I add motor control?**  
A: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#adapting-for-motor-control) - step-by-step examples.

---

## ✅ Success Checklist

After following the guides, you should have:

- [x] Flashed ESP32 successfully
- [x] Tested standalone communication (0% loss)
- [x] Tested ROS pipeline (RViz → ESP32)
- [x] Understood firmware code structure
- [x] Know how to add motor control
- [x] Can analyze trajectory logs

**If stuck:** Check the Troubleshooting section in each guide!

---

## 🔗 Related Documentation

**In parent directory:**
- [../GET_STARTED.md](../GET_STARTED.md) - New team member onboarding
- [../docs/RVIZ_SETUP_GUIDE.md](../docs/RVIZ_SETUP_GUIDE.md) - RViz troubleshooting

**Analysis tools:**
- [../scripts/quick_log_analysis.py](../scripts/quick_log_analysis.py) - Interactive log analysis
- [../scripts/test_driver_communication.py](../scripts/test_driver_communication.py) - Communication test

---

**Last Updated:** January 2026  
**Team:** PAROL6 Robotics

**Questions?** Check the specific guide or ask the team!
