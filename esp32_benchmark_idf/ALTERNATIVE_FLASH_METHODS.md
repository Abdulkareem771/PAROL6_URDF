# ESP32 Firmware - Alternative: Flash with esptool.py

## Problem
ESP-IDF is not installed in your Docker container, so we can't build from source.

## Solution 1: Use Python esptool.py (Recommended)

Install esptool on your **host machine** (not Docker):

```bash
# Install esptool
pip3 install esptool

# Flash the pre-compiled Arduino .ino file instead
# (We'll use the Arduino CLI or PlatformIO)
```

---

## Solution 2: Install ESP-IDF in Docker (One-time setup)

Since you already have the Docker image, let's add ESP-IDF to it.

### Step 1: Enter your running container
```bash
docker start parol6_dev
docker exec -it parol6_dev bash
```

### Step 2: Install ESP-IDF
```bash
# Inside container
apt-get update
apt-get install -y git wget flex bison gperf python3 python3-pip python3-venv \
    cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0

# Clone ESP-IDF
cd /opt
git clone --recursive --branch v5.1.2 https://github.com/espressif/esp-idf.git esp-idf

# Install tools
cd /opt/esp-idf
./install.sh esp32

# Add to bashrc
echo '. /opt/esp-idf/export.sh' >> ~/.bashrc
```

**Time**: ~15-20 minutes (downloads ~2GB)

### Step 3: Commit the container
```bash
# Exit container
exit

# Save changes
docker commit parol6_dev parol6-ultimate:latest
```

### Step 4: Now flash.sh will work!
```bash
cd esp32_benchmark_idf
./flash.sh /dev/ttyUSB0
```

---

## Solution 3: Use PlatformIO (Easiest for Arduino code)

Install PlatformIO on host:

```bash
pip3 install platformio

# Create PlatformIO project from Arduino code
cd /path/to/PAROL6_URDF
mkdir -p platformio_esp32
cd platformio_esp32

# Initialize
pio project init --board esp32dev

# Copy Arduino code
cp ../PAROL6/firmware/benchmark_firmware.ino src/main.cpp

# Build and upload
pio run --target upload
```

---

## My Recommendation

**Quickest**: Use Solution 3 (PlatformIO) - it handles everything automatically.

**Most Robust**: Use Solution 2 (Install ESP-IDF in Docker) - good for long-term.

**For now**: Let me convert the firmware to PlatformIO format for you!
