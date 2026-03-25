# Teensy 4.1 Setup & Installation Guide

This guide covers everything required to build and flash the Teensy 4.1 for the PAROL6 robot.

The project uses **PlatformIO** running inside the existing `parol6-ultimate` Docker environment. This guarantees exactly reproducible builds and eliminates driver conflicts.

**Do NOT use the Arduino IDE** for this repository.

## 1. Prerequisites (Hardware)

*   **Teensy 4.1 Development Board**
*   **Micro-USB Cable** (must support data transfer, not just charging)

## 2. Docker Setup (Automatic)

The required tools (`platformio` and `teensy_loader_cli`) are automatically installed in the `parol6-ultimate` Docker image via the repository `Dockerfile`.

To ensure you have the latest image:
```bash
# From the repository root
docker build -t parol6-ultimate .
```

## 3. Building the Firmware

To compile the firmware without flashing it to the board:

1. Connect to your running Docker container (or run `start_container.sh`).
2. Navigate to the firmware directory:
   ```bash
   cd /workspace/parol6_firmware
   ```
3. Run the PlatformIO build command:
   ```bash
   pio run
   ```
   *You should see a `[SUCCESS]` message at the end of the compilation.*

## 4. Flashing the Teensy 4.1

We provide a dedicated script `flash_teensy.sh` that exactly mirrors the ESP32 workflow.

1. **Plug in** the Teensy 4.1 via USB to the host computer.
2. Ensure the Docker container has access to the USB bus (the `--device=/dev/bus/usb` flag is used in our run scripts).
3. From the **host machine** (outside the docker container), navigate to the firmware directory:
   ```bash
   cd ~/PAROL6_URDF/parol6_firmware
   ```
4. Run the flash script:
   ```bash
   ./flash_teensy.sh
   ```

### ⚠️ Important Flashing Caveats

*   **First time flashing:** You may need to press the physical white **PROGRAM** button on the Teensy board while the script says `Flashing Teensy...` and is waiting for the device.
*   **Subsequent flashes:** `teensy_loader_cli` normally issues a software reboot to put the board in programming mode automatically.
*   **Permissions:** If the CLI complains about inability to open the USB device, ensure `udev` rules for PJRC/Teensy are installed on the host machine (`/etc/udev/rules.d/00-teensy.rules`).

## 5. Serial Monitor Testing

Once flashed, the Teensy will act as a USB CDC Serial device natively (typically `/dev/ttyACM0`).

To monitor output from the board:
```bash
# Inside the docker container
pio device monitor -b 115200
```
*(Use `Ctrl+C` to exit the monitor)*
