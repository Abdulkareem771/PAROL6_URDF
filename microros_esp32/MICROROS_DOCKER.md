# micro-ROS & ESP-IDF Docker Image

This repo now ships a fully reproducible Dockerfile that bakes in the ESP-IDF toolchain and micro-ROS Agent so you can share a single image without running post-install scripts.

## What the image contains
- ROS 2 Humble desktop + MoveIt + Gazebo Classic + Ignition packages (same as before)
- ESP-IDF v5.1 installed at `/opt/esp-idf`
- Python toolchain for ESP-IDF and micro-ROS builds
- Pre-built micro-ROS agent workspace at `/microros_ws` (ready to run `micro_ros_agent`)
- Quality-of-life shell setup (`/root/.bashrc` sources ROS, ESP-IDF, micro-ROS)

## Build the image
```bash
cd /home/kareem/Desktop/PAROL6_URDF
docker build -t parol6-ultimate:latest -f Dockerfile .
```

> The tag `parol6-ultimate:latest` keeps compatibility with `start_ignition.sh`.

## Verify inside a fresh container
```bash
docker run --rm -it parol6-ultimate:latest bash
source /opt/esp-idf/export.sh
idf.py --version                # confirms ESP-IDF
source /microros_ws/install/setup.bash
ros2 run micro_ros_agent micro_ros_agent --help   # confirms agent build
```

## Use with the existing workflow
- Start the dev container as usual (it will use this image): `./start_ignition.sh`
- Build firmware inside the running container:
  ```bash
  docker exec -it parol6_dev bash -lc "
    source /opt/esp-idf/export.sh &&
    cd /workspace/microros_esp32 &&
    idf.py set-target esp32 &&
    idf.py build
  "
  ```
- Run the micro-ROS agent from the container:
  ```bash
  docker exec -it parol6_dev bash -lc "
    source /opt/ros/humble/setup.bash &&
    source /microros_ws/install/setup.bash &&
    ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
  "
  ```

## Sharing
Push the built image to any registry (e.g., Docker Hub, GHCR) and teammates can `docker pull` it directly; no private base images or local scripts are required.

## Load the provided tarball locally (no rebuild/download)
If you received `parol6-ultimate-with-servo-updated.tar.gz`, just load it:
```bash
docker load -i parol6-ultimate-with-servo-updated.tar.gz
docker tag parol6-ultimate:latest parol6-ultimate:latest  # keep expected tag
```
Then run your usual workflow (e.g., `./start_ignition.sh`). No extra downloads are needed.

