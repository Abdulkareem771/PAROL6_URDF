# PAROL6 Robot Workspace

## 🚀 Quick Start

1. **Start the Simulator** (Ignition Gazebo):
   ```bash
   ./start_ignition.sh
   ```

2. **Enable Motion Planning** (MoveIt 2):
   *Open a new terminal and run:*
   ```bash
   ./add_moveit.sh
   ```

---

## 📂 Project Structure

- **`start_ignition.sh`**: Main entry point. Starts the Docker container and launches the robot simulation.
- **`add_moveit.sh`**: Adds MoveIt 2 motion planning capabilities to the running simulation.
- **`docs/`**: Detailed documentation, guides, and notes.
- **`scripts/`**: Helper scripts and utilities.
  - **`scripts/setup/`**: Scripts for building the Docker image and environment setup.
  - **`scripts/legacy/`**: Older scripts for Gazebo Classic and other modes.
  - **`scripts/utils/`**: Utility scripts for diagnostics and maintenance.
- **`PAROL6/`**: Main robot description package (URDF, meshes).
- **`parol6_moveit_config/`**: MoveIt configuration package.

## 🛠️ Setup & Installation

This project uses **Docker** to ensure a consistent environment.

1. **Install Docker** on your system.
2. **Build/Pull the Image**:
   The `start_ignition.sh` script will automatically look for the `parol6-ultimate:latest` image.
   If you need to rebuild it manually, run:
   ```bash
   ./scripts/setup/rebuild_image.sh
   ```

## � Documentation
Check the `docs/` folder for more details:
- [Project Explanation](docs/PROJECT_EXPLANATION.md)
- [Gazebo Guide](docs/GAZEBO_NOTE.md)
- [Ultimate Image Guide](docs/ULTIMATE_IMAGE.md)
