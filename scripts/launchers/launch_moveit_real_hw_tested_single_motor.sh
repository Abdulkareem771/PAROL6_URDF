#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Method 5 intentionally reuses the standard real-hardware bringup so the GUI
# path matches the teammate's working RViz/controller topology.
exec ./scripts/launchers/launch_moveit_real_hw.sh
