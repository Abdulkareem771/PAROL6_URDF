#!/bin/bash
# Start Mobile Control Bridge for PAROL6

echo "🚀 Starting PAROL6 Mobile Control Bridge"
echo "================================================"
echo

# Check if container is running
if ! docker ps | grep -q parol6_dev; then
    echo "❌ ERROR: parol6_dev container is not running!"
    echo "Please start the simulation first:"
    echo "  ./start_ignition.sh"
    exit 1
fi

# Install Flask if not present
docker exec parol6_dev bash -c "pip list | grep -q Flask || pip install flask flask-cors"

# Start the bridge
echo "📡 Starting Mobile Bridge on http://localhost:5000"
echo
docker exec -it parol6_dev bash -c "cd /workspace && python3 mobile_bridge.py"
