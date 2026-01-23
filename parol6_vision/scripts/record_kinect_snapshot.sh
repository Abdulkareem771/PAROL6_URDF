#!/bin/bash
# Kinect Snapshot Bag Recording Script
# Usage: ./record_kinect_snapshot.sh [duration_seconds] [output_name]

set -e

# Configuration
DURATION=${1:-3}  # Default 3 seconds
OUTPUT_NAME=${2:-"kinect_snapshot_bag"}
DATE=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/workspace/test_data"
FULL_OUTPUT="${OUTPUT_DIR}/${OUTPUT_NAME}_${DATE}"

# Required topics
TOPICS=(
    "/kinect2/qhd/image_color_rect"
    "/kinect2/qhd/image_depth_rect"
    "/kinect2/qhd/camera_info"
    "/tf"
    "/tf_static"
)

echo "=========================================="
echo "Kinect Snapshot Bag Recorder"
echo "=========================================="
echo "Duration: ${DURATION} seconds"
echo "Output: ${FULL_OUTPUT}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if topics are available
echo "Checking topic availability..."
MISSING_TOPICS=()
for topic in "${TOPICS[@]}"; do
    if ! ros2 topic info "$topic" &>/dev/null; then
        MISSING_TOPICS+=("$topic")
    else
        echo "  ✓ $topic"
    fi
done

if [ ${#MISSING_TOPICS[@]} -ne 0 ]; then
    echo ""
    echo "⚠️  WARNING: The following topics are not available:"
    for topic in "${MISSING_TOPICS[@]}"; do
        echo "  ✗ $topic"
    done
    echo ""
    echo "Make sure kinect2_bridge is running:"
    echo "  ros2 launch kinect2_bridge kinect2_bridge.launch.py"
    echo ""
    read -p "Continue recording anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Start recording
echo ""
echo "Starting recording in 2 seconds..."
sleep 2

echo "🔴 RECORDING..."
timeout ${DURATION}s ros2 bag record \
    /kinect2/qhd/image_color_rect \
    /kinect2/qhd/image_depth_rect \
    /kinect2/qhd/camera_info \
    /tf \
    /tf_static \
    -o "${FULL_OUTPUT}" || true

echo "✅ Recording complete!"
echo ""

# Verify bag
echo "Bag information:"
echo "----------------------------------------"
ros2 bag info "${FULL_OUTPUT}"
echo "----------------------------------------"
echo ""

# Create metadata
METADATA_FILE="${FULL_OUTPUT}/metadata.json"
cat > "${METADATA_FILE}" << EOF
{
  "dataset_id": "${OUTPUT_NAME}_${DATE}",
  "date": "${DATE}",
  "duration_sec": ${DURATION},
  "camera": "Kinect v2",
  "resolution": "960x540 (QHD)",
  "topics": [
    "/kinect2/qhd/image_color_rect",
    "/kinect2/qhd/image_depth_rect",
    "/kinect2/qhd/camera_info",
    "/tf",
    "/tf_static"
  ],
  "environment": "Lab workspace",
  "notes": "Captured for vision pipeline development"
}
EOF

echo "✅ Metadata saved to: ${METADATA_FILE}"
echo ""
echo "📦 To compress for sharing:"
echo "  cd ${OUTPUT_DIR}"
echo "  tar czf ${OUTPUT_NAME}_${DATE}.tar.gz ${OUTPUT_NAME}_${DATE}"
echo ""
echo "🔁 To replay:"
echo "  ros2 bag play ${FULL_OUTPUT} --loop"
