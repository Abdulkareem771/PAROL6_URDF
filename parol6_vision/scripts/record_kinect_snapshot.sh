#!/bin/bash
# Kinect Snapshot Bag Recording Script (Production Grade)
# Usage: ./record_kinect_snapshot.sh [duration_seconds] [output_name]

set -e

# Configuration
DURATION=${1:-3}  # Default 3 seconds
OUTPUT_NAME=${2:-"kinect_snapshot_bag"}
DATE=$(date +%Y%m%d_%H%M%S)
ISO_DATE=$(date -Iseconds)
OUTPUT_DIR="/workspace/test_data"
FULL_OUTPUT="${OUTPUT_DIR}/${OUTPUT_NAME}_${DATE}"
QOS_OVERRIDE="/workspace/src/parol6_vision/config/qos_override.yaml"
MIN_MESSAGES=5  # Minimum messages to consider success

# Git commit hash for reproducibility (robust check)
if git -C /workspace rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_HASH=$(git -C /workspace rev-parse HEAD)
else
    GIT_HASH="not_a_git_repo"
fi

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
echo "Git commit: ${GIT_HASH:0:8}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if QoS override file exists
if [ ! -f "${QOS_OVERRIDE}" ]; then
    echo "⚠️  WARNING: QoS override file not found: ${QOS_OVERRIDE}"
    echo "Recording may drop messages on some systems."
    echo ""
fi

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
if [ -f "${QOS_OVERRIDE}" ]; then
    timeout ${DURATION}s ros2 bag record \
        --qos-profile-overrides-path "${QOS_OVERRIDE}" \
        /kinect2/qhd/image_color_rect \
        /kinect2/qhd/image_depth_rect \
        /kinect2/qhd/camera_info \
        /tf \
        /tf_static \
        -o "${FULL_OUTPUT}" || true
else
    # Fallback without QoS override
    timeout ${DURATION}s ros2 bag record \
        /kinect2/qhd/image_color_rect \
        /kinect2/qhd/image_depth_rect \
        /kinect2/qhd/camera_info \
        /tf \
        /tf_static \
        -o "${FULL_OUTPUT}" || true
fi

echo "✅ Recording complete!"
echo ""

# Verify bag contains messages
echo "Validating bag contents..."
BAG_INFO=$(ros2 bag info "${FULL_OUTPUT}" 2>&1)
echo "$BAG_INFO"

# Extract message count and validate
MESSAGE_COUNT=$(echo "$BAG_INFO" | grep -oP 'Messages:\s+\K\d+' || echo "0")
if [ "$MESSAGE_COUNT" -lt "$MIN_MESSAGES" ]; then
    echo ""
    echo "❌ ERROR: Bag contains insufficient messages (${MESSAGE_COUNT} < ${MIN_MESSAGES})"
    echo "This usually means:"
    echo "  - Topics were not publishing during recording"
    echo "  - QoS mismatch prevented message capture"
    echo "  - Duration was too short"
    echo ""
    echo "Bag directory: ${FULL_OUTPUT}"
    exit 1
fi

echo ""
echo "✅ Validation passed: ${MESSAGE_COUNT} messages recorded"
echo "----------------------------------------"
echo ""

# Create metadata
METADATA_FILE="${FULL_OUTPUT}/metadata.json"
cat > "${METADATA_FILE}" << EOF
{
  "dataset_id": "${OUTPUT_NAME}_${DATE}",
  "date_iso": "${ISO_DATE}",
  "date_compact": "${DATE}",
  "duration_sec": ${DURATION},
  "git_commit": "${GIT_HASH}",
  "camera": "Kinect v2",
  "resolution": "960x540 (QHD)",
  "message_count": ${MESSAGE_COUNT},
  "topics": [
    "/kinect2/qhd/image_color_rect",
    "/kinect2/qhd/image_depth_rect",
    "/kinect2/qhd/camera_info",
    "/tf",
    "/tf_static"
  ],
  "environment": "Lab workspace",
  "notes": "Captured for vision pipeline development",
  "qos_override_used": $([ -f "${QOS_OVERRIDE}" ] && echo "true" || echo "false")
}
EOF

echo "✅ Metadata saved to: ${METADATA_FILE}"
echo ""

# Generate checksums for data integrity verification
echo "Generating checksums for data integrity..."
if ls ${FULL_OUTPUT}/*.db3 1> /dev/null 2>&1; then
    sha256sum ${FULL_OUTPUT}/*.db3 > ${FULL_OUTPUT}/checksums.sha256
    echo "✅ Checksums saved to: ${FULL_OUTPUT}/checksums.sha256"
else
    echo "⚠️  No .db3 files found for checksum generation"
fi
echo ""

# Create auto-generated README in dataset folder
README_FILE="${FULL_OUTPUT}/README.md"
cat > "${README_FILE}" << READMEEOF
# Kinect Snapshot Dataset

## Dataset Information
- **Dataset ID:** ${OUTPUT_NAME}_${DATE}
- **Capture Date:** ${ISO_DATE}
- **Duration:** ${DURATION} seconds
- **Message Count:** ${MESSAGE_COUNT}
- **Git Commit:** \`${GIT_HASH}\`

## Camera Configuration
- **Model:** Kinect v2
- **Resolution:** 960x540 (QHD)
- **Environment:** Lab workspace

## Topics Recorded
- \`/kinect2/qhd/image_color_rect\` - RGB image (rectified)
- \`/kinect2/qhd/image_depth_rect\` - Depth map (rectified)
- \`/kinect2/qhd/camera_info\` - Camera calibration
- \`/tf\` - Dynamic transforms
- \`/tf_static\` - Static transforms

## Usage

### Replay in Loop
\`\`\`bash
ros2 bag play $(basename ${FULL_OUTPUT}) --loop
\`\`\`

### Verify Integrity
\`\`\`bash
sha256sum -c checksums.sha256
\`\`\`

### View Bag Info
\`\`\`bash
ros2 bag info $(basename ${FULL_OUTPUT})
\`\`\`

## Files
- \`metadata.json\` - Structured dataset metadata
- \`checksums.sha256\` - Data integrity checksums
- \`*.db3\` - ROS bag database files
- \`metadata.yaml\` - ROS bag metadata

## Notes
Captured for vision pipeline development. See \`metadata.json\` for detailed information.

---
Generated automatically by \`record_kinect_snapshot.sh\`
READMEEOF

echo "✅ Dataset README created: ${README_FILE}"
echo ""

echo "📦 To compress for sharing:"
echo "  cd ${OUTPUT_DIR}"
echo "  tar czf ${OUTPUT_NAME}_${DATE}.tar.gz ${OUTPUT_NAME}_${DATE}"
echo ""
echo "🔁 To replay:"
echo "  ros2 bag play ${FULL_OUTPUT} --loop"
echo ""
echo "✅ To verify integrity:"
echo "  cd ${FULL_OUTPUT} && sha256sum -c checksums.sha256"
echo ""
echo "🎉 Success! Dataset ready for distribution."

