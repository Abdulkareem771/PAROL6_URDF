#!/bin/bash
# Kinect Snapshot Bag Recording Script (Docker-safe, ROS-native duration control)
# Usage: ./record_kinect_snapshot.sh [duration_seconds] [output_name]

set -e

# =========================
# Configuration
# =========================
DURATION=${1:-3}  # Default 3 seconds
OUTPUT_NAME=${2:-"kinect_snapshot_bag"}
DATE=$(date +%Y%m%d_%H%M%S)
ISO_DATE=$(date -Iseconds)
OUTPUT_DIR="/workspace/test_data"
FULL_OUTPUT="${OUTPUT_DIR}/${OUTPUT_NAME}_${DATE}"
QOS_OVERRIDE="/workspace/src/parol6_vision/config/qos_override.yaml"
MIN_MESSAGES=5

# =========================
# Git commit hash
# =========================
if git -C /workspace rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_HASH=$(git -C /workspace rev-parse HEAD)
else
    GIT_HASH="not_a_git_repo"
fi

# =========================
# Topics
# =========================
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
echo "Output:   ${FULL_OUTPUT}"
echo "Git hash: ${GIT_HASH:0:8}"
echo ""

# =========================
# Prepare directories
# =========================
mkdir -p "${OUTPUT_DIR}"

# =========================
# QoS check
# =========================
if [ ! -f "${QOS_OVERRIDE}" ]; then
    echo "⚠️  WARNING: QoS override file not found:"
    echo "   ${QOS_OVERRIDE}"
    echo ""
fi

# =========================
# Topic availability check
# =========================
echo "Checking topic availability..."
MISSING_TOPICS=()

for topic in "${TOPICS[@]}"; do
    if ros2 topic info "$topic" &>/dev/null; then
        echo "  ✓ $topic"
    else
        echo "  ✗ $topic"
        MISSING_TOPICS+=("$topic")
    fi
done

if [ ${#MISSING_TOPICS[@]} -ne 0 ]; then
    echo ""
    echo "⚠️  WARNING: Missing topics:"
    for topic in "${MISSING_TOPICS[@]}"; do
        echo "  - $topic"
    done
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# =========================
# Start recording
# =========================
echo ""
echo "Starting recording in 2 seconds..."
sleep 2

echo "🔴 RECORDING for ${DURATION} seconds..."

if [ -f "${QOS_OVERRIDE}" ]; then
    ros2 bag record \
        --qos-profile-overrides-path "${QOS_OVERRIDE}" \
        --max-bag-duration ${DURATION} \
        /kinect2/qhd/image_color_rect \
        /kinect2/qhd/image_depth_rect \
        /kinect2/qhd/camera_info \
        /tf \
        /tf_static \
        -o "${FULL_OUTPUT}"
else
    ros2 bag record \
        --max-bag-duration ${DURATION} \
        /kinect2/qhd/image_color_rect \
        /kinect2/qhd/image_depth_rect \
        /kinect2/qhd/camera_info \
        /tf \
        /tf_static \
        -o "${FULL_OUTPUT}"
fi

echo "✅ Recording complete!"
echo ""

# =========================
# Validate bag
# =========================
echo "Validating bag contents..."
BAG_INFO=$(ros2 bag info "${FULL_OUTPUT}" 2>&1)
echo "$BAG_INFO"

MESSAGE_COUNT=$(echo "$BAG_INFO" | grep -oP 'Messages:\s+\K\d+' || echo "0")

if [ "$MESSAGE_COUNT" -lt "$MIN_MESSAGES" ]; then
    echo ""
    echo "❌ ERROR: Insufficient messages: ${MESSAGE_COUNT}"
    echo "Possible causes:"
    echo "  - Topics not publishing"
    echo "  - QoS mismatch"
    echo "  - Duration too short"
    exit 1
fi

echo "✅ Validation passed: ${MESSAGE_COUNT} messages"
echo ""

# =========================
# Metadata JSON
# =========================
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
  "qos_override_used": $([ -f "${QOS_OVERRIDE}" ] && echo "true" || echo "false")
}
EOF

echo "✅ Metadata saved: ${METADATA_FILE}"

# =========================
# Checksums
# =========================
echo "Generating checksums..."
if ls ${FULL_OUTPUT}/*.db3 1> /dev/null 2>&1; then
    sha256sum ${FULL_OUTPUT}/*.db3 > ${FULL_OUTPUT}/checksums.sha256
    echo "✅ Checksums saved."
else
    echo "⚠️  No .db3 files found."
fi

# =========================
# README
# =========================
README_FILE="${FULL_OUTPUT}/README.md"

cat > "${README_FILE}" << READMEEOF
# Kinect Snapshot Dataset

- Dataset ID: ${OUTPUT_NAME}_${DATE}
- Date: ${ISO_DATE}
- Duration: ${DURATION}s
- Messages: ${MESSAGE_COUNT}
- Git commit: ${GIT_HASH}

## Topics
- /kinect2/qhd/image_color_rect
- /kinect2/qhd/image_depth_rect
- /kinect2/qhd/camera_info
- /tf
- /tf_static

## Replay
\`\`\`bash
ros2 bag play ${FULL_OUTPUT}
\`\`\`

## Integrity
\`\`\`bash
sha256sum -c checksums.sha256
\`\`\`

Generated automatically.
READMEEOF

echo "✅ README created."
echo ""
echo "🎉 Dataset ready: ${FULL_OUTPUT}"
