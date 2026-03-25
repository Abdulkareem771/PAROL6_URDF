# =========================
# Start recording (robust method)
# =========================
echo ""
echo "Starting recording in 2 seconds..."
sleep 2

echo "🔴 RECORDING for ${DURATION} seconds..."

if [ -f "${QOS_OVERRIDE}" ]; then
    ros2 bag record \
        --qos-profile-overrides-path "${QOS_OVERRIDE}" \
        /kinect2/qhd/image_color_rect \
        /kinect2/qhd/image_depth_rect \
        /kinect2/qhd/camera_info \
        /tf \
        /tf_static \
        -o "${FULL_OUTPUT}" &
else
    ros2 bag record \
        /kinect2/qhd/image_color_rect \
        /kinect2/qhd/image_depth_rect \
        /kinect2/qhd/camera_info \
        /tf \
        /tf_static \
        -o "${FULL_OUTPUT}" &
fi

RECORDER_PID=$!

# Wait desired duration
sleep "${DURATION}"

# Gracefully stop rosbag (same as Ctrl+C)
echo "🛑 Stopping recorder..."
kill -INT ${RECORDER_PID}

# Wait for clean shutdown
wait ${RECORDER_PID} 2>/dev/null || true

echo "✅ Recording complete!"
echo ""
