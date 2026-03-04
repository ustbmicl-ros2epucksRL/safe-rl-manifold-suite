#!/bin/bash
# ROS2 + Webots Safety Navigation Demo
#
# Launches Webots with E-puck arena via ROS2 interface,
# then runs the safety navigation node.
#
# Usage:
#   bash webots/ros2/run_ros2_demo.sh [safe|unsafe]

set -e

# Deactivate conda (ROS2 Jazzy needs system Python 3.12)
eval "$(conda shell.bash hook)" 2>/dev/null
conda deactivate 2>/dev/null

# Clear proxy (Webots can't bind TCP with SOCKS proxy)
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# Setup
export WEBOTS_HOME=/usr/local/webots
source /opt/ros/jazzy/setup.bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IROS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORLD_FILE="$IROS_DIR/webots/worlds/epuck_navigation_ros2.wbt"
URDF_FILE="$SCRIPT_DIR/resource/epuck_safety.urdf"
ROS2_CTRL="$(ros2 pkg prefix webots_ros2_epuck)/share/webots_ros2_epuck/resource/ros2_control.yml"

MODE="${1:-safe}"
echo "=================================="
echo "ROS2 Safety Navigation Demo"
echo "Mode: $MODE"
echo "World: $WORLD_FILE"
echo "=================================="

# Launch Webots + ROS2 driver in background
ros2 launch webots_ros2_epuck robot_launch.py world:="$WORLD_FILE" &
ROS2_PID=$!

echo "Waiting for Webots ROS2 to initialize..."
sleep 15

echo "=== Active topics ==="
ros2 topic list

# Run safety nav node
echo "Starting safety navigation node (mode=$MODE)..."
PYTHONPATH="$IROS_DIR/src:$PYTHONPATH" python3 "$SCRIPT_DIR/scripts/safety_nav_node.py" \
    --ros-args \
    -p goal_x:=0.5 \
    -p goal_y:=0.5 \
    -p use_safety:=$([ "$MODE" = "safe" ] && echo true || echo false) \
    -p use_ekf:=$([ "$MODE" = "safe" ] && echo true || echo false) &
NAV_PID=$!

echo "Safety nav PID: $NAV_PID, ROS2 PID: $ROS2_PID"
echo "Press Ctrl+C to stop"

# Wait for either process
wait $NAV_PID $ROS2_PID
