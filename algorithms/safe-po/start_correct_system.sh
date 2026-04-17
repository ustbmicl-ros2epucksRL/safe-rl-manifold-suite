#!/bin/bash

# 启动正确分离的系统
# ros_webots_docker容器运行epuck机器人
# safetyRL_ws运行evaluate_ros.py

echo "启动正确分离的系统..."

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "错误: Docker未运行，请先启动Docker"
    exit 1
fi

# 停止现有容器
echo "停止现有容器..."
docker stop ros2_webots_container 2>/dev/null || true
docker rm ros2_webots_container 2>/dev/null || true

# 启动ros_webots_docker容器
echo "启动ros_webots_docker容器..."
docker run -dit \
    --name ros2_webots_container \
    -p 11311:11311 \
    -p 1234:1234 \
    -p 8082:8082 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /home/czz/ros_webots_docker/http_bridge.py:/home/ros/http_bridge.py:ro \
    -v /home/czz/ros_webots_docker/start_epuck_system.sh:/home/ros/start_epuck_system.sh:ro \
    safe-rl/ros2_webots:v2 \
    bash -c "chmod +x /home/ros/start_epuck_system.sh && /home/ros/start_epuck_system.sh"

# 等待容器启动
echo "等待ros_webots_docker容器启动..."
sleep 15

# 检查HTTP桥接服务是否运行
echo "检查HTTP桥接服务..."
for i in {1..30}; do
    if curl -s http://localhost:8082/health > /dev/null; then
        echo "✓ HTTP桥接服务启动成功"
        break
    fi
    echo "等待HTTP桥接服务启动... ($i/30)"
    sleep 2
done

# 运行evaluate_ros.py
echo "运行evaluate_ros.py..."
cd /home/czz/safetyRL_ws/Safe-Policy-Optimization
python3 safepo/evaluate_ros.py --benchmark-dir ./runs/benchmark --eval-episodes 3 --algo ppo_cm --task Goal

echo "系统运行完成"
