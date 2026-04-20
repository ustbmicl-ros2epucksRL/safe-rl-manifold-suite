# 正确架构说明文档

本文档说明正确的系统架构，将epuck机器人运行在ros_webots_docker容器中，evaluate_ros.py运行在safetyRL_ws中。

## 正确架构

```
┌─────────────────────────────────────┐
│           safetyRL_ws               │
│  ┌─────────────────────────────────┐ │
│  │      evaluate_ros.py           │ │
│  │  ┌─────────────────────────────┐│ │
│  │  │     HTTP客户端              ││ │
│  │  └─────────────────────────────┘│ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
                    │
                    │ HTTP通信
                    │
┌─────────────────────────────────────┐
│        ros_webots_docker            │
│  ┌─────────────────────────────────┐ │
│  │      HTTP桥接服务              │ │
│  │  ┌─────────────────────────────┐│ │
│  │  │     ROS/Webots              ││ │
│  │  │  ┌─────────────────────────┐││ │
│  │  │  │    epuck机器人          │││ │
│  │  │  └─────────────────────────┘││ │
│  │  └─────────────────────────────┘│ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

## 文件分布

### ros_webots_docker容器中的文件：
```
/home/czz/ros_webots_docker/
├── http_bridge.py              # HTTP桥接服务
├── start_epuck_system.sh       # 启动epuck系统脚本
├── Dockerfile                  # 容器配置
└── webots_ws/                  # Webots工作空间
    └── src/webots_ros2/
        └── webots_ros2_epuck/  # epuck机器人包
```

### safetyRL_ws中的文件：
```
/home/czz/safetyRL_ws/Safe-Policy-Optimization/
├── http_client.py              # HTTP客户端
├── safepo/evaluate_ros.py      # 评估脚本
├── test_correct_architecture.py # 架构测试脚本
├── start_correct_system.sh     # 启动脚本
└── test_data/                  # 测试数据
```

## 使用方法

### 方法1：本地测试（推荐）

1. **测试正确架构**：
   ```bash
   cd /home/czz/safetyRL_ws/Safe-Policy-Optimization
   python3 test_correct_architecture.py
   ```

2. **运行完整系统**：
   ```bash
   ./start_correct_system.sh
   ```

### 方法2：Docker集成

1. **启动Docker服务**：
   ```bash
   docker-compose -f docker-compose-correct.yml up -d
   ```

2. **查看日志**：
   ```bash
   docker-compose -f docker-compose-correct.yml logs -f
   ```

## 系统组件说明

### 1. ros_webots_docker容器
- **职责**: 运行Webots仿真和epuck机器人
- **包含**: 
  - Webots仿真环境
  - epuck机器人模型
  - ROS节点和话题
  - HTTP桥接服务
- **端口**: 
  - 11311 (ROS Master)
  - 1234 (Webots Web界面)
  - 8082 (HTTP API)

### 2. safetyRL_ws
- **职责**: 运行evaluate_ros.py
- **包含**:
  - 强化学习评估脚本
  - HTTP客户端
  - 测试数据
- **功能**: 发送动作到epuck机器人，接收状态反馈

### 3. HTTP桥接服务
- **位置**: ros_webots_docker容器中
- **功能**: 
  - 接收evaluate_ros.py的动作
  - 转换为ROS cmd_vel消息
  - 发送给epuck机器人
  - 收集epuck机器人状态
  - 通过HTTP API返回给evaluate_ros.py

## 数据流

1. **动作流**: evaluate_ros.py → HTTP客户端 → HTTP桥接服务 → ROS cmd_vel → epuck机器人
2. **状态流**: epuck机器人 → ROS话题 → HTTP桥接服务 → HTTP API → evaluate_ros.py

## 配置说明

### HTTP客户端配置
在 `safepo/evaluate_ros.py` 中：
```python
# HTTP服务器配置
http_server_address = 'localhost'  # 或容器IP
http_server_port = 8082
```

### ROS话题映射
- `/cmd_vel` - 速度命令（Twist消息）
- `/odom` - 里程计数据
- `/imu` - IMU数据
- `/magnetic_field` - 磁力计数据

## 启动顺序

1. **启动ros_webots_docker容器**：
   - 启动roscore
   - 启动HTTP桥接服务
   - 启动Webots仿真
   - 启动epuck机器人

2. **运行evaluate_ros.py**：
   - 连接到HTTP桥接服务
   - 开始评估循环
   - 发送动作，接收状态

## 测试和调试

### 1. 检查ros_webots_docker容器
```bash
# 查看容器状态
docker ps

# 进入容器
docker exec -it ros2_webots_container bash

# 查看ROS话题
rostopic list
rostopic echo /cmd_vel
```

### 2. 检查HTTP桥接服务
```bash
# 健康检查
curl http://localhost:8082/health

# 获取机器人状态
curl http://localhost:8082/api/robot/current_status
```

### 3. 检查evaluate_ros.py
```bash
# 运行测试
python3 test_correct_architecture.py
```

## 优势

1. **职责分离**: 每个容器有明确的职责
2. **独立开发**: 可以独立开发和测试各个组件
3. **资源隔离**: 避免资源冲突
4. **易于维护**: 清晰的架构便于维护
5. **可扩展性**: 易于添加新功能

## 注意事项

1. 确保ros_webots_docker容器先启动
2. 检查端口映射是否正确
3. 确保网络连接正常
4. 定期检查容器状态
5. 查看日志输出

## 故障排除

### 1. 容器启动失败
- 检查Docker是否运行
- 检查端口是否被占用
- 检查镜像是否存在

### 2. HTTP连接失败
- 检查HTTP桥接服务是否运行
- 检查端口映射
- 检查网络连接

### 3. ROS通信问题
- 检查roscore是否运行
- 检查话题是否正确发布
- 检查环境变量设置

这个架构确保了epuck机器人运行在ros_webots_docker容器中，evaluate_ros.py运行在safetyRL_ws中，通过HTTP桥接服务进行通信。
