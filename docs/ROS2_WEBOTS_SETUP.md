# ROS2 + Webots + E-puck 环境配置指南

## 系统要求

| 组件 | 推荐版本 | 备注 |
|------|---------|------|
| **操作系统** | Ubuntu 22.04 LTS | ROS2 Humble 官方支持 |
| **ROS2** | Humble Hawksbill | LTS 版本，支持到 2027 |
| **Webots** | R2023b 或更新 | 开源机器人仿真器 |
| **Python** | 3.10+ | ROS2 Humble 默认版本 |
| **GPU** | 可选 | 用于训练加速 |

## 一、安装 ROS2 Humble

```bash
# 1. 设置 locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 2. 添加 ROS2 APT 源
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 3. 安装 ROS2
sudo apt update
sudo apt install ros-humble-desktop -y

# 4. 安装开发工具
sudo apt install python3-colcon-common-extensions python3-rosdep -y

# 5. 初始化 rosdep
sudo rosdep init
rosdep update

# 6. 添加到 bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 7. 验证安装
ros2 --version
```

## 二、安装 Webots

### 方法 1: APT 安装 (推荐)

```bash
# 添加 Webots APT 源
sudo mkdir -p /etc/apt/keyrings
cd /etc/apt/keyrings
sudo wget -q https://cyberbotics.com/Cyberbotics.asc
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/Cyberbotics.asc] https://cyberbotics.com/debian binary-amd64/" | sudo tee /etc/apt/sources.list.d/Cyberbotics.list

# 安装
sudo apt update
sudo apt install webots -y

# 设置环境变量
echo 'export WEBOTS_HOME=/usr/local/webots' >> ~/.bashrc
source ~/.bashrc
```

### 方法 2: 官网下载

1. 访问 https://cyberbotics.com/download
2. 下载 Linux 版本 (.deb)
3. 安装: `sudo dpkg -i webots_*.deb && sudo apt install -f`

### 方法 3: Snap 安装

```bash
sudo snap install webots
```

## 三、安装 Webots-ROS2 桥接

```bash
# 安装 webots_ros2 包
sudo apt install ros-humble-webots-ros2 -y

# 安装 E-puck 支持
sudo apt install ros-humble-webots-ros2-epuck -y

# 验证
ros2 pkg list | grep webots
```

## 四、配置项目工作空间

```bash
# 1. 创建工作空间
mkdir -p ~/cosmos_ws/src
cd ~/cosmos_ws/src

# 2. 克隆项目
git clone https://github.com/ustbmicl-ros2epucksRL/safe-rl-manifold-suite.git

# 3. 复制 ROS2 包
cp -r safe-rl-manifold-suite/ros2_ws/src/epuck_formation .

# 4. 安装 Python 依赖
cd ~/cosmos_ws
pip3 install torch numpy scipy matplotlib hydra-core omegaconf

# 5. 安装 COSMOS
pip3 install -e src/safe-rl-manifold-suite

# 6. 安装 ROS 依赖
rosdep install --from-paths src --ignore-src -r -y

# 7. 构建
colcon build --symlink-install

# 8. Source 环境
source install/setup.bash
echo "source ~/cosmos_ws/install/setup.bash" >> ~/.bashrc
```

## 五、运行仿真

### 5.1 启动 E-puck 编队控制

```bash
# 终端 1: 启动 Webots 仿真
ros2 launch epuck_formation epuck_formation.launch.py

# 终端 2: 发送编队目标
ros2 topic pub /formation/goal geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'world'}, pose: {position: {x: 0.3, y: 0.3, z: 0}}}"

# 终端 3: 查看状态
ros2 topic echo /formation/status
```

### 5.2 参数配置

```bash
# 使用 6 个机器人
ros2 launch epuck_formation epuck_formation.launch.py num_robots:=6

# 关闭安全滤波器
ros2 launch epuck_formation epuck_formation.launch.py use_safety:=false

# 使用训练好的策略
ros2 launch epuck_formation epuck_formation.launch.py \
  policy_model:=/path/to/model.pt
```

### 5.3 话题列表

| 话题 | 类型 | 方向 | 描述 |
|------|------|------|------|
| `/epuck{i}/odom` | nav_msgs/Odometry | Sub | 机器人里程计 |
| `/epuck{i}/proximity` | std_msgs/Float32MultiArray | Sub | 8个红外传感器 |
| `/epuck{i}/cmd_vel` | geometry_msgs/Twist | Pub | 速度命令 |
| `/formation/goal` | geometry_msgs/PoseStamped | Sub | 编队目标位置 |
| `/formation/status` | std_msgs/String | Pub | 编队状态 |

## 六、训练与部署流程

```
┌─────────────────────────────────────────────────────────────┐
│                    训练与部署流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 快速训练 (纯 Python, ~10k steps/sec)                    │
│     python -m cosmos.train env=epuck_sim algo=mappo         │
│                        ↓                                    │
│  2. Webots 验证 (高保真, ~100 steps/sec)                    │
│     python -m cosmos.train env=webots_epuck algo=mappo      │
│                        ↓                                    │
│  3. ROS2 部署 (话题接口)                                     │
│     ros2 launch epuck_formation epuck_formation.launch.py   │
│                        ↓                                    │
│  4. 实物部署 (相同话题接口)                                  │
│     ros2 launch epuck_formation epuck_real.launch.py        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 七、常见问题

### Q1: Webots 启动失败

```bash
# 检查 WEBOTS_HOME
echo $WEBOTS_HOME

# 如果为空，设置它
export WEBOTS_HOME=/usr/local/webots

# 检查显示
echo $DISPLAY

# 如果在远程服务器，使用虚拟显示
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

### Q2: ROS2 包找不到

```bash
# 确保已 source
source /opt/ros/humble/setup.bash
source ~/cosmos_ws/install/setup.bash

# 重新构建
cd ~/cosmos_ws
colcon build --symlink-install
```

### Q3: Python 包导入失败

```bash
# 确保 COSMOS 已安装
pip3 install -e ~/cosmos_ws/src/safe-rl-manifold-suite

# 检查 Python 路径
python3 -c "import cosmos; print(cosmos.__file__)"
```

### Q4: 多机器人同步问题

```bash
# 使用 use_sim_time
ros2 launch epuck_formation epuck_formation.launch.py \
  --ros-args -p use_sim_time:=true
```

## 八、macOS 用户

macOS 不支持完整的 ROS2，推荐使用：

1. **Docker**: 运行 Ubuntu + ROS2 容器
2. **纯 Python 模式**: 使用 `epuck_sim` 环境

```bash
# macOS: 使用纯 Python 仿真
cd safe-rl-manifold-suite
pip install -e .
python -m cosmos.train env=epuck_sim algo=mappo safety=cbf
```

## 参考链接

- [ROS2 Humble 安装](https://docs.ros.org/en/humble/Installation.html)
- [Webots 官网](https://cyberbotics.com/)
- [webots_ros2 文档](https://github.com/cyberbotics/webots_ros2)
- [E-puck 官方文档](https://www.gctronic.com/doc/index.php/E-puck2)
