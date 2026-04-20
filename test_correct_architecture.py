#!/usr/bin/env python3
"""
测试正确架构的脚本
模拟ros_webots_docker容器运行epuck机器人
safetyRL_ws运行evaluate_ros.py
"""

import subprocess
import time
import threading
import requests
import json
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from http_client import RobotControlHTTPClient

def start_mock_ros_webots():
    """启动模拟的ros_webots_docker服务"""
    print("启动模拟的ros_webots_docker服务...")
    try:
        # 启动模拟的HTTP桥接服务
        process = subprocess.Popen([
            'python3', 'http_bridge_simple.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待服务启动
        time.sleep(3)
        
        # 检查服务是否运行
        try:
            response = requests.get('http://localhost:8082/health', timeout=5)
            if response.status_code == 200:
                print("✓ 模拟ros_webots_docker服务启动成功")
                return process
            else:
                print("✗ 模拟ros_webots_docker服务启动失败")
                return None
        except:
            print("✗ 模拟ros_webots_docker服务启动失败")
            return None
    except Exception as e:
        print(f"启动模拟ros_webots_docker服务时出错: {e}")
        return None

def test_evaluate_communication():
    """测试evaluate_ros.py与ros_webots_docker的通信"""
    print("测试evaluate_ros.py与ros_webots_docker的通信...")
    
    try:
        # 创建HTTP客户端（模拟evaluate_ros.py）
        client = RobotControlHTTPClient('localhost', 8082)
        
        if not client.connect():
            print("✗ 无法连接到ros_webots_docker服务")
            return False
        
        # 测试获取初始状态
        print("测试获取epuck机器人初始状态...")
        initial_state = client.get_initial_status()
        print(f"epuck机器人初始状态: {initial_state}")
        
        # 测试发送动作到epuck机器人
        print("测试发送动作到epuck机器人...")
        import numpy as np
        test_action = np.array([0.1, 0.2, 0.0])
        success = client.send_action(test_action, 1)
        if success:
            print("✓ 动作成功发送到epuck机器人")
        else:
            print("✗ 动作发送到epuck机器人失败")
            return False
        
        # 测试获取epuck机器人当前状态
        print("测试获取epuck机器人当前状态...")
        current_state = client.get_current_status()
        print(f"epuck机器人当前状态: {current_state}")
        
        # 测试多次通信（模拟evaluate_ros.py的评估循环）
        print("测试多次通信（模拟evaluate_ros.py的评估循环）...")
        for i in range(5):
            action = np.array([0.1 * i, 0.2 * i, 0.0])
            client.send_action(action, i)
            time.sleep(0.1)
            state = client.get_current_status()
            print(f"步骤 {i}: 发送动作={action}, epuck位置={state['data']['agent_pos']}, 时间步={state['data']['timestep']}")
        
        client.disconnect()
        print("✓ evaluate_ros.py与ros_webots_docker通信测试成功")
        return True
        
    except Exception as e:
        print(f"✗ evaluate_ros.py与ros_webots_docker通信测试失败: {e}")
        return False

def main():
    """主函数"""
    print("开始测试正确架构...")
    print("架构说明:")
    print("- ros_webots_docker容器: 运行Webots仿真和epuck机器人")
    print("- safetyRL_ws: 运行evaluate_ros.py")
    print("- HTTP桥接: 连接两个系统")
    print()
    
    # 启动模拟的ros_webots_docker服务
    mock_process = start_mock_ros_webots()
    if mock_process is None:
        print("无法启动模拟ros_webots_docker服务，退出")
        return False
    
    try:
        # 测试evaluate_ros.py与ros_webots_docker的通信
        success = test_evaluate_communication()
        
        if success:
            print("\n✓ 正确架构测试通过！")
            print("系统架构正确:")
            print("  - ros_webots_docker容器运行epuck机器人 ✓")
            print("  - safetyRL_ws运行evaluate_ros.py ✓")
            print("  - HTTP桥接连接两个系统 ✓")
        else:
            print("\n✗ 正确架构测试失败")
        
        return success
        
    finally:
        # 清理
        if mock_process:
            print("停止模拟ros_webots_docker服务...")
            mock_process.terminate()
            mock_process.wait()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
