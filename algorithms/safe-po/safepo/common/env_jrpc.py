from typing import Dict, Any, Optional
from jsonrpcclient.requests import request_json as jrpc_request_json
from jsonrpcclient.responses import parse_json as jrpc_response_parse
from jsonrpcclient.responses import Error
import requests
import time
import numpy as np

# 后续可以尝试直接实现一个task类对象
# from safety_gymnasium.bases.base_task import BaseTask
from gymnasium.spaces import Box


class RemoteEnvJRPC:
    def __init__(
        self, hazard_num=8, vase_num=1, server: str = "ros2-webots-compose-czz", port=10086
    ) -> None:
        # json rpc 参数配置
        self.server_url = "http://" + server + ":" + str(port)
        self.jrpc_headers = {"Content-Type": "application/json"}
        self.time_step = 0.2
        self.last_obs = {}
        # task相关属性：
        self.action_space = Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float64
        )
        # 生成（虚拟）障碍物坐标
        self.hazard = np.random.random((hazard_num, 2)) * 3
        self.hazard = np.hstack((self.hazard, np.zeros((hazard_num, 1))))
        self.vase = np.random.random((vase_num, 2)) * 3
        self.vase = np.hstack((self.vase, np.zeros((vase_num, 1))))
        # print(f"Generate {self.hazard.shape[0]} hazard points")
        # for index in range(self.hazard.shape[0]):
        #     print(f"\t-> {self.hazard[index]}")
        # 生成（虚拟）目标坐标
        self.goal = np.random.random((1, 2)) * 3
        self.goal = np.hstack((self.goal, np.zeros((self.goal.shape[0], 1))))
        # print(f"Generate {self.goal.shape[0]} goal points")
        # for index in range(self.goal.shape[0]):
        #     print(f"\t-> {self.goal[index]}")
        # 障碍物雷达线数
        self.hazard_lidar_num = 16
        self.hazard_size = 0.2
        self.hazard_cost = 1.0
        # 目标雷达线数
        self.goal_lidar_num = 16
        self.goal_size = 0.2
        self.goal_reward = 1.0
        self.goal_reward_distance = 1.0
        self.goal_last_distance = np.linalg.norm(self.goal[:2])

    def __remote_call(self, req_json: str, jrpc_timeout=1) -> Optional[Dict[str, Any]]:
        """
        调用 JSON-RPC 方法
        """
        try:
            response = requests.post(
                self.server_url,
                data=req_json,
                headers=self.jrpc_headers,
                timeout=jrpc_timeout,
            )

            response.raise_for_status()
            result = jrpc_response_parse(response.text)

            # 检查 JSON-RPC 错误
            if isinstance(result, Error):
                print(f"JSON-RPC 错误: {result.code} - {result.message}")
                return None

            return result.result

        except Exception as e:
            print(f"调用方法失败: {e}")
            return None

    def _jrpc_get_obs(self) -> Optional[Dict[str, Any]]:
        request_json = jrpc_request_json("get_obs", {})
        result = self.__remote_call(request_json)
        if result is not None:
            return result
        else:
            print(f"Error: 获取观测值失败，request json - {request_json}")
            return None

    def _jrpc_act(self, action) -> Optional[Any]:
        request_json = jrpc_request_json("act", {"v": action[0], "w": action[1]})
        result = self.__remote_call(request_json)
        if result is None:
            print(f"Error: act方法调用失败， request json - {request_json}")
        return result

    def _jrpc_reset(self) -> Optional[Any]:
        request_json = jrpc_request_json("reset", {})
        result = self.__remote_call(request_json, jrpc_timeout=10)
        if result is None:
            print(f"Error: reset方法调用失败， request json - {request_json}")
        return result

    def __calculate_lidar(self, pose, angle):
        """计算虚拟雷达观测结果，包含hazard和goal两个雷达的结果

        Args:
            pose: 智能体位置 (x, y, z)
            angle: 智能体yaw航向角（弧度制）

        Returns:
            tuple: (hazard_lidar, goal_lidar) 两个雷达观测数组
        """

        def calculate_single_lidar(target_positions, num_bins, max_dist=3.0):
            """计算单个雷达的观测结果"""
            obs = np.zeros(num_bins)

            for pos in target_positions:
                pos = np.asarray(pos)
                # 忽略Z坐标，使用XY平面
                if pos.shape == (3,):
                    pos = pos[:2]

                # 转换为以智能体为中心的坐标系
                dx = pos[0] - pose[0]
                dy = pos[1] - pose[1]

                # 考虑智能体朝向，进行坐标旋转
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                # 旋转到智能体坐标系
                rotated_x = dx * cos_angle + dy * sin_angle
                rotated_y = -dx * sin_angle + dy * cos_angle

                # 使用复数表示相对位置
                z = complex(rotated_x, rotated_y)

                # 计算距离和角度
                dist = np.abs(z)
                target_angle = np.angle(z) % (2 * np.pi)

                # 计算bin信息
                bin_size = (2 * np.pi) / num_bins
                bin_idx = int(target_angle / bin_size)
                bin_angle = bin_size * bin_idx

                # 计算传感器读数（线性归一化）
                sensor = max(0.0, max_dist - dist) / max_dist

                # 更新观测值
                obs[bin_idx] = max(obs[bin_idx], sensor)

                # 抗锯齿处理
                alias = (target_angle - bin_angle) / bin_size
                assert 0 <= alias <= 1, f"bad alias {alias}"

                bin_plus = (bin_idx + 1) % num_bins
                bin_minus = (bin_idx - 1) % num_bins

                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)

            return obs

        # 计算hazard雷达观测
        hazard_lidar = calculate_single_lidar(
            self.hazard, self.hazard_lidar_num, max_dist=3.0
        )

        # 计算goal雷达观测
        goal_lidar = calculate_single_lidar(
            self.goal, self.goal_lidar_num, max_dist=3.0
        )

        return hazard_lidar, goal_lidar

    def __calculate_cost(self):
        agent_xy = self.last_obs["agent_pos"][:2]

        hazard_xy = self.hazard[:, :2]
        distances = np.linalg.norm(hazard_xy - agent_xy, axis=1)
        hazard_costs = self.hazard_cost * np.maximum(0.0, self.hazard_size - distances)

        return {"cost_hazards": np.sum(hazard_costs)}

    def __calculate_reward(self):
        agent_xy = self.last_obs["agent_pos"][:2]
        goal_xy = self.goal[0, :2]
        dist = np.linalg.norm(agent_xy - goal_xy)
        if self.goal_last_distance is None:
            reward = 0
        else:
            reward = (self.goal_last_distance - dist) * self.goal_reward_distance
        if dist <= self.goal_size:
            reward += self.goal_reward
        return reward

    def update_obs(self) -> Optional[Dict[str, Any]]:
        """观测值格式
        self.observation = {
            "accelerometer": [0.0, 0.0, 0.0],
            "velocimeter": [0.0, 0.0, 0.0],
            "gyro": [0.0, 0.0, 0.0],
            "magnetometer": [0.0, 0.0, 0.0],
            "agent_pos": [0.0, 0.0, 0.0],
            "yaw": 0.0 # 航向角（弧度制）
        }
        """
        # 获取jrpc obs
        #   - 暂时认为里程计信息准确
        #   - 智能体的位置和方向信息通过/odom消息计算
        obs_raw = self._jrpc_get_obs()
        if obs_raw is None:
            return None

        # 计算雷达观测
        (hazard_lidar, goal_lidar) = self.__calculate_lidar(
            obs_raw["agent_pos"], obs_raw["yaw"]
        )

        # 创建vases_lidar（全零，因为当前没有vases对象）
        vases_lidar = np.zeros(16, dtype=np.float64)

        # 拼接最终观测
        self.last_obs = {
            "accelerometer": np.array(obs_raw["accelerometer"], dtype=np.float64),
            "velocimeter": np.array(obs_raw["velocimeter"], dtype=np.float64),
            "gyro": np.array(obs_raw["gyro"], dtype=np.float64),
            "magnetometer": np.array(obs_raw["magnetometer"], dtype=np.float64),
            "agent_pos": np.array(obs_raw["agent_pos"], dtype=np.float64),
            "yaw": np.float64(obs_raw["yaw"]),
            "goal_lidar": goal_lidar.astype(np.float64),
            "hazards_lidar": hazard_lidar.astype(np.float64),
            "vases_lidar": vases_lidar.astype(np.float64),
        }

        return self.last_obs

    def env_obstacles(self):
        return [
            {"name": "hazards", "items": self.hazard},
            {"name": "vases", "items": self.vase},
            {"name": "goal", "items": self.goal},
        ]

    def agent_pose(self):
        if self.last_obs == {}:
            self.update_obs()
        return self.last_obs["agent_pos"]

    def agent_angle(self):
        if self.last_obs is {}:
            self.update_obs()
        return self.last_obs["yaw"]

    def step(self, action):
        # 发布动作并等待
        if self._jrpc_act(action) is None:
            return (None, None, None, None, True, None)
        time.sleep(self.time_step)

        # 观测新状态
        next_obs = self.update_obs()
        if next_obs is None:
            return (None, None, None, None, True, None)

        # 计算cost和reward
        cost = self.__calculate_cost()

        reward = self.__calculate_reward()
        if cost["cost_hazards"] > 0:
            reward -= 0.1

        # 计算是否到达目标
        goal_dist = np.linalg.norm(self.last_obs["agent_pos"][:2] - self.goal[0, :2])
        if goal_dist < self.goal_size:
            terminated = True
        else:
            terminated = None

        truncated = None
        info = None

        # 更新状态
        self.goal_last_distance = goal_dist if goal_dist > self.goal_size else None

        return (next_obs, reward, cost, terminated, truncated, info)

    def reset(self):
        result = self._jrpc_reset()
        if result is None:
            print("failed to reset environment, return truncate")
            return (None, None, None, None, True, None)
        return self.step([0.0, 0.0])


def main():
    env = RemoteEnvJRPC(8)
    print(env.step([0.2, 0.0]))
    print(env.reset())


if __name__ == "__main__":
    main()
