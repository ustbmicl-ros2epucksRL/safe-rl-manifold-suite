import argparse
import os
import json
import time
from collections import deque
from common.env import (
    make_sa_mujoco_env,
    make_ma_mujoco_env,
    make_ma_multi_goal_env,
    make_sa_isaac_env,
)
from common.model import ActorVCritic
from utils.config import multi_agent_velocity_map, multi_agent_goal_tasks
from common.single_cm import SingleNavAtacom
import numpy as np
import joblib
import torch

# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32MultiArray
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


global_save_path: str
cur_env: str
cur_algo: str
index_counter: int = 0
# HTTP服务器配置
http_server_address = "ros2_webots_czz"
http_server_port = 8082

record_data: dict = {
    "seed": "string",
    "hazard": [],
    "goal": [],
    "vases": [],
    "agent_state": [],
}

record_data["hazard"] = [
    [1.2665247455083704, -0.011676145860331388, 0.02],
    [0.5050650567290091, 0.8595964916385668, 0.02],
]
record_data["goal"] = [[-0.2859023586470627, -0.5706664551365962, 0.16]]
record_data["vases"] = [[-0.2414969952038617, 1.012934898722183, 0.09996000000000001]]

# 定义ROS节点类，用于发布和订阅机器人状态
# class RLAgentNode(Node):
#     def __init__(self):
#         super().__init__('rl_agent_node')
#         self.action_pub = self.create_publisher(Float32MultiArray, '/robot_action', 10)
#         self.obs_sub = self.create_subscription(Float32MultiArray, '/robot_obs', self.obs_callback, 10)
#         self.latest_obs = None

#     def obs_callback(self, msg):
#         self.latest_obs = np.array(msg.data)


def calculate_reward(agent_pos, goal_pos, last_dist_goal, cost):
    """Determine reward depending on the agent and tasks."""
    # pylint: disable=no-member
    reward = 0.0
    dist_goal = np.sqrt(
        (agent_pos[0] - goal_pos[0]) ** 2 + (agent_pos[1] - goal_pos[1]) ** 2
    )
    reward += last_dist_goal - dist_goal  #
    last_dist_goal = dist_goal

    if dist_goal < 0.3:  # 到达目标
        reward += 1
    if cost > 0:
        reward -= 0.1  # 发生碰撞

    return reward, last_dist_goal


def cal_cost(agent_pos, hazards_pos):
    """Contacts Processing."""
    cost = 0.0
    for h_pos in hazards_pos:
        h_dist = np.sqrt(
            (agent_pos[0] - h_pos[0]) ** 2 + (agent_pos[1] - h_pos[1]) ** 2
        )
        # pylint: disable=no-member
        if h_dist <= 0.2:
            cost += 0.2 - h_dist
    return cost


def _obs_lidar_pseudo(positions: np.ndarray, agent_pos, agent_mat) -> np.ndarray:
    positions = np.array(positions, ndmin=2)
    obs = np.zeros(16)
    for pos in positions:
        pos = np.asarray(pos)
        if pos.shape == (3,):
            pos = pos[:2]  # Truncate Z coordinate
        # pylint: disable-next=invalid-name
        z = complex(
            *_ego_xy(pos, agent_pos, agent_mat)
        )  # X, Y as real, imaginary components
        dist = np.abs(z)
        angle = np.angle(z) % (np.pi * 2)
        bin_size = (np.pi * 2) / 16
        bin = int(angle / bin_size)  # pylint: disable=redefined-builtin
        bin_angle = bin_size * bin
        sensor = max(0, 3 - dist) / 3
        obs[bin] = max(obs[bin], sensor)
        # Aliasing
        alias = (angle - bin_angle) / bin_size
        assert (
            0 <= alias <= 1
        ), f"bad alias {alias}, dist {dist}, angle {angle}, bin {bin}"
        bin_plus = (bin + 1) % 16
        bin_minus = (bin - 1) % 16
        obs[bin_plus] = max(obs[bin_plus], alias * sensor)
        obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
    return obs


def mat(orientation):
    return np.array(
        [
            [orientation[0], orientation[1], 0],
            [-orientation[1], orientation[0], 0],
            [0, 0, 1],
        ]
    )


def _ego_xy(pos: np.ndarray, agent_pos, agent_mat) -> np.ndarray:
    """Return the egocentric XY vector to a position from the agent."""
    assert pos.shape == (2,), f"Bad pos {pos}"
    agent_3vec = agent_pos
    pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
    world_3vec = pos_3vec - agent_3vec
    return np.matmul(world_3vec, agent_mat)[:2]


def eval_single_agent(eval_dir, eval_episodes):
    global record_data
    global global_save_path
    global cur_env
    global cur_algo
    global index_counter

    # 初始化ROS节点
    # rclpy.init()
    # agent_node = RLAgentNode()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    config_path = eval_dir + "/config.json"
    config = json.load(open(config_path, "r"))
    env_id = config["task"] if "task" in config.keys() else config["env_name"]
    env_norms = os.listdir(eval_dir)
    env_norms = [env_norm for env_norm in env_norms if env_norm.endswith(".pkl")]
    final_norm_name = sorted(env_norms)[-1]

    model_dir = eval_dir + "/torch_save"
    models = os.listdir(model_dir)
    models = [model for model in models if model.endswith(".pt")]
    final_model_name = sorted(models)[-1]
    model_path = model_dir + "/" + final_model_name
    norm_path = eval_dir + "/" + final_norm_name

    eval_env, obs_space, act_space = make_sa_mujoco_env(
        num_envs=config["num_envs"], env_id=env_id, seed=None
    )
    model = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    )
    model.actor.load_state_dict(torch.load(model_path, map_location=device))

    if os.path.exists(norm_path):
        norm = joblib.load(open(norm_path, "rb"))["Normalizer"]
        eval_env.obs_rms = norm

    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    eval_env = SingleNavAtacom(base_env=eval_env, timestep=0.002)

    # HTTP连接
    try:
        # 使用HTTP客户端
        jrpc_client = RobotControlHTTPClient(http_server_address, http_server_port)
        if not jrpc_client.connect():
            raise Exception("无法连接到HTTP服务器")

        # 进行评估
        for _ in range(eval_episodes):
            eval_done = False
            eval_obs, _ = eval_env.reset()

            # 获取机器人初始状态
            print("获取机器人初始状态...")
            response = jrpc_client.get_initial_status()
            print(f"收到初始状态: {response}")
            last_timestep = response["data"]["timestep"]

            robot_data = response["data"]
            agent_mat = mat(robot_data["agent_orientation"])
            obs_lidar = (
                _obs_lidar_pseudo(
                    record_data["goal"], robot_data["agent_pos"], agent_mat
                )
                + _obs_lidar_pseudo(
                    record_data["hazard"], robot_data["agent_pos"], agent_mat
                )
                + _obs_lidar_pseudo(
                    record_data["vases"], robot_data["agent_pos"], agent_mat
                )
            )
            eval_obs = (
                robot_data["accelerometer"]
                + robot_data["velocimeter"]
                + robot_data["gyro"]
                + robot_data["magnetometer"]
                + obs_lidar
            )

            eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32)
            eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
            last_dist_goal = None

            # 执行评估步骤
            while not eval_done:
                with torch.no_grad():
                    act, _, _, _ = model.step(eval_obs, deterministic=True)

                # 发送动作
                action = act.detach().squeeze().cpu().numpy()
                print(f"发送动作: {action}")
                jrpc_client.send_action(action, last_timestep)

                # 获取当前状态
                print("获取机器人当前状态...")
                response = jrpc_client.get_current_status()

                # 等待新的时间步
                while response["data"]["timestep"] == last_timestep:
                    time.sleep(0.001)  # 短暂等待
                    response = jrpc_client.get_current_status()

                last_timestep = response["data"]["timestep"]
                print(f"收到状态更新: timestep={last_timestep}")

                robot_data = response["data"]
                agent_mat = mat(robot_data["agent_orientation"])
                obs_lidar = (
                    _obs_lidar_pseudo(
                        record_data["goal"], robot_data["agent_pos"], agent_mat
                    )
                    + _obs_lidar_pseudo(
                        record_data["hazard"], robot_data["agent_pos"], agent_mat
                    )
                    + _obs_lidar_pseudo(
                        record_data["vases"], robot_data["agent_pos"], agent_mat
                    )
                )
                eval_obs = (
                    robot_data["accelerometer"]
                    + robot_data["velocimeter"]
                    + robot_data["gyro"]
                    + robot_data["magnetometer"]
                    + obs_lidar
                )

                # 计算reward、cost
                cost = cal_cost(robot_data["agent_pos"], record_data["hazard"])
                reward, last_dist_goal = calculate_reward(
                    robot_data["agent_pos"], record_data["goal"], last_dist_goal
                )

                # 记录智能体状态
                record_data["agent_state"].append(
                    {
                        "pose": robot_data["agent_pos"],
                        "v": robot_data["velocimeter"],
                        "action": act.tolist()[0],
                    }
                )
                eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32)
                eval_rew += reward
                eval_cost += cost
                eval_len += 1
                eval_done = last_dist_goal < 0.3 or eval_len > 1000
            # 将评估结果添加到队列中
            eval_rew_deque.append(eval_rew)
            eval_cost_deque.append(eval_cost)
            eval_len_deque.append(eval_len)

            # 保存评估数据
            file_name = os.path.join(
                global_save_path, f"{cur_env}_{cur_algo}_{index_counter}.json"
            )
            with open(file_name, "w") as file:
                json.dump(record_data, file)

            # 更新索引计数器
            index_counter += 1

        # 关闭HTTP连接
        jrpc_client.disconnect()

    except Exception as e:
        print(f"HTTP连接/通信错误: {e}")
        if "jrpc_client" in locals():
            jrpc_client.disconnect()

    # 返回平均评估奖励和成本
    return sum(eval_rew_deque) / len(eval_rew_deque), sum(eval_cost_deque) / len(
        eval_cost_deque
    )


def eval_multi_agent(eval_dir, eval_episodes):

    config_path = eval_dir + "/config.json"
    config = json.load(open(config_path, "r"))

    env_name = config["env_name"]
    if env_name in multi_agent_velocity_map.keys():
        env_info = multi_agent_velocity_map[env_name]
        agent_conf = env_info["agent_conf"]
        scenario = env_info["scenario"]
        eval_env = make_ma_mujoco_env(
            scenario=scenario,
            agent_conf=agent_conf,
            seed=np.random.randint(0, 1000),
            cfg_train=config,
        )
    else:
        eval_env = make_ma_multi_goal_env(
            task=env_name,
            seed=np.random.randint(0, 1000),
            cfg_train=config,
        )

    model_dir = eval_dir + f"/models_seed{config['seed']}"
    algo = config["algorithm_name"]
    if algo == "macpo":
        from multi_agent.macpo import Runner
    elif algo == "mappo":
        from multi_agent.mappo import Runner
    elif algo == "mappolag":
        from multi_agent.mappolag import Runner
    elif algo == "happo":
        from multi_agent.happo import Runner
    else:
        raise NotImplementedError
    torch.set_num_threads(4)
    runner = Runner(
        vec_env=eval_env,
        vec_eval_env=eval_env,
        config=config,
        model_dir=model_dir,
    )
    return runner.eval(eval_episodes)


def single_runs_eval(eval_dir, eval_episodes):

    config_path = eval_dir + "/config.json"
    config = json.load(open(config_path, "r"))
    env = config["task"] if "task" in config.keys() else config["env_name"]
    if env in multi_agent_velocity_map.keys() or env in multi_agent_goal_tasks:
        reward, cost = eval_multi_agent(eval_dir, eval_episodes)
    else:
        reward, cost = eval_single_agent(eval_dir, eval_episodes)

    return reward, cost


def benchmark_eval():
    # 全局变量声明
    global global_save_path
    global record_data
    global cur_algo
    global cur_env

    # 创建解析器
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir", type=str, default="", help="the directory of the evaluation"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="the number of episodes to evaluate",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="the directory to save the evaluation result",
    )
    parser.add_argument(
        "--algo", type=str, default=None, help="the algorithm to evaluate"
    )
    parser.add_argument(
        "--task", type=str, default=None, help="the environment to evaluate"
    )

    # 解析参数
    args = parser.parse_args()

    # 获取参数值
    benchmark_dir = args.benchmark_dir
    eval_episodes = args.eval_episodes
    eval_algo = args.algo
    eval_task = args.task
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        # 若未指定保存目录，则使用基准目录的results路径
        save_dir = benchmark_dir.replace("runs", "results")
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
    global_save_path = save_dir

    # 列出基准目录下的所有环境
    envs = os.listdir(benchmark_dir)
    for env in envs:
        if env.find(eval_task) < 0:
            continue
        env_path = os.path.join(benchmark_dir, env)
        # 列出当前环境下的所有算法
        algos = os.listdir(env_path)
        cur_env = env
        for algo in algos:
            # 跳过不包含"trpo"的算法
            if algo.find(eval_algo) < 0:
                continue
            print(f"Start evaluating {algo} in {env}")
            cur_algo = algo
            algo_path = os.path.join(env_path, algo)
            # 列出当前算法下的所有种子
            seeds = os.listdir(algo_path)
            rewards, costs = [], []
            for seed in seeds:
                record_data["seed"] = seed
                seed_path = os.path.join(algo_path, seed)
                reward, cost = single_runs_eval(seed_path, eval_episodes)
                rewards.append(reward)
                costs.append(cost)
            # 打开输出文件
            output_file = open(f"{save_dir}/eval_result.txt", "a")
            # two wise after point
            reward_mean = round(np.mean(rewards), 2)
            reward_std = round(np.std(rewards), 2)
            cost_mean = round(np.mean(costs), 2)
            cost_std = round(np.std(costs), 2)
            print(
                f"After {eval_episodes} episodes evaluation, the {algo} in {env} evaluation reward: {reward_mean}±{reward_std}, cost: {cost_mean}±{cost_std}, the reuslt is saved in {save_dir}/eval_result.txt"
            )
            output_file.write(
                f"After {eval_episodes} episodes evaluation, the {algo} in {env} evaluation reward: {reward_mean}±{reward_std}, cost: {cost_mean}±{cost_std} \n"
            )


if __name__ == "__main__":
    benchmark_eval()
