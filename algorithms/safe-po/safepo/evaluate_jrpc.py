import argparse
import os
import re
import json
import time
from collections import deque

from common.model import ActorVCritic
from utils.config import multi_agent_velocity_map, multi_agent_goal_tasks
from common.single_cm_jrpc import SingleNavAtacomJRPC
import numpy as np
import joblib
import torch

from common.env_jrpc import RemoteEnvJRPC

# 添加项目根目录到Python路径
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


global_save_path: str
cur_env: str
cur_algo: str
index_counter: int = 0

record_data: dict = {
    "seed": "string",
    "hazards": [],
    "goal": [],
    "vases": [],
    "agent_state": [],
}

# record_data["hazard"] = [
#     [1.2665247455083704, -0.011676145860331388, 0.02],
#     [0.5050650567290091, 0.8595964916385668, 0.02],
# ]
# record_data["goal"] = [[-0.2859023586470627, -0.5706664551365962, 0.16]]
# record_data["vases"] = [[-0.2414969952038617, 1.012934898722183, 0.09996000000000001]]


def flatten_obs(obs_dict):
    # print(obs_dict["goal_lidar"])
    return np.concatenate(
        [
            obs_dict["accelerometer"],
            obs_dict["velocimeter"],
            obs_dict["gyro"],
            obs_dict["magnetometer"],
            obs_dict["goal_lidar"],
            obs_dict["hazards_lidar"],
            obs_dict["vases_lidar"],
        ]
    ).astype(np.float32)


def eval_single_agent(eval_dir, eval_episodes):
    global record_data
    global global_save_path
    global cur_env
    global cur_algo
    global index_counter
    torch.set_num_threads(4)
    device = torch.device("cpu")
    config_path = eval_dir + "/config.json"
    config = json.load(open(config_path, "r"))
    env_id = config["task"] if "task" in config.keys() else config["env_name"]

    env_norms = os.listdir(eval_dir)
    env_norms = [env_norm for env_norm in env_norms if env_norm.endswith(".pkl")]
    max_norms_step = -1
    final_norm_name = ""
    for norms_file in env_norms:
        match = re.match(r"state(\d+)\.pkl", norms_file)
        if match:
            if int(match.group(1)) > max_norms_step:
                max_norms_step = int(match.group(1))
                final_norm_name = norms_file

    model_dir = eval_dir + "/torch_save"
    models = os.listdir(model_dir)
    models = [model for model in models if model.endswith(".pt")]
    max_model_step = -1
    final_model_name = ""
    for model_file in models:
        match = re.match(r"model(\d+)\.pt", model_file)
        if match:
            if int(match.group(1)) > max_model_step:
                max_model_step = int(match.group(1))
                final_model_name = model_file

    model_path = model_dir + "/" + final_model_name
    norm_path = eval_dir + "/" + final_norm_name

    eval_env = RemoteEnvJRPC(hazard_num=8)
    model = ActorVCritic(
        obs_dim=60,
        act_dim=2,
        hidden_sizes=config["hidden_sizes"],
    )
    model.actor.load_state_dict(torch.load(model_path, map_location=device))

    if os.path.exists(norm_path):
        norm = joblib.load(open(norm_path, "rb"))["Normalizer"]
        eval_env.obs_rms = norm

    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    eval_env = SingleNavAtacomJRPC(base_env=eval_env, timestep=0.2)

    obstacles = eval_env.env.env_obstacles()
    for obj in ["hazards", "vases", "goal"]:
        record_data[obj] = next(
            item["items"].tolist() for item in obstacles if item["name"] == obj
        )

    # 进行评估
    for _ in range(eval_episodes):
        eval_done = False
        eval_obs, reward, cost, terminated, truncated, info = eval_env.reset()

        # 获取机器人初始状态
        flat_obs = flatten_obs(eval_obs)
        eval_obs = torch.as_tensor(flat_obs, dtype=torch.float32)
        eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0

        # 执行评估步骤
        while not eval_done:
            with torch.no_grad():
                act, _, _, _ = model.step(eval_obs, deterministic=True)

            # 发送动作
            action = act.detach().squeeze().cpu().numpy()
            # print(f"发送动作: {action}")
            (_, next_obs, reward, cost, terminated, truncated, info) = eval_env.step(
                action
            )
            # 记录智能体状态
            agent_dic = {
                "pose": next_obs["agent_pos"].tolist(),
                "v": next_obs["velocimeter"].tolist(),
                "action": act.tolist()[0],
            }
            record_data["agent_state"].append(agent_dic)
            agent_dic["reward"] = reward
            print(agent_dic)
            eval_obs = torch.as_tensor(flatten_obs(next_obs), dtype=torch.float32)
            eval_rew += reward
            eval_cost += cost["cost_hazards"]
            eval_len += 1
            eval_done = terminated or truncated or eval_len > 1000
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
    # 返回平均评估奖励和成本
    return sum(eval_rew_deque) / len(eval_rew_deque), sum(eval_cost_deque) / len(
        eval_cost_deque
    )


# def eval_multi_agent(eval_dir, eval_episodes):

#     config_path = eval_dir + "/config.json"
#     config = json.load(open(config_path, "r"))

#     env_name = config["env_name"]
#     if env_name in multi_agent_velocity_map.keys():
#         env_info = multi_agent_velocity_map[env_name]
#         agent_conf = env_info["agent_conf"]
#         scenario = env_info["scenario"]
#         eval_env = make_ma_mujoco_env(
#             scenario=scenario,
#             agent_conf=agent_conf,
#             seed=np.random.randint(0, 1000),
#             cfg_train=config,
#         )
#     else:
#         eval_env = make_ma_multi_goal_env(
#             task=env_name,
#             seed=np.random.randint(0, 1000),
#             cfg_train=config,
#         )

#     model_dir = eval_dir + f"/models_seed{config['seed']}"
#     algo = config["algorithm_name"]
#     if algo == "macpo":
#         from multi_agent.macpo import Runner
#     elif algo == "mappo":
#         from multi_agent.mappo import Runner
#     elif algo == "mappolag":
#         from multi_agent.mappolag import Runner
#     elif algo == "happo":
#         from multi_agent.happo import Runner
#     else:
#         raise NotImplementedError
#     torch.set_num_threads(4)
#     runner = Runner(
#         vec_env=eval_env,
#         vec_eval_env=eval_env,
#         config=config,
#         model_dir=model_dir,
#     )
#     return runner.eval(eval_episodes)


def single_runs_eval(eval_dir, eval_episodes):

    # config_path = eval_dir + "/config.json"
    # config = json.load(open(config_path, "r"))
    # env = config["task"] if "task" in config.keys() else config["env_name"]
    # if env in multi_agent_velocity_map.keys() or env in multi_agent_goal_tasks:
    #     reward, cost = eval_multi_agent(eval_dir, eval_episodes)
    # else:
    #     reward, cost = eval_single_agent(eval_dir, eval_episodes)
    env_norms = [file for file in os.listdir(eval_dir) if file.endswith(".pkl")]
    if len(env_norms) == 0:
        reward, cost = (None, None)
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
            seed = sorted(seeds)[-1]
            seed_path = os.path.join(algo_path, seed)
            reward, cost = single_runs_eval(seed_path, eval_episodes)
            if reward is None or cost is None:
                pass
            else:
                rewards.append(reward)
                costs.append(cost)
            # for seed in seeds:
            #     record_data["seed"] = seed
            #     seed_path = os.path.join(algo_path, seed)
            #     reward, cost = single_runs_eval(seed_path, eval_episodes)
            #     rewards.append(reward)
            #     costs.append(cost)
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
