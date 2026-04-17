# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import shutil
import sys
import tempfile

# 无显示环境（Docker/SSH）下用 xvfb-run 执行，避免 GLFW/X11 报错
# 用法: python evaluate.py --headless ...  或  DISPLAY= python evaluate.py ...
_use_headless = "--headless" in sys.argv
if _use_headless:
    sys.argv.remove("--headless")
if _use_headless or not os.environ.get("DISPLAY", "").strip():
    try:
        import subprocess
        rc = subprocess.call(["xvfb-run", "-a", sys.executable] + sys.argv)
        sys.exit(rc)
    except FileNotFoundError:
        if _use_headless:
            print("Warning: xvfb-run not found, install with: apt install xvfb")

import json
from collections import deque
from safepo.common.env import make_sa_mujoco_env, make_ma_mujoco_env, make_ma_multi_goal_env, make_sa_isaac_env
from safepo.common.model import ActorVCritic
from safepo.utils.config import multi_agent_velocity_map, multi_agent_goal_tasks, multi_agent_formation_tasks
from safepo.common.single_cm import SingleNavAtacom
import numpy as np
import joblib
import torch

global_save_path : str
cur_env : str
cur_algo : str
index_counter: int = 0 

record_data: dict = {
    "seed": "string",
    "hazard": [],
    "goal": [],
    "vases": [],
    "agent_state": [],
    # 评估相关统计
    "noise_std": 0.0,
    "episode_return": 0.0,
    "episode_cost": 0.0,
    "success": False,
}

def eval_single_agent(eval_dir, eval_episodes):
    # 全局变量
    global record_data
    global global_save_path
    global cur_env
    global cur_algo
    global index_counter

    # 设置torch线程数
    torch.set_num_threads(4)

    # 加载配置文件
    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))

    # 获取环境ID
    env_id = config['task'] if 'task' in config.keys() else config['env_name']

    # 获取环境规范文件列表
    env_norms = os.listdir(eval_dir)
    env_norms = [env_norm for env_norm in env_norms if env_norm.endswith('.pkl')]

    # 获取最新的环境规范文件名
    final_norm_name = sorted(env_norms)[-1]

    # 获取模型目录
    model_dir = eval_dir + '/torch_save'

    # 获取模型文件列表
    models = os.listdir(model_dir)
    models = [model for model in models if model.endswith('.pt')]

    # 获取最新的模型文件名
    final_model_name = sorted(models)[-1]

    # 获取模型路径
    model_path = model_dir + '/' + final_model_name

    # 获取环境规范路径
    norm_path = eval_dir + '/' + final_norm_name

    # 创建环境
    eval_env, obs_space, act_space = make_sa_mujoco_env(num_envs=config['num_envs'], env_id=env_id, seed=None)

    # 创建模型
    model = ActorVCritic(
        # 模型参数
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            hidden_sizes=config['hidden_sizes'],
        )

    # 加载模型参数
    model.actor.load_state_dict(torch.load(model_path))

    # 加载环境规范
    if os.path.exists(norm_path):
        norm = joblib.load(open(norm_path, 'rb'))['Normalizer']
        eval_env.obs_rms = norm

    # 初始化评估结果队列
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    episode_results = []  # 每个 episode 的 (noise_std, reward, cost, success)

    # 创建评估环境（包装成 SingleNavAtacom），并记录噪声大小
    # eval_env = SingleNavAtacom(base_env=eval_env, timestep=0.002)
    eval_env = SingleNavAtacom(base_env=eval_env, timestep=0.002, noise_std=7.0)
    record_data["noise_std"] = float(getattr(eval_env, "noise_std", 0.0))

    # 进行评估
    for _ in range(eval_episodes):
        eval_done = False
        eval_obs, _ = eval_env.reset()
        eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32)
        eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0

        # 记录环境障碍物信息
        for obs in eval_env.base_env.task._obstacles:
            if obs.name == "hazards":
                record_data['hazard'] = [pos.tolist() for pos in obs.pos]
            if obs.name == "goal":
                record_data['goal'] = [obs.pos.tolist()]
            if obs.name == "vases":
                record_data['vases'] = [pos.tolist() for pos in obs.pos]

        # 执行评估步骤
        while not eval_done:
            with torch.no_grad():
                act, _, _, _ = model.step(
                    eval_obs, deterministic=True
                )

            # 记录智能体状态
            record_data['agent_state'].append(
                {
                    "pose": eval_env.base_env.task.agent.pos.tolist(),
                    "v":    eval_env.base_env.task.agent.vel.tolist(),
                    "action": act.tolist()[0]
                }
            )

            # 执行环境步骤（SingleNavAtacom.step 返回 p, next_obs, reward, cost, terminated, truncated, info）
            _, eval_obs, reward, cost, terminated, truncated, info = eval_env.step(
                act.detach().squeeze().cpu().numpy()
            )
            eval_obs = torch.as_tensor(
                eval_obs, dtype=torch.float32
            )
            eval_rew += reward[0]
            eval_cost += cost[0]
            eval_len += 1
            eval_done = terminated[0] or truncated[0]

        # 判断是否成功：到达目标且未发生碰撞
        goal_met = info.get("goal_met", False) if isinstance(info, dict) else False
        if hasattr(goal_met, "__len__") and not isinstance(goal_met, (str, dict)) and len(goal_met) > 0:
            goal_met = goal_met[0]
        success = bool(goal_met) and (float(eval_cost) == 0)

        # 将评估结果添加到队列中
        eval_rew_deque.append(eval_rew)
        eval_cost_deque.append(eval_cost)
        eval_len_deque.append(eval_len)

        # 在记录文件中写入本次 episode 的总回报、总 cost 与是否成功
        record_data["episode_return"] = float(eval_rew)
        record_data["episode_cost"] = float(eval_cost)
        record_data["success"] = success

        # 收集每个 episode 的详细结果（用于写入 eval_result.txt）
        noise_std = float(getattr(eval_env, "noise_std", 0.0))
        episode_results.append((noise_std, float(eval_rew), float(eval_cost), success))

        # 保存评估数据
        file_name = os.path.join(global_save_path, f"{cur_env}_{cur_algo}_{index_counter}.json")
        with open(file_name,"w") as file:
                json.dump(record_data, file)

        # 更新索引计数器
        index_counter += 1


    # 返回平均评估奖励、成本、成功率，以及每个 episode 的 (noise, reward, cost, success) 列表
    avg_reward = sum(eval_rew_deque) / len(eval_rew_deque)
    avg_cost = sum(eval_cost_deque) / len(eval_cost_deque)
    success_count = sum(1 for r in episode_results if r[3])
    success_rate = success_count / len(episode_results) if episode_results else 0.0
    return avg_reward, avg_cost, success_rate, success_count, episode_results


def eval_multi_agent(eval_dir, eval_episodes):

    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))

    env_name = config['env_name']
    if env_name in multi_agent_velocity_map.keys():
        env_info = multi_agent_velocity_map[env_name]
        agent_conf = env_info['agent_conf']
        scenario = env_info['scenario']
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
    # 评估时使用临时 log_dir，避免覆盖训练目录下的 progress.csv 和 config.json
    eval_config = config.copy()
    eval_config["log_dir"] = tempfile.mkdtemp(prefix="eval_")
    eval_config["use_tensorboard"] = False  # 评估时无需 TensorBoard
    algo = config['algorithm_name']
    if algo == 'macpo':
        from safepo.multi_agent.macpo import Runner
    elif algo == 'mappo':
        from safepo.multi_agent.mappo import Runner
    elif algo == 'mappo_rmp':
        from safepo.multi_agent.mappo_rmp import Runner
    elif algo == 'mappolag':
        from safepo.multi_agent.mappolag import Runner
    elif algo == 'happo':
        from safepo.multi_agent.happo import Runner
    else:
        raise NotImplementedError
    torch.set_num_threads(4)
    runner = Runner(
        vec_env=eval_env,
        vec_eval_env=eval_env,
        config=eval_config,
        model_dir=model_dir,
    )
    try:
        result = runner.eval(eval_episodes)
        if isinstance(result, tuple) and len(result) == 3:
            reward, cost, success_count = result
            success_rate = success_count / eval_episodes if eval_episodes > 0 else 0.0
            return reward, cost, success_rate, success_count
        else:
            reward, cost = result[0], result[1]
            return reward, cost, None, None
    finally:
        if os.path.exists(eval_config["log_dir"]):
            shutil.rmtree(eval_config["log_dir"], ignore_errors=True)


def single_runs_eval(eval_dir, eval_episodes):

    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))
    env = config['task'] if 'task' in config.keys() else config['env_name']
    if env in multi_agent_velocity_map.keys() or env in multi_agent_goal_tasks or env in multi_agent_formation_tasks:
        reward, cost, success_rate, success_count = eval_multi_agent(eval_dir, eval_episodes)
        return reward, cost, success_rate, success_count, None  # 多智能体暂无逐 episode 详细结果
    else:
        reward, cost, success_rate, success_count, episode_results = eval_single_agent(eval_dir, eval_episodes)
        return reward, cost, success_rate, success_count, episode_results

def benchmark_eval():
    # 全局变量声明
    global global_save_path
    global record_data
    global cur_algo 
    global cur_env
    
    # 创建解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=str, default='', help="父目录，需包含任务子目录(如 SafetyPointMultiFormationGoal0-v0)，任务目录下为算法(mappo/mappo_rmp)")
    parser.add_argument("--eval-episodes", type=int, default=20, help="the number of episodes to evaluate")
    parser.add_argument("--save-dir", type=str, default=None, help="the directory to save the evaluation result")

    # 解析参数
    args = parser.parse_args()

    # 获取参数值
    benchmark_dir = args.benchmark_dir
    eval_episodes = args.eval_episodes
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        # 若未指定保存目录，则使用基准目录的results路径
        save_dir = benchmark_dir.replace('runs', 'results')
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
    global_save_path = save_dir
    
    # 列出基准目录下的所有环境
    eval_result_path = os.path.join(save_dir, "eval_result.txt")
    first_write = True  # 首次写入时写表头并清空文件
    envs = os.listdir(benchmark_dir)
    for env in envs:
        if env.find("Goal") < 0:
            continue
        env_path = os.path.join(benchmark_dir, env)
        algos = os.listdir(env_path)
        cur_env = env
        for algo in algos:
            # 跳过不包含"ppo_cm"、"mappo"或"mappo_rmp"的算法
            if algo.find("mappo_rmp") < 0 and algo.find("mappo") < 0:
                continue
            print(f"Start evaluating {algo} in {env}")
            cur_algo = algo
            algo_path = os.path.join(env_path, algo)
            # 列出当前算法下的所有种子
            seeds = os.listdir(algo_path)
            rewards, costs, success_rates, success_counts = [], [], [], []
            for seed in seeds:
                seed_path = os.path.join(algo_path, seed)
                if not os.path.isdir(seed_path):
                    continue
                # 检查是否有可用的模型文件（跳过空或未完成的训练目录）
                config_path = os.path.join(seed_path, "config.json")
                if not os.path.exists(config_path):
                    continue
                seed_config = json.load(open(config_path, "r"))
                seed_env = seed_config.get("task") or seed_config.get("env_name", "")
                if seed_env in multi_agent_velocity_map.keys() or seed_env in multi_agent_goal_tasks or seed_env in multi_agent_formation_tasks:
                    # 多智能体：检查 models_seed{seed}/actor_agent0.pt
                    model_dir = os.path.join(seed_path, f"models_seed{seed_config.get('seed', 0)}")
                    if not os.path.exists(os.path.join(model_dir, "actor_agent0.pt")):
                        print(f"  Skip {seed}: no model found in {model_dir}")
                        continue
                else:
                    # 单智能体：检查 torch_save/*.pt
                    torch_save_dir = os.path.join(seed_path, "torch_save")
                    if not os.path.isdir(torch_save_dir) or not any(f.endswith(".pt") for f in os.listdir(torch_save_dir)):
                        print(f"  Skip {seed}: no model found in torch_save/")
                        continue
                record_data['seed'] = seed
                reward, cost, success_rate, success_count, episode_results = single_runs_eval(
                    seed_path, eval_episodes
                )
                rewards.append(reward)
                costs.append(cost)
                if success_rate is not None and success_count is not None:
                    success_rates.append(success_rate)
                    success_counts.append(success_count)
                # 将每个 episode 的详细结果写入 eval_result.txt
                if episode_results is not None:
                    mode = 'w' if first_write else 'a'
                    with open(eval_result_path, mode, encoding='utf-8') as f:
                        if first_write:
                            f.write("algorithm\ttask\tseed\tepisode\tnoise\treward\tcost\tsuccess\tsuccess_rate\n")
                            first_write = False
                        for ep_idx, (noise, ep_reward, ep_cost, success) in enumerate(episode_results, start=1):
                            ep_success = 1 if success else 0
                            f.write(f"{algo}\t{env}\t{seed}\t{ep_idx}\t{noise}\t{round(ep_reward, 2)}\t{round(ep_cost, 2)}\t{ep_success}\t-\n")
                    # 每个 seed 写入一行成功率汇总
                    if success_rate is not None and success_count is not None:
                        with open(eval_result_path, 'a', encoding='utf-8') as f:
                            f.write(f"{algo}\t{env}\t{seed}\tSUMMARY\t-\t-\t-\t{success_count}/{eval_episodes}\t{success_rate:.2%}\n")
                else:
                    # 多智能体：写入汇总行（含成功率）
                    mode = 'w' if first_write else 'a'
                    with open(eval_result_path, mode, encoding='utf-8') as f:
                        if first_write:
                            f.write("algorithm\ttask\tseed\tepisode\tnoise\treward\tcost\tsuccess\tsuccess_rate\n")
                            first_write = False
                        sr_str = f"{success_count}/{eval_episodes}" if success_count is not None else "-"
                        rate_str = f"{success_rate:.2%}" if success_rate is not None else "-"
                        f.write(f"{algo}\t{env}\t{seed}\tSUMMARY\t-\t{round(reward, 2)}\t{round(cost, 2)}\t{sr_str}\t{rate_str}\n")
            # 打印汇总信息
            reward_mean = round(np.mean(rewards), 2)
            reward_std = round(np.std(rewards), 2)
            cost_mean = round(np.mean(costs), 2)
            cost_std = round(np.std(costs), 2)
            msg = f"After {eval_episodes} episodes evaluation, the {algo} in {env} evaluation reward: {reward_mean}±{reward_std}, cost: {cost_mean}±{cost_std}"
            if success_rates:
                sr_mean = np.mean(success_rates)
                sr_std = np.std(success_rates)
                total_success = sum(success_counts)
                total_eps = len(seeds) * eval_episodes
                msg += f", success_rate: {total_success}/{total_eps} ({sr_mean:.2%}±{sr_std:.2%})"
            msg += f", the result is saved in {eval_result_path}"
            print(msg)
            
if __name__ == '__main__':
    benchmark_eval()
