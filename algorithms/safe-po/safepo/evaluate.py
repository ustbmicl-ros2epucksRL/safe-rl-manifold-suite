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
from datetime import datetime
from distutils.util import strtobool
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

import copy
import json
from collections import deque
from safepo.common.env import (
    make_sa_mujoco_env,
    make_ma_mujoco_env,
    make_ma_multi_goal_env,
    make_formation_nav_env,
    make_sa_isaac_env,
)
from safepo.common.model import ActorVCritic
from safepo.utils.config import (
    multi_agent_velocity_map,
    multi_agent_goal_tasks,
    multi_agent_formation_tasks,
)
from safepo.common.single_cm import SingleNavAtacom
import numpy as np
import joblib
import torch

global_save_path : str
cur_env : str
cur_algo : str
index_counter: int = 0 

def _actor_obs_dim_from_checkpoint(model_dir: str):
    """Observation dimension stored in ``actor_agent0.pt`` (``base.feature_norm``)."""
    path = os.path.join(model_dir, "actor_agent0.pt")
    if not os.path.exists(path):
        return None
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)  # PyTorch >= 2.4
    except TypeError:
        sd = torch.load(path, map_location="cpu")
    w = sd.get("base.feature_norm.weight")
    if w is not None:
        return int(w.shape[0])
    return None


def _infer_num_agents_formation(task_id: str, cfg_train: dict, target_obs_dim: int):
    """Brute-force ``num_agents`` until ``MultiFormationNavEnv.obs_size`` matches the checkpoint."""
    cfg = copy.deepcopy(cfg_train)
    cfg["n_rollout_threads"] = 1
    cfg["n_eval_rollout_threads"] = 1
    for n in range(2, 17):
        try:
            env = make_formation_nav_env(task_id, 0, n, cfg, render_mode=None)
            d = env.envs[0].obs_size
            env.close()
            if d == target_obs_dim:
                return n
        except Exception:
            continue
    return None


def _to_float_or_none(v):
    try:
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        if hasattr(v, "item"):
            v = v.item()
        return float(v)
    except (TypeError, ValueError):
        return None


def _extract_formation_completion_score(info_obj, error_threshold: float):
    """Extract per-step formation completion score in [0, 1].

    Priority:
    1) direct completion score-like key (e.g. formation_completion)
    2) convert formation error to score: score = max(0, 1 - error / threshold)
    3) boolean completion flag -> 1/0
    """
    info = info_obj
    if isinstance(info, (list, tuple)) and len(info) > 0:
        info = info[0]
    if not isinstance(info, dict):
        return None

    direct_keys = (
        "formation_completion",
        "formation_completion_ratio",
        "formation_score",
    )
    for key in direct_keys:
        if key in info:
            score = _to_float_or_none(info[key])
            if score is not None:
                return float(np.clip(score, 0.0, 1.0))

    error_keys = (
        "formation_error",
        "formation_distance_error",
        "formation_shape_error",
        "formation_completion_error",
    )
    for key in error_keys:
        if key in info:
            err = _to_float_or_none(info[key])
            if err is not None and error_threshold > 0:
                return float(np.clip(1.0 - err / error_threshold, 0.0, 1.0))

    flag_keys = ("formation_completed", "formation_complete", "formation_met")
    for key in flag_keys:
        if key in info:
            return 1.0 if bool(info[key]) else 0.0
    return None


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


def eval_multi_agent(eval_dir, eval_episodes, config_overrides=None):
    """Evaluate multi-agent policy. ``config_overrides`` is merged into the run's ``config.json`` (e.g. ``num_agents``)."""
    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))
    if config_overrides:
        config = {**config, **config_overrides}

    model_dir = eval_dir + f"/models_seed{config['seed']}"
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
    elif env_name in multi_agent_formation_tasks:
        num_agents = int(config.get('num_agents', 2))
        target_dim = _actor_obs_dim_from_checkpoint(model_dir)
        if target_dim is not None:
            inferred = _infer_num_agents_formation(env_name, config, target_dim)
            if inferred is not None:
                if inferred != num_agents:
                    print(
                        f"[evaluate] Checkpoint expects obs_dim={target_dim} (formation num_agents={inferred}); "
                        f"config had num_agents={num_agents}. Using num_agents={inferred}.",
                    )
                num_agents = inferred
            else:
                print(
                    f"[evaluate] Warning: could not infer num_agents for obs_dim={target_dim}; "
                    f"using num_agents={num_agents}. Load may fail if mismatch.",
                )
        eval_env = make_formation_nav_env(
            task=env_name,
            seed=np.random.randint(0, 1000),
            num_agents=num_agents,
            cfg_train=config,
            render_mode=None,
        )
    else:
        eval_env = make_ma_multi_goal_env(
            task=env_name,
            seed=np.random.randint(0, 1000),
            cfg_train=config,
        )

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
    elif algo == 'cbf_mappo':
        from safepo.multi_agent.cbf_mappo import Runner
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
        # Formation tasks: add a dedicated formation-completion metric.
        if env_name in multi_agent_formation_tasks:
            eval_episode = 0
            eval_episode_rewards = []
            eval_episode_costs = []
            eval_episode_successes = []
            eval_episode_formation_scores = []  # 1.0 if goal_met else 0.0

            one_episode_rewards = torch.zeros(
                1, eval_config["n_eval_rollout_threads"], device=eval_config["device"],
            )
            one_episode_costs = torch.zeros(
                1, eval_config["n_eval_rollout_threads"], device=eval_config["device"],
            )

            eval_obs, _, _ = runner.eval_envs.reset()
            eval_rnn_states = torch.zeros(
                eval_config["n_eval_rollout_threads"],
                runner.num_agents,
                eval_config["recurrent_N"],
                eval_config["hidden_size"],
                device=eval_config["device"],
            )
            eval_masks = torch.ones(
                eval_config["n_eval_rollout_threads"], runner.num_agents, 1, device=eval_config["device"],
            )

            while eval_episode < eval_episodes:
                with torch.no_grad():
                    eval_actions_collector = []
                    for agent_id in range(runner.num_agents):
                        runner.trainer[agent_id].prep_rollout()
                        if 'Frank' in eval_config['env_name']:
                            obs_to_eval = eval_obs[agent_id]
                        else:
                            obs_to_eval = eval_obs[:, agent_id]
                        eval_actions, temp_rnn_state = runner.policy[agent_id].act(
                            obs_to_eval,
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = temp_rnn_state
                        eval_actions_collector.append(eval_actions.detach())

                    if eval_config["env_name"] == "Safety9|8HumanoidVelocity-v0":
                        zeros = torch.zeros(eval_actions_collector[-1].shape[0], 1, device=eval_actions_collector[-1].device)
                        eval_actions_collector[-1] = torch.cat((eval_actions_collector[-1], zeros), dim=1)

                    # Some runners (e.g. plain MAPPO) do not have `rmp_corrector`.
                    if hasattr(runner, "rmp_corrector") and getattr(runner.rmp_corrector, "rmp_enabled", False):
                        try:
                            agent_positions, agent_velocities = runner.rmp_corrector.get_agent_positions_from_env(
                                runner.eval_envs, eval_config["n_eval_rollout_threads"],
                            )
                            eval_actions_collector = runner.rmp_corrector.apply_correction(
                                eval_actions_collector, agent_positions, agent_velocities,
                            )
                        except Exception:
                            pass

                    eval_obs, _, eval_rewards, eval_costs, eval_dones, eval_infos, _ = runner.eval_envs.step(
                        eval_actions_collector,
                    )
                reward_env = torch.mean(eval_rewards, dim=1).flatten()
                cost_env = torch.mean(eval_costs, dim=1).flatten()
                one_episode_rewards += reward_env
                one_episode_costs += cost_env

                eval_dones_env = torch.all(eval_dones, dim=1)
                eval_rnn_states[eval_dones_env == True] = torch.zeros(
                    (eval_dones_env == True).sum(),
                    runner.num_agents,
                    eval_config["recurrent_N"],
                    eval_config["hidden_size"],
                    device=eval_config["device"],
                )
                eval_masks = torch.ones(
                    eval_config["n_eval_rollout_threads"], runner.num_agents, 1, device=eval_config["device"],
                )
                eval_masks[eval_dones_env == True] = torch.zeros(
                    (eval_dones_env == True).sum(), runner.num_agents, 1, device=eval_config["device"],
                )

                for eval_i in range(eval_config["n_eval_rollout_threads"]):
                    if not bool(eval_dones_env[eval_i]):
                        continue
                    eval_episode += 1
                    ep_reward = one_episode_rewards[:, eval_i].mean().item()
                    ep_cost = one_episode_costs[:, eval_i].mean().item()
                    goal_met = False
                    if eval_infos is not None and eval_i < len(eval_infos):
                        info = eval_infos[eval_i]
                        if isinstance(info, (list, tuple)) and len(info) > 0:
                            info = info[0]
                        goal_met = info.get("goal_met", False) if isinstance(info, dict) else False

                    ep_formation = 1.0 if goal_met else 0.0
                    eval_episode_rewards.append(ep_reward)
                    eval_episode_costs.append(ep_cost)
                    eval_episode_formation_scores.append(ep_formation)
                    one_episode_rewards[:, eval_i] = 0
                    one_episode_costs[:, eval_i] = 0
                    success = bool(goal_met) and (ep_cost == 0)
                    eval_episode_successes.append(success)

                    if eval_episode >= eval_episodes:
                        break

            success_count = sum(eval_episode_successes)
            success_rate = success_count / eval_episodes if eval_episodes > 0 else 0.0
            formation_completion_rate = float(np.mean(eval_episode_formation_scores)) if eval_episode_formation_scores else 0.0
            return (
                float(np.mean(eval_episode_rewards)),
                float(np.mean(eval_episode_costs)),
                success_rate,
                success_count,
                formation_completion_rate,
            )

        result = runner.eval(eval_episodes)
        if isinstance(result, tuple) and len(result) == 3:
            reward, cost, success_count = result
            success_rate = success_count / eval_episodes if eval_episodes > 0 else 0.0
            return reward, cost, success_rate, success_count, None
        else:
            reward, cost = result[0], result[1]
            return reward, cost, None, None, None
    finally:
        if os.path.exists(eval_config["log_dir"]):
            shutil.rmtree(eval_config["log_dir"], ignore_errors=True)


def single_runs_eval(eval_dir, eval_episodes, config_overrides=None):

    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))
    env = config['task'] if 'task' in config.keys() else config['env_name']
    if env in multi_agent_velocity_map.keys() or env in multi_agent_goal_tasks or env in multi_agent_formation_tasks:
        reward, cost, success_rate, success_count, formation_completion_rate = eval_multi_agent(
            eval_dir, eval_episodes, config_overrides=config_overrides,
        )
        return reward, cost, success_rate, success_count, None, formation_completion_rate  # 多智能体暂无逐 episode 详细结果
    else:
        reward, cost, success_rate, success_count, episode_results = eval_single_agent(eval_dir, eval_episodes)
        return reward, cost, success_rate, success_count, episode_results, None

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
    parser.add_argument(
        "--num-agents",
        type=int,
        default=None,
        help="Override num_agents for formation tasks (must match training; old config.json may omit this field)",
    )
    parser.add_argument(
        "--formation-shape",
        type=str,
        default=None,
        choices=["mesh", "line", "wedge", "circle"],
        help="Override formation shape for formation tasks during evaluation",
    )
    parser.add_argument(
        "--eval-num-envs",
        type=int,
        default=None,
        help="Override eval parallel env count; set to 1 for single-process evaluation",
    )
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="Append to existing result file instead of overwriting it",
    )
    parser.add_argument(
        "--timestamp-result-file",
        action="store_true",
        help="Create a timestamped result filename (e.g., eval_result_YYYYmmdd_HHMMSS.txt)",
    )
    parser.add_argument(
        "--use-rmp-collision",
        type=lambda x: bool(strtobool(x)),
        default=None,
        help="Override RMP collision-node switch during evaluation",
    )
    parser.add_argument(
        "--use-rmp-formation",
        type=lambda x: bool(strtobool(x)),
        default=None,
        help="Override RMP formation-node switch during evaluation",
    )

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
    config_overrides = {}
    if args.num_agents is not None:
        config_overrides["num_agents"] = int(args.num_agents)
    if args.formation_shape is not None:
        config_overrides["formation_shape"] = str(args.formation_shape)
    if args.eval_num_envs is not None:
        n_envs = int(args.eval_num_envs)
        config_overrides["n_rollout_threads"] = n_envs
        config_overrides["n_eval_rollout_threads"] = n_envs
    if args.use_rmp_collision is not None:
        config_overrides["use_rmp_collision"] = bool(args.use_rmp_collision)
    if args.use_rmp_formation is not None:
        config_overrides["use_rmp_formation"] = bool(args.use_rmp_formation)

    # 列出基准目录下的所有环境
    if args.timestamp_result_file:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_parts = []
        if args.use_rmp_collision is not None:
            suffix_parts.append(f"col{1 if args.use_rmp_collision else 0}")
        if args.use_rmp_formation is not None:
            suffix_parts.append(f"for{1 if args.use_rmp_formation else 0}")
        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
        eval_result_path = os.path.join(save_dir, f"eval_result_{ts}{suffix}.txt")
    else:
        eval_result_path = os.path.join(save_dir, "eval_result.txt")

    file_exists = os.path.exists(eval_result_path)
    should_append = bool(args.append_result or args.timestamp_result_file)
    # 覆盖模式: 首次写入使用 'w' 并写表头
    # 追加模式: 若文件已存在则直接 'a' 且不重复写表头；新文件仍需写表头
    first_write = not (should_append and file_exists)
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
            rewards, costs, success_rates, success_counts, formation_completion_rates = [], [], [], [], []
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
                reward, cost, success_rate, success_count, episode_results, formation_completion_rate = single_runs_eval(
                    seed_path, eval_episodes, config_overrides=config_overrides or None,
                )
                rewards.append(reward)
                costs.append(cost)
                if success_rate is not None and success_count is not None:
                    success_rates.append(success_rate)
                    success_counts.append(success_count)
                if formation_completion_rate is not None:
                    formation_completion_rates.append(formation_completion_rate)
                # 将每个 episode 的详细结果写入 eval_result.txt
                if episode_results is not None:
                    mode = 'w' if first_write else 'a'
                    with open(eval_result_path, mode, encoding='utf-8') as f:
                        if first_write:
                            f.write("algorithm\ttask\tseed\tepisode\tnoise\treward\tcost\tsuccess\tsuccess_rate\tformation_completion\n")
                            first_write = False
                        for ep_idx, (noise, ep_reward, ep_cost, success) in enumerate(episode_results, start=1):
                            ep_success = 1 if success else 0
                            f.write(f"{algo}\t{env}\t{seed}\t{ep_idx}\t{noise}\t{round(ep_reward, 2)}\t{round(ep_cost, 2)}\t{ep_success}\t-\t-\n")
                    # 每个 seed 写入一行成功率汇总
                    if success_rate is not None and success_count is not None:
                        with open(eval_result_path, 'a', encoding='utf-8') as f:
                            f.write(f"{algo}\t{env}\t{seed}\tSUMMARY\t-\t-\t-\t{success_count}/{eval_episodes}\t{success_rate:.2%}\t-\n")
                else:
                    # 多智能体：写入汇总行（含成功率）
                    mode = 'w' if first_write else 'a'
                    with open(eval_result_path, mode, encoding='utf-8') as f:
                        if first_write:
                            f.write("algorithm\ttask\tseed\tepisode\tnoise\treward\tcost\tsuccess\tsuccess_rate\tformation_completion\n")
                            first_write = False
                        sr_str = f"{success_count}/{eval_episodes}" if success_count is not None else "-"
                        rate_str = f"{success_rate:.2%}" if success_rate is not None else "-"
                        formation_str = f"{formation_completion_rate:.2%}" if formation_completion_rate is not None else "-"
                        f.write(f"{algo}\t{env}\t{seed}\tSUMMARY\t-\t{round(reward, 2)}\t{round(cost, 2)}\t{sr_str}\t{rate_str}\t{formation_str}\n")
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
            if formation_completion_rates:
                fc_mean = np.mean(formation_completion_rates)
                fc_std = np.std(formation_completion_rates)
                msg += f", formation_completion: {fc_mean:.2%}±{fc_std:.2%}"
            msg += f", the result is saved in {eval_result_path}"
            print(msg)
            
if __name__ == '__main__':
    benchmark_eval()
