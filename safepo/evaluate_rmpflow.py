import argparse
import json
import os
import tempfile
import shutil
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from safepo.common.env import make_formation_nav_env
from safepo.multi_agent.rmp_corrector import RMPCorrector


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def _extract_goal_met(info_obj) -> bool:
    info = info_obj
    if isinstance(info, (list, tuple)) and len(info) > 0:
        info = info[0]
    if isinstance(info, dict):
        goal_met = info.get("goal_met", False)
        if hasattr(goal_met, "__len__") and not isinstance(goal_met, (str, dict)):
            return bool(goal_met[0]) if len(goal_met) > 0 else False
        return bool(goal_met)
    return False


def eval_rmpflow_only(seed_dir: str, eval_episodes: int, num_agents_override: Optional[int] = None):
    config_path = os.path.join(seed_dir, "config.json")
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    env_name = cfg["env_name"]
    if "FormationGoal" not in env_name:
        raise ValueError(f"Only formation-goal tasks are supported, got: {env_name}")

    eval_cfg = dict(cfg)
    eval_cfg["n_rollout_threads"] = int(eval_cfg.get("n_eval_rollout_threads", 1))
    eval_cfg["n_eval_rollout_threads"] = int(eval_cfg.get("n_eval_rollout_threads", 1))
    eval_cfg["use_tensorboard"] = False
    eval_cfg["log_dir"] = tempfile.mkdtemp(prefix="eval_rmpflow_")
    eval_cfg["device"] = eval_cfg.get("device", "cpu")

    num_agents = int(num_agents_override or eval_cfg.get("num_agents", 2))
    eval_env = make_formation_nav_env(
        task=env_name,
        seed=np.random.randint(0, 1000000),
        num_agents=num_agents,
        cfg_train=eval_cfg,
        render_mode=None,
    )

    rmp = RMPCorrector(
        num_agents=num_agents,
        num_envs=eval_cfg["n_eval_rollout_threads"],
        device=eval_cfg["device"],
        config=eval_cfg,
    )
    if not rmp.rmp_enabled:
        raise RuntimeError(
            "RMPFlow is not enabled/available. Check /workspace/multi-robot-rmpflow and use_rmp settings."
        )

    reward_ep = []
    cost_ep = []
    success_ep = []
    one_episode_rewards = torch.zeros(1, eval_cfg["n_eval_rollout_threads"], device=eval_cfg["device"])
    one_episode_costs = torch.zeros(1, eval_cfg["n_eval_rollout_threads"], device=eval_cfg["device"])

    eval_obs, _, _ = eval_env.reset()
    del eval_obs  # RMPFlow-only does not use policy observation
    eval_episode = 0

    try:
        while eval_episode < eval_episodes:
            zero_actions = []
            for agent_id in range(num_agents):
                act_dim = int(np.prod(eval_env.action_space[agent_id].shape))
                zero_actions.append(
                    torch.zeros(
                        eval_cfg["n_eval_rollout_threads"],
                        act_dim,
                        device=eval_cfg["device"],
                        dtype=torch.float32,
                    )
                )

            agent_positions, agent_velocities = rmp.get_agent_positions_from_env(
                eval_env, eval_cfg["n_eval_rollout_threads"]
            )
            eval_actions = rmp.apply_correction(zero_actions, agent_positions, agent_velocities)

            _, _, eval_rewards, eval_costs, eval_dones, eval_infos, _ = eval_env.step(eval_actions)

            reward_env = torch.mean(eval_rewards, dim=1).flatten()
            cost_env = torch.mean(eval_costs, dim=1).flatten()
            one_episode_rewards += reward_env
            one_episode_costs += cost_env

            eval_dones_env = torch.all(eval_dones, dim=1)
            for env_idx in range(eval_cfg["n_eval_rollout_threads"]):
                if not bool(eval_dones_env[env_idx]):
                    continue
                eval_episode += 1
                ep_reward = one_episode_rewards[:, env_idx].mean().item()
                ep_cost = one_episode_costs[:, env_idx].mean().item()
                one_episode_rewards[:, env_idx] = 0
                one_episode_costs[:, env_idx] = 0

                goal_met = False
                if eval_infos is not None and env_idx < len(eval_infos):
                    goal_met = _extract_goal_met(eval_infos[env_idx])
                success = bool(goal_met) and (float(ep_cost) == 0.0)

                reward_ep.append(ep_reward)
                cost_ep.append(ep_cost)
                success_ep.append(success)
                if eval_episode >= eval_episodes:
                    break
    finally:
        try:
            eval_env.close()
        except Exception:
            pass
        if os.path.exists(eval_cfg["log_dir"]):
            shutil.rmtree(eval_cfg["log_dir"], ignore_errors=True)

    success_count = sum(1 for s in success_ep if s)
    success_rate = success_count / len(success_ep) if success_ep else 0.0
    return _safe_mean(reward_ep), _safe_mean(cost_ep), success_rate, success_count


def benchmark_eval_rmpflow():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Parent dir containing task subdirs (e.g., SafetyPointMultiFormationGoal0-v0)",
    )
    parser.add_argument(
        "--seed-dir",
        type=str,
        default=None,
        help="Single seed directory that contains config.json (no benchmark traversal needed)",
    )
    parser.add_argument("--eval-episodes", type=int, default=20, help="episodes per seed")
    parser.add_argument("--save-dir", type=str, default=None, help="result dir (default: benchmark runs->results)")
    parser.add_argument(
        "--task-keyword",
        type=str,
        default="MultiFormationGoal",
        help="only evaluate tasks containing this keyword",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=None,
        help="override num_agents for formation task",
    )
    parser.add_argument(
        "--algo-name",
        type=str,
        default="rmpflow_only",
        help="algorithm label written to eval_result.txt",
    )
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=3,
        help="Repeat full evaluation N times in one launch (default: 3)",
    )
    args = parser.parse_args()

    if not args.benchmark_dir and not args.seed_dir:
        raise ValueError("Please provide either --benchmark-dir or --seed-dir.")

    if args.seed_dir:
        seed_dir = args.seed_dir
        cfg = json.load(open(os.path.join(seed_dir, "config.json"), "r", encoding="utf-8"))
        env_name = cfg["env_name"]
        save_dir = args.save_dir or os.path.join(seed_dir, "results_rmpflow")
        os.makedirs(save_dir, exist_ok=True)
        eval_result_path = os.path.join(save_dir, "eval_result.txt")
        with open(eval_result_path, "w", encoding="utf-8") as f:
            f.write("algorithm\ttask\tseed\tepisode\tnoise\treward\tcost\tsuccess\tsuccess_rate\n")
            for run_idx in range(1, args.repeat_runs + 1):
                reward, cost, success_rate, success_count = eval_rmpflow_only(
                    seed_dir=seed_dir,
                    eval_episodes=args.eval_episodes,
                    num_agents_override=args.num_agents,
                )
                f.write(
                    f"{args.algo_name}\t{env_name}\tseed_dir\tRUN{run_idx}\t-\t{round(reward, 2)}\t{round(cost, 2)}\t"
                    f"{success_count}/{args.eval_episodes}\t{success_rate:.2%}\n"
                )
                print(
                    f"[RUN {run_idx}/{args.repeat_runs}] {args.algo_name} in {env_name} "
                    f"reward: {round(reward, 2)}, cost: {round(cost, 2)}, "
                    f"success_rate: {success_count}/{args.eval_episodes} ({success_rate:.2%})"
                )
        print(f"Result saved in {eval_result_path}")
        return

    benchmark_dir = args.benchmark_dir
    save_dir = args.save_dir or benchmark_dir.replace("runs", "results")
    os.makedirs(save_dir, exist_ok=True)
    eval_result_path = os.path.join(save_dir, "eval_result.txt")

    first_write = True
    envs = os.listdir(benchmark_dir)
    for env in envs:
        if args.task_keyword not in env:
            continue
        env_path = os.path.join(benchmark_dir, env)
        if not os.path.isdir(env_path):
            continue

        algos = os.listdir(env_path)
        for algo in algos:
            algo_path = os.path.join(env_path, algo)
            if not os.path.isdir(algo_path):
                continue

            print(f"Start evaluating RMPFlow-only in {env}, source algo dir: {algo}")
            rewards, costs, success_rates, success_counts = [], [], [], []
            seeds = os.listdir(algo_path)
            valid_seed_num = 0

            for seed in seeds:
                seed_path = os.path.join(algo_path, seed)
                config_path = os.path.join(seed_path, "config.json")
                if not os.path.isdir(seed_path) or not os.path.exists(config_path):
                    continue
                valid_seed_num += 1

                reward, cost, success_rate, success_count = eval_rmpflow_only(
                    seed_path=seed_path,
                    eval_episodes=args.eval_episodes,
                    num_agents_override=args.num_agents,
                )
                rewards.append(reward)
                costs.append(cost)
                success_rates.append(success_rate)
                success_counts.append(success_count)

                mode = "w" if first_write else "a"
                with open(eval_result_path, mode, encoding="utf-8") as f:
                    if first_write:
                        f.write("algorithm\ttask\tseed\tepisode\tnoise\treward\tcost\tsuccess\tsuccess_rate\n")
                        first_write = False
                    f.write(
                        f"{args.algo_name}\t{env}\t{seed}\tSUMMARY\t-\t{round(reward, 2)}\t{round(cost, 2)}\t"
                        f"{success_count}/{args.eval_episodes}\t{success_rate:.2%}\n"
                    )

            if not rewards:
                continue

            reward_mean = round(np.mean(rewards), 2)
            reward_std = round(np.std(rewards), 2)
            cost_mean = round(np.mean(costs), 2)
            cost_std = round(np.std(costs), 2)
            sr_mean = np.mean(success_rates)
            sr_std = np.std(success_rates)
            total_success = sum(success_counts)
            total_eps = valid_seed_num * args.eval_episodes
            msg = (
                f"After {args.eval_episodes} episodes evaluation, {args.algo_name} in {env} "
                f"reward: {reward_mean}±{reward_std}, cost: {cost_mean}±{cost_std}, "
                f"success_rate: {total_success}/{total_eps} ({sr_mean:.2%}±{sr_std:.2%}), "
                f"result saved in {eval_result_path}"
            )
            print(msg)


if __name__ == "__main__":
    benchmark_eval_rmpflow()
