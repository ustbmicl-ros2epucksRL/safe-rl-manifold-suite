import copy
import os
import sys

import torch

from safepo.common.env import (
    make_formation_nav_env,
    make_ma_isaac_env,
    make_ma_mujoco_env,
    make_ma_multi_goal_env,
)
from safepo.multi_agent.cbf_corrector import CBFCorrector
from safepo.multi_agent.mappo_rmp import Runner as RMPBasedRunner
from safepo.utils.config import (
    isaac_gym_map,
    multi_agent_args,
    multi_agent_goal_tasks,
    multi_agent_formation_tasks,
    multi_agent_velocity_map,
    parse_sim_params,
    set_np_formatting,
    set_seed,
)


class Runner(RMPBasedRunner):
    """Reuse MAPPO training loop and swap RMP correction with CBF correction."""

    def __init__(self, vec_env, vec_eval_env, config, model_dir=""):
        super().__init__(vec_env=vec_env, vec_eval_env=vec_eval_env, config=config, model_dir=model_dir)
        self.rmp_corrector = CBFCorrector(
            num_agents=self.num_agents,
            num_envs=self.config["n_rollout_threads"],
            device=self.config["device"],
            config=self.config,
        )


def train(args, cfg_train):
    env = None
    eval_env = None
    agent_index = [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]]
    try:
        if args.task in multi_agent_velocity_map:
            env = make_ma_mujoco_env(
                scenario=args.scenario,
                agent_conf=args.agent_conf,
                seed=args.seed,
                cfg_train=cfg_train,
            )
            cfg_eval = copy.deepcopy(cfg_train)
            cfg_eval["seed"] = args.seed + 10000
            cfg_eval["n_rollout_threads"] = cfg_eval["n_eval_rollout_threads"]
            eval_env = make_ma_mujoco_env(
                scenario=args.scenario,
                agent_conf=args.agent_conf,
                seed=cfg_eval["seed"],
                cfg_train=cfg_eval,
            )
        elif args.task in isaac_gym_map:
            sim_params = parse_sim_params(args, cfg_env, cfg_train)
            env = make_ma_isaac_env(args, cfg_env, cfg_train, sim_params, agent_index)
            cfg_train["n_rollout_threads"] = env.num_envs
            cfg_train["n_eval_rollout_threads"] = env.num_envs
            eval_env = env
        elif args.task in multi_agent_goal_tasks:
            env = make_ma_multi_goal_env(task=args.task, seed=args.seed, cfg_train=cfg_train)
            cfg_eval = copy.deepcopy(cfg_train)
            cfg_eval["seed"] = args.seed + 10000
            cfg_eval["n_rollout_threads"] = cfg_eval["n_eval_rollout_threads"]
            eval_env = make_ma_multi_goal_env(
                task=args.task, seed=args.seed + 10000, cfg_train=cfg_eval
            )
        elif args.task in multi_agent_formation_tasks:
            render_mode = getattr(args, "render_mode", None)
            if render_mode is None and getattr(args, "render", False):
                render_mode = "human"
            cfg_train["render"] = getattr(args, "render", False) or render_mode == "human"
            env = make_formation_nav_env(
                task=args.task,
                seed=args.seed,
                num_agents=args.num_agents,
                cfg_train=cfg_train,
                render_mode=render_mode,
            )
            cfg_eval = copy.deepcopy(cfg_train)
            cfg_eval["seed"] = args.seed + 10000
            cfg_eval["n_rollout_threads"] = cfg_eval["n_eval_rollout_threads"]
            eval_env = make_formation_nav_env(
                task=args.task,
                seed=args.seed + 10000,
                num_agents=args.num_agents,
                cfg_train=cfg_eval,
            )
        else:
            raise NotImplementedError

        torch.set_num_threads(4)
        runner = Runner(env, eval_env, cfg_train, args.model_dir)
        if args.model_dir != "":
            runner.eval(100000)
        else:
            runner.run()
    finally:
        print("Closing environments...", flush=True)
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception:
                pass
        if env is not None and env is not eval_env:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    set_np_formatting()
    args, cfg_env, cfg_train = multi_agent_args(algo="cbf_mappo")
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))

    # Explicitly default-enable CBF correction.
    cfg_train["use_cbf"] = bool(cfg_train.get("use_cbf", True))

    if args.write_terminal:
        train(args=args, cfg_train=cfg_train)
    else:
        terminal_log_name = f"seed{args.seed}_terminal.log"
        error_log_name = f"seed{args.seed}_error.log"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(cfg_train["log_dir"]):
            os.makedirs(cfg_train["log_dir"], exist_ok=True)
        with open(os.path.join(cfg_train["log_dir"], terminal_log_name), "w", encoding="utf-8") as f_out:
            sys.stdout = f_out
            with open(os.path.join(cfg_train["log_dir"], error_log_name), "w", encoding="utf-8") as f_error:
                sys.stderr = f_error
                train(args=args, cfg_train=cfg_train)
