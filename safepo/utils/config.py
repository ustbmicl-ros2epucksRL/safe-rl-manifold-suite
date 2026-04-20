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


import os
import random
from distutils.util import strtobool

import numpy as np
import copy
import torch
import time
import yaml
import argparse


multi_agent_velocity_map = {
    'Safety2x4AntVelocity-v0': {
        'agent_conf': '2x4',
        'scenario': 'Ant',
    },
    'Safety4x2AntVelocity-v0': {
        'agent_conf': '4x2',
        'scenario': 'Ant',
    },
    'Safety2x3HalfCheetahVelocity-v0': {
        'agent_conf': '2x3',
        'scenario': 'HalfCheetah',
    },
    'Safety6x1HalfCheetahVelocity-v0': {
        'agent_conf': '6x1',
        'scenario': 'HalfCheetah',
    },
    'Safety3x1HopperVelocity-v0': {
        'agent_conf': '3x1',
        'scenario': 'Hopper',
    },
    'Safety2x3Walker2dVelocity-v0': {
        'agent_conf': '2x3',
        'scenario': 'Walker2d',
    },
    'Safety2x1SwimmerVelocity-v0': {
        'agent_conf': '2x1',
        'scenario': 'Swimmer',
    },
    'Safety9|8HumanoidVelocity-v0': {
        'agent_conf': '9|8',
        'scenario': 'Humanoid',
    },
}

multi_agent_goal_tasks = [
    "SafetyPointMultiGoal0-v0",
    "SafetyPointMultiGoal1-v0",
    "SafetyPointMultiGoal2-v0",
    "SafetyAntMultiGoal0-v0",
    "SafetyAntMultiGoal1-v0",
    "SafetyAntMultiGoal2-v0",
]

# 编队导航任务
multi_agent_formation_tasks = [
    # 例如：
    "SafetyPointMultiFormationGoal0-v0",
    # "SafetyAntMultiFormation0-v0",
]

isaac_gym_map = {
    "ShadowHandOver_Safe_finger": "shadow_hand_over_safe_finger",
    "ShadowHandOver_Safe_joint": "shadow_hand_over_safe_joint",
    "ShadowHandCatchOver2Underarm_Safe_finger": "shadow_hand_catch_over_2_underarm_safe_finger",
    "ShadowHandCatchOver2Underarm_Safe_joint": "shadow_hand_catch_over_2_underarm_safe_joint",
    "FreightFrankaCloseDrawer": "freight_franka_close_drawer",
    "FreightFrankaPickAndPlace": "freight_franka_pick_and_place",
}

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)

def warn_task_name():
    raise Exception(
        "Unrecognized task!")

def set_seed(seed, torch_deterministic=False):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.autograd.set_detect_anomaly(True)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    return seed

def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    try:
        from isaacgym import gymapi, gymutil
    except ImportError:
        raise Exception("Please install isaacgym to run Isaac Gym tasks!")
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def single_agent_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default":0, "help": "Random seed"},
        {"name": "--use-eval", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Use evaluation environment for testing"},
        {"name": "--task", "type": str, "default": "SafetyPointGoal1-v0", "help": "The task to run"},
        {"name": "--num-envs", "type": int, "default": 10, "help": "The number of parallel game environments"},
        {"name": "--experiment", "type": str, "default": "single_agent_exp", "help": "Experiment name"},
        {"name": "--log-dir", "type": str, "default": "../runs", "help": "directory to save agent logs"},
        {"name": "--device", "type": str, "default": "cuda", "help": "The device to run the model on"},
        {"name": "--device-id", "type": int, "default": 0, "help": "The device id to run the model on"},
        {"name": "--write-terminal", "type": lambda x: bool(strtobool(x)), "default": True, "help": "Toggles terminal logging"},
        {"name": "--headless", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Toggles headless mode"},
        {"name": "--total-steps", "type": int, "default": 10000000, "help": "Total timesteps of the experiments"},
        {"name": "--steps-per-epoch", "type": int, "default": 20000, "help": "The number of steps to run in each environment per policy rollout"},
        {"name": "--randomize", "type": bool, "default": False, "help": "Wheather to randomize the environments' initial states"},
        {"name": "--cost-limit", "type": float, "default": 25.0, "help": "cost_lim"},
        {"name": "--lagrangian-multiplier-init", "type": float, "default": 0.001, "help": "initial value of lagrangian multiplier"},
        {"name": "--lagrangian-multiplier-lr", "type": float, "default": 0.035, "help": "learning rate of lagrangian multiplier"},
    ]
    # Create argument parser
    parser = argparse.ArgumentParser(description="RL Policy")
    issac_parameters = copy.deepcopy(custom_parameters)
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    # Parse arguments

    args = parser.parse_args()
    # Disable file-based terminal log redirection across entrypoints.
    args.write_terminal = True
    cfg_env={}
    base_path = os.path.dirname(os.path.abspath(__file__)).replace("utils", "multi_agent")
    if args.task in isaac_gym_map.keys():
        try:
            from isaacgym import gymutil
        except ImportError:
            raise Exception("Please install isaacgym to run Isaac Gym tasks!")
        args = gymutil.parse_arguments(description="RL Policy", custom_parameters=issac_parameters)
        args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'
        args.write_terminal = True
        cfg_env_path = "marl_cfg/{}.yaml".format(isaac_gym_map[args.task])
        with open(os.path.join(base_path, cfg_env_path), 'r') as f:
            cfg_env = yaml.load(f, Loader=yaml.SafeLoader)
            cfg_env["name"] = args.task
            if "task" in cfg_env:
                if "randomize" not in cfg_env["task"]:
                    cfg_env["task"]["randomize"] = args.randomize
                else:
                    cfg_env["task"]["randomize"] = False
    return args, cfg_env


def multi_agent_args(algo):

    # Define custom parameters
    custom_parameters = [
        {"name": "--use-eval", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Use evaluation environment for testing"},
        {"name": "--task", "type": str, "default": "Safety2x4AntVelocity-v0", "help": "The task to run"},
        {"name": "--agent-conf", "type": str, "default": "2x4", "help": "The agent configuration"},
        {"name": "--scenario", "type": str, "default": "Ant", "help": "The scenario"},
        {"name": "--experiment", "type": str, "default": "Base", "help": "Experiment name"},
        {"name": "--seed", "type": int, "default":0, "help": "Random seed"},
        {"name": "--model-dir", "type": str, "default": "", "help": "Choose a model dir"},
        {"name": "--cost-limit", "type": float, "default": 25.0, "help": "cost_lim"},
        {"name": "--device", "type": str, "default": "cpu", "help": "The device to run the model on"},
        {"name": "--device-id", "type": int, "default": 0, "help": "The device id to run the model on"},
        {"name": "--write-terminal", "type": lambda x: bool(strtobool(x)), "default": True, "help": "Toggles terminal logging"},
        {"name": "--headless", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Toggles headless mode"},
        {"name": "--total-steps", "type": int, "default": None, "help": "Total timesteps of the experiments"},
        {"name": "--num-envs", "type": int, "default": None, "help": "The number of parallel game environments"},
        {"name": "--randomize", "type": bool, "default": False, "help": "Wheather to randomize the environments' initial states"},
        {"name": "--render-mode", "type": str, "default": None, "help": "Render mode: 'human' to display, None to disable"},
        {"name": "--render", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Enable rendering during training"},
        {"name": "--use-tensorboard", "type": lambda x: bool(strtobool(x)), "default": True, "help": "Enable TensorBoard logging for training visualization"},
        # For formation navigation tasks: number of robots/agents in formation
        {"name": "--num_agents", "type": int, "default": 2, "help": "Number of robots (agents) in the formation for multi-goal navigation"},
        # RMP formation graph (mappo_rmp; passed through cfg_train to RMPCorrector)
        {
            "name": "--formation-shape",
            "type": str,
            "default": "mesh",
            "choices": ["mesh", "line", "wedge", "circle"],
            "help": "RMP formation: mesh | line | wedge | circle",
        },
        {"name": "--formation-line-axis", "type": str, "default": "x", "help": "For line: x (row +x) or y (column +y)"},
        {"name": "--formation-wedge-half-angle-deg", "type": float, "default": 35.0, "help": "For wedge: half opening angle at apex (degrees)"},
        {"name": "--formation-target-distance", "type": float, "default": 0.5, "help": "Target neighbor spacing for RMP and formation task reward"},
        {"name": "--use-rmp-collision", "type": lambda x: bool(strtobool(x)), "default": True, "help": "Enable RMP collision-avoidance nodes (inter-agent + wall)"},
        {"name": "--use-rmp-formation", "type": lambda x: bool(strtobool(x)), "default": True, "help": "Enable RMP formation nodes"},
        {"name": "--use-cbf", "type": lambda x: bool(strtobool(x)), "default": True, "help": "Enable CBF safety correction"},
        {"name": "--cbf-safety-radius", "type": float, "default": 0.35, "help": "Safety radius for CBF pairwise barrier"},
        {"name": "--cbf-gain", "type": float, "default": 0.8, "help": "Repulsive gain in CBF correction"},
        {"name": "--cbf-max-correction", "type": float, "default": 0.6, "help": "Max correction norm per agent action"},
        {"name": "--cbf-weight", "type": float, "default": 1.0, "help": "Weight for fusing CBF correction into action"},
    ]
    # Create argument parser
    parser = argparse.ArgumentParser(description="RL Policy")
    issac_parameters = copy.deepcopy(custom_parameters)
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    # Parse arguments

    args = parser.parse_args()
    # Disable file-based terminal log redirection across entrypoints.
    args.write_terminal = True

    if args.task in isaac_gym_map.keys():
        try:
            from isaacgym import gymutil
        except ImportError:
            raise Exception("Please install isaacgym to run Isaac Gym tasks!")
        args = gymutil.parse_arguments(description="RL Policy", custom_parameters=issac_parameters)
        args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'
        args.write_terminal = True
    cfg_train_path = "marl_cfg/{}/config.yaml".format(algo)
    base_path = os.path.dirname(os.path.abspath(__file__)).replace("utils", "multi_agent")
    with open(os.path.join(base_path, cfg_train_path), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)
        if args.task in multi_agent_velocity_map.keys():
            cfg_train.update(cfg_train.get("mamujoco"))
            args.agent_conf = multi_agent_velocity_map[args.task]["agent_conf"]
            args.scenario = multi_agent_velocity_map[args.task]["scenario"]
        elif args.task in multi_agent_goal_tasks:
            cfg_train.update(cfg_train.get("mamujoco"))

    cfg_train["use_eval"] = args.use_eval
    cfg_train["use_tensorboard"] = getattr(args, "use_tensorboard", True)
    cfg_train["cost_limit"]=args.cost_limit
    cfg_train["algorithm_name"]=algo
    cfg_train["device"] = args.device + ":" + str(args.device_id)

    cfg_train["env_name"] = args.task

    cfg_train["formation_shape"] = getattr(args, "formation_shape", "mesh")
    cfg_train["formation_line_axis"] = getattr(args, "formation_line_axis", "x")
    cfg_train["formation_wedge_half_angle_deg"] = getattr(
        args, "formation_wedge_half_angle_deg", 35.0
    )
    cfg_train["formation_target_distance"] = getattr(
        args, "formation_target_distance", 0.5
    )
    cfg_train["use_rmp_collision"] = getattr(args, "use_rmp_collision", True)
    cfg_train["use_rmp_formation"] = getattr(args, "use_rmp_formation", True)
    cfg_train["num_agents"] = getattr(args, "num_agents", 2)
    cfg_train["use_cbf"] = getattr(args, "use_cbf", True)
    cfg_train["cbf_safety_radius"] = getattr(args, "cbf_safety_radius", 0.35)
    cfg_train["cbf_gain"] = getattr(args, "cbf_gain", 0.8)
    cfg_train["cbf_max_correction"] = getattr(args, "cbf_max_correction", 0.6)
    cfg_train["cbf_weight"] = getattr(args, "cbf_weight", 1.0)

    if args.total_steps:
        cfg_train["num_env_steps"] = args.total_steps
    if args.num_envs:
        cfg_train["n_rollout_threads"] = args.num_envs
        cfg_train["n_eval_rollout_threads"] = args.num_envs
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    # Encode RMP node configuration into run folder name for easier comparison.
    if algo == "mappo_rmp":
        rmp_collision_tag = 1 if cfg_train.get("use_rmp_collision", True) else 0
        rmp_formation_tag = 1 if cfg_train.get("use_rmp_formation", True) else 0
        relpath = f"{relpath}-col{rmp_collision_tag}_for{rmp_formation_tag}"
    cfg_train['log_dir']="../runs/"+args.experiment+'/'+args.task+'/'+algo+'/'+relpath
    cfg_env={}
    if args.task in isaac_gym_map.keys():
        cfg_env_path = "marl_cfg/{}.yaml".format(isaac_gym_map[args.task])
        with open(os.path.join(base_path, cfg_env_path), 'r') as f:
            cfg_env = yaml.load(f, Loader=yaml.SafeLoader)
            cfg_env["name"] = args.task
            if "task" in cfg_env:
                if "randomize" not in cfg_env["task"]:
                    cfg_env["task"]["randomize"] = args.randomize
                else:
                    cfg_env["task"]["randomize"] = False
            else:
                cfg_env["task"] = {"randomize": False}
    elif args.task in multi_agent_velocity_map.keys() or args.task in multi_agent_goal_tasks or args.task in multi_agent_formation_tasks:
        pass
    else:
        warn_task_name()

    return args, cfg_env, cfg_train

