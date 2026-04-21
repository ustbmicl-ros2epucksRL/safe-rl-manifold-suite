"""
MAPPO checkpoint 加载器 · 用于 chap5 部署

目标: 从 safepo.multi_agent.mappo_rmp / mappo / mappolag 训练保存的 checkpoint
      (actor_agent*.pt + config.json) 复原策略网络, 提供 act(obs) -> action.

两层 API:

1. MAPPOPolicyLoader:
   * 输入: 一个 run 目录 (例如 runs/Base/<env>/mappo_rmp/seed-000-XXX)
   * 自动读 config.json / models_seedK/actor_agent*.pt
   * 暴露 act(obs: np.ndarray (N, obs_dim)) -> np.ndarray (N, 2)
   * 确定性推理 (deterministic=True)

2. safetygym_obs_mirror:
   * 在 Webots/实物部署时用来生成与训练一致的 obs 向量
   * 开一个一次性 Safety-Gymnasium env, 把给定的 (positions, velocities) teleport 进去,
     然后读 env.obs() 作为真实 obs, 避免手动复现 lidar + compass 等复杂管线
   * 仅在 sim-to-X 阶段使用; 对于不需要 lidar 的简化实验, 可以跳过

设计说明:
    safepo 训练时保存的 checkpoint 是 per-agent (每 agent 一个 actor),
    原因是虽然参数共享但各 agent 的 learner 各自复制 state_dict. 部署时
    我们加载全部 N 个 actor, 每步对每个 agent 分别调用 forward(obs_i).

依赖:
    torch, gymnasium, safepo (editable), safety_gymnasium (editable)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# =============================================================================
# 1. Core loader
# =============================================================================

class MAPPOPolicyLoader:
    """Load MAPPO actors from a safepo training run and expose act(obs).

    用法:
        loader = MAPPOPolicyLoader("runs/Base/<env>/mappo_rmp/seed-000-XXX")
        # or:
        loader = MAPPOPolicyLoader.latest("runs/Base/<env>/mappo_rmp")
        actions = loader.act(obs)   # obs shape (N, obs_dim) -> actions (N, 2)

    字段:
        num_agents, obs_dim, act_dim, device, config (dict), actors (list[nn.Module])
    """

    def __init__(
        self,
        run_dir: str | os.PathLike,
        device: str = "cpu",
        seed_override: Optional[int] = None,
    ):
        import torch

        self.device = torch.device(device)
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"run_dir 不存在: {self.run_dir}")

        # 读训练时的 config.json
        cfg_path = self.run_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"{cfg_path} 不存在. 请确认该 run 是 safepo 训练产生的 (config.json 应由 logger.save_config 写出)"
            )
        with open(cfg_path, "r") as f:
            self.config = json.load(f)

        self.num_agents = int(self.config.get("num_agents", 2))
        seed_k = int(self.config.get("seed", 0)) if seed_override is None else int(seed_override)
        models_dir = self.run_dir / f"models_seed{seed_k}"
        if not models_dir.is_dir():
            # 兜底: 扫 models_seed* 目录
            candidates = sorted(self.run_dir.glob("models_seed*"))
            if not candidates:
                raise FileNotFoundError(f"未找到 models_seedK 目录 (run={self.run_dir})")
            models_dir = candidates[0]
        self.models_dir = models_dir

        # 定位 actor checkpoint
        actor_paths = [models_dir / f"actor_agent{i}.pt" for i in range(self.num_agents)]
        for p in actor_paths:
            if not p.is_file():
                raise FileNotFoundError(f"actor 缺失: {p}")

        # 构造 obs/action space 以对齐 MultiAgentActor 的 API
        obs_dim = self._infer_obs_dim_from_actor(actor_paths[0])
        self.obs_dim = obs_dim
        self.act_dim = 2  # Point agent 动作维度 (v, omega) 规范化

        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        # 实例化并加载 N 个 actor
        from safepo.common.model import MultiAgentActor
        actor_config = self._build_actor_config(self.config)
        self.actors = []
        for i, p in enumerate(actor_paths):
            actor = MultiAgentActor(actor_config, obs_space, action_space, self.device)
            state = torch.load(p, map_location=self.device, weights_only=False)
            actor.load_state_dict(state)
            actor.eval()
            self.actors.append(actor)

        # 预留 RNN state (MAPPO 默认非 recurrent, 但 API 要求传)
        self._recurrent_N = int(self.config.get("recurrent_N", 1))
        self._hidden_size = int(self.config.get("hidden_size", 512))
        print(
            f"[MAPPOPolicyLoader] loaded {self.num_agents} actors from {self.models_dir}, "
            f"obs_dim={obs_dim}, act_dim={self.act_dim}, algo={self.config.get('algorithm_name')}"
        )

    # ---- factories ----
    @classmethod
    def latest(cls, algo_dir: str | os.PathLike, **kw) -> "MAPPOPolicyLoader":
        """Pick the lexicographically latest seed-* run under an algo directory."""
        base = Path(algo_dir)
        candidates = sorted([p for p in base.glob("seed-*") if p.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"{base} 下没有 seed-* 目录")
        return cls(candidates[-1], **kw)

    # ---- config reconstruction ----
    @staticmethod
    def _build_actor_config(train_cfg: dict) -> dict:
        """Fill in fields MultiAgentActor expects; fall back to safepo defaults when missing.

        Default 值与 safepo/multi_agent/marl_cfg/*/config.yaml 对齐.
        """
        return {
            "hidden_size":                 int(train_cfg.get("hidden_size", 512)),
            "gain":                        float(train_cfg.get("gain", 0.01)),
            "actor_gain":                  float(train_cfg.get("actor_gain", 0.01)),
            "std_x_coef":                  float(train_cfg.get("std_x_coef", 1.0)),
            "std_y_coef":                  float(train_cfg.get("std_y_coef", 0.5)),
            "use_orthogonal":              bool(train_cfg.get("use_orthogonal", True)),
            "use_policy_active_masks":     bool(train_cfg.get("use_policy_active_masks", True)),
            "use_naive_recurrent_policy":  bool(train_cfg.get("use_naive_recurrent_policy", False)),
            "use_recurrent_policy":        bool(train_cfg.get("use_recurrent_policy", False)),
            "recurrent_N":                 int(train_cfg.get("recurrent_N", 1)),
            "use_feature_normalization":   bool(train_cfg.get("use_feature_normalization", True)),
            "stacked_frames":              int(train_cfg.get("stacked_frames", 1)),
            "layer_N":                     int(train_cfg.get("layer_N", 2)),
            "use_ReLU":                    bool(train_cfg.get("use_ReLU", False)),
        }

    @staticmethod
    def _infer_obs_dim_from_actor(actor_path: Path) -> int:
        """Peek the first MLP weight matrix to recover obs_dim."""
        import torch
        state = torch.load(actor_path, map_location="cpu", weights_only=False)
        # feature_norm.weight shape is (obs_dim,) 或 base.feature_norm.weight
        # mlp.fc1.0.weight shape is (hidden, obs_dim) 或 base.mlp.fc1.0.weight
        for k, v in state.items():
            if k.endswith("feature_norm.weight"):
                return int(v.shape[0])
        for k, v in state.items():
            if "mlp.fc1.0.weight" in k:
                return int(v.shape[1])
        raise RuntimeError(f"无法从 {actor_path} 推断 obs_dim; state keys = {list(state.keys())[:10]}")

    # ---- inference ----
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Run all N actors once.

        Args:
            obs: shape (N, obs_dim) or (obs_dim,) (单 agent 退化)
            deterministic: True = 取分布均值 (部署推荐)

        Returns:
            actions: shape (N, act_dim) numpy
        """
        import torch
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        N, D = obs.shape
        if N != self.num_agents:
            raise ValueError(f"obs 第 0 维应 = num_agents={self.num_agents}, 实际 {N}")
        if D != self.obs_dim:
            raise ValueError(f"obs 第 1 维应 = obs_dim={self.obs_dim}, 实际 {D}")

        # RNN state 传零 (非 recurrent 策略不会实际使用)
        rnn_states = np.zeros((1, self._recurrent_N, self._hidden_size), dtype=np.float32)
        masks = np.ones((1, 1), dtype=np.float32)

        out = np.zeros((N, self.act_dim), dtype=np.float32)
        with torch.no_grad():
            for i in range(N):
                obs_i = torch.as_tensor(obs[i:i + 1], device=self.device)
                rnn_i = torch.as_tensor(rnn_states, device=self.device)
                m_i = torch.as_tensor(masks, device=self.device)
                actions, _, _ = self.actors[i](obs_i, rnn_i, m_i, deterministic=deterministic)
                out[i] = actions.detach().cpu().numpy().reshape(-1)[: self.act_dim]
        return out


# =============================================================================
# 2. Webots/实物部署辅助: 用 safety-gymnasium env 作 "obs mirror"
# =============================================================================

class _ObsMirror:
    """延迟初始化的 safety-gymnasium env, 用来为外部 (Webots) 状态生成 obs.

    原理: safety-gymnasium 的 obs 是 lidar + compass + 自体状态的组合, 手动复现易错.
    我们 make 一个无头 env, 每步把 Webots 读出来的机器人/障碍/目标的 mujoco state
    直接 set 进去, 然后调 env.obs() 生成 obs. 本质是用 safety-gymnasium 自己做 obs builder.

    限制:
        * 需要能 `import safety_gymnasium` 并创建同名任务 (chap5 conda env 满足)
        * 仅在 sim-to-X 验证期使用, 不适合 resource-constrained 实物节点
        * mujoco state-set 的细节依赖任务实现; 简单任务通常只需要 teleport agents
    """

    def __init__(self, task_name: str, num_agents: int, formation_shape: str = "wedge"):
        import safety_gymnasium
        self._task_name = task_name
        self._num_agents = num_agents
        self._env = safety_gymnasium.make(
            task_name,
            num_agents=num_agents,
            formation_shape=formation_shape,
        )
        self._obs_dim = None
        # 先 reset 一次 (对齐 safety_gymnasium 内部状态); safepo 的 MultiFormationNavEnv
        # 在 task.obs() 后再拼 one-hot agent_id 并做 z-score, 所以训练时真实 obs_dim =
        # env.task.obs().size + num_agents. 这里用 task.obs() 直接推断.
        self._env.reset(seed=0)
        raw = np.asarray(self._env.task.obs(), dtype=np.float32).reshape(-1)
        self._obs_dim = int(raw.size + num_agents)

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def obs_from_state(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        goal: np.ndarray,
        hazards: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Teleport internal env to given state, return per-agent obs.

        Args:
            positions: (N, 2)
            velocities: (N, 2)
            goal: (2,) 目标位置 (world frame)
            hazards: (K, 2) 危险物/墙体中心; 若 None 则保留 env 默认

        Returns:
            obs: (N, obs_dim)  每行是对应 agent 的观测
        """
        task = self._env.task
        N = self._num_agents
        # 机器人
        for i in range(N):
            agent_i = task.agents[i] if hasattr(task, 'agents') else None
            if agent_i is None:
                # fallback: task.agent 单 agent 结构 — 不适用多 agent 场景
                break
            agent_i.pos[:2] = positions[i]
            agent_i.vel[:2] = velocities[i]
        # 目标
        if hasattr(task, 'goal') and task.goal is not None:
            task.goal.pos[:2] = goal
        # 障碍/hazards
        if hazards is not None and hasattr(task, 'hazards') and task.hazards is not None:
            K = min(len(task.hazards.pos), len(hazards))
            for k in range(K):
                task.hazards.pos[k][:2] = hazards[k]
        # 重算 obs. safepo 的 MultiFormationNavEnv._get_obs() 在 env.task.obs()
        # 基础上对每个 agent 追加 one-hot agent_id 再做标准化 (见
        # safepo/common/wrappers.py 第 243-252 行). 训练时用的就是这个 87 维 obs,
        # 我们必须精确复现才能让加载的 actor 正确工作.
        state = np.asarray(self._env.task.obs(), dtype=np.float32).reshape(-1)
        out = np.zeros((N, self._obs_dim), dtype=np.float32)
        for a in range(N):
            one_hot = np.zeros(N, dtype=np.float32)
            one_hot[a] = 1.0
            obs_i = np.concatenate([state, one_hot])
            # safepo 在 _get_obs 里做的是逐样本 z-score
            mean = obs_i.mean()
            std = obs_i.std() + 1e-8
            obs_i = (obs_i - mean) / std
            if obs_i.size != self._obs_dim:
                # fallback: 训练时的 obs_dim 与这里的 env 偏差 — 截断/零填充到一致
                if obs_i.size > self._obs_dim:
                    obs_i = obs_i[: self._obs_dim]
                else:
                    obs_i = np.concatenate([obs_i, np.zeros(self._obs_dim - obs_i.size, dtype=np.float32)])
            out[a] = obs_i
        return out

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


_MIRROR_CACHE: dict[tuple, _ObsMirror] = {}


def safetygym_obs_mirror(
    task_name: str,
    num_agents: int,
    formation_shape: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    goal: np.ndarray,
    hazards: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience: 自动 cache 一个 mirror env 并调其 obs_from_state.

    典型使用 (Webots supervisor 里):
        obs = safetygym_obs_mirror(
            task_name='SafetyPointMultiFormationGoal0-v0',
            num_agents=3, formation_shape='wedge',
            positions=pos_from_webots,
            velocities=vel_from_webots,
            goal=np.array([0.0, 2.0]),
            hazards=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        )
        actions = mappo_loader.act(obs)
    """
    key = (task_name, num_agents, formation_shape)
    if key not in _MIRROR_CACHE:
        _MIRROR_CACHE[key] = _ObsMirror(task_name, num_agents, formation_shape)
    mirror = _MIRROR_CACHE[key]
    return mirror.obs_from_state(positions, velocities, goal, hazards)
