from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class EnvConfig:
    num_agents: int = 4
    num_obstacles: int = 4
    arena_size: float = 10.0
    dt: float = 0.05
    max_steps: int = 500
    formation_shape: str = "square"  # "polygon", "line", "v"
    formation_radius: float = 1.0
    max_velocity: float = 2.0
    max_acceleration: float = 1.0
    obstacle_radius: float = 0.5
    goal_threshold: float = 0.5


@dataclass
class SafetyConfig:
    safety_radius: float = 0.4
    K_c: float = 100.0
    slack_type: str = "softcorner"
    slack_beta: float = 30.0
    slack_beta_formation: float = 10.0
    slack_threshold: float = 1e-3
    rmp_formation_blend: float = 0.3
    dq_max: float = 1.0
    boundary_margin: float = 0.5
    eps_pinv: float = 1e-6


@dataclass
class AlgoConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epochs: int = 10
    num_mini_batch: int = 4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_shared_actor: bool = True


@dataclass
class RewardConfig:
    w_nav: float = 1.0
    w_formation: float = 0.1
    w_smooth: float = 0.01
    goal_bonus: float = 10.0


@dataclass
class TrainConfig:
    total_episodes: int = 5000
    episode_length: int = 500
    seed: int = 0
    log_interval: int = 10
    save_interval: int = 100
    eval_episodes: int = 10
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    device: str = "cpu"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
