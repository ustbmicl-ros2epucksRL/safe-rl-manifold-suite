"""
Unified Trainer for COSMOS Framework

Handles training loop with:
- Component creation from registries
- WandB logging
- Checkpoint saving/loading
- Evaluation
"""

import os
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
import torch

from cosmos.registry import ENV_REGISTRY, ALGO_REGISTRY, SAFETY_REGISTRY
from cosmos.envs.base import BaseMultiAgentEnv
from cosmos.algos.base import BaseMARLAlgo
from cosmos.safety.base import BaseSafetyFilter

# Import buffer
from formation_nav.algo.buffer import RolloutBuffer


class Trainer:
    """
    Unified trainer for safe multi-agent RL.

    Handles:
    - Building components from registries
    - Training loop with safety filtering
    - Logging (console, WandB)
    - Checkpoint saving/loading
    - Evaluation
    """

    def __init__(self, cfg: Any):
        """
        Initialize trainer from Hydra config.

        Args:
            cfg: OmegaConf config object.
        """
        self.cfg = cfg

        # Set device
        if cfg.experiment.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.experiment.device

        # Set seed
        np.random.seed(cfg.experiment.seed)
        torch.manual_seed(cfg.experiment.seed)

        # Create output directory
        self.output_dir = os.getcwd()  # Hydra changes cwd
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Build components
        self._build_env()
        self._build_algo()
        self._build_safety()
        self._build_buffer()

        # Initialize logger
        self._init_logger()

        # Training state
        self.start_episode = 0
        self.metrics = {
            'episode': [], 'reward': [], 'cost': [],
            'formation_error': [], 'min_dist': [], 'collisions': []
        }
        self.best_reward = float('-inf')

    def _build_env(self):
        """Build environment from config."""
        env_cfg = dict(self.cfg.env)
        env_name = env_cfg.pop('name')
        self.env = ENV_REGISTRY.build(env_name, cfg=env_cfg)
        print(f"Environment: {env_name} ({self.env.num_agents} agents)")

    def _build_algo(self):
        """Build algorithm from config."""
        algo_cfg = dict(self.cfg.algo)
        algo_name = algo_cfg.pop('name')

        self.algo = ALGO_REGISTRY.build(
            algo_name,
            obs_dim=self.env.get_obs_dim(),
            share_obs_dim=self.env.get_share_obs_dim(),
            act_dim=self.env.get_act_dim(),
            num_agents=self.env.num_agents,
            cfg=algo_cfg,
            device=self.device
        )
        print(f"Algorithm: {algo_name}")

    def _build_safety(self):
        """Build safety filter from config."""
        safety_cfg = dict(self.cfg.safety)
        safety_name = safety_cfg.pop('name')

        # Get constraint info from env
        constraint_info = self.env.get_constraint_info()

        self.safety = SAFETY_REGISTRY.build(
            safety_name,
            env_cfg=self.cfg.env,
            safety_cfg=safety_cfg,
            desired_distances=constraint_info.get('desired_distances'),
            topology_edges=constraint_info.get('topology_edges'),
            obstacle_positions=constraint_info.get('obstacles'),
            mode=safety_cfg.get('mode', 'decentralized')
        )
        print(f"Safety Filter: {safety_name}")

    def _build_buffer(self):
        """Build experience buffer."""
        self.buffer = RolloutBuffer(
            episode_length=self.cfg.experiment.max_steps,
            num_agents=self.env.num_agents,
            obs_dim=self.env.get_obs_dim(),
            share_obs_dim=self.env.get_share_obs_dim(),
            act_dim=self.env.get_act_dim(),
            gamma=self.cfg.algo.gamma,
            gae_lambda=self.cfg.algo.gae_lambda,
            device=self.device
        )

    def _init_logger(self):
        """Initialize logging."""
        self.use_wandb = self.cfg.logging.use_wandb

        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.cfg.logging.wandb_project,
                    name=f"{self.cfg.experiment.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=dict(self.cfg),
                )
                print(f"WandB: {wandb.run.get_url()}")
            except ImportError:
                print("Warning: WandB not installed, disabling")
                self.use_wandb = False

        # Save config
        config_path = os.path.join(self.output_dir, "config.yaml")
        with open(config_path, 'w') as f:
            from omegaconf import OmegaConf
            f.write(OmegaConf.to_yaml(self.cfg))

    def train(self):
        """Run training loop."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        start_time = time.time()
        total_steps = 0
        log_interval = self.cfg.logging.log_interval
        save_interval = self.cfg.logging.save_interval

        for episode in range(self.start_episode, self.cfg.experiment.num_episodes):
            metrics = self._train_episode(episode)
            total_steps += metrics['steps'] * self.env.num_agents

            # Log metrics
            self._log_metrics(episode, metrics, total_steps, start_time)

            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1)

            # Save best model
            if metrics['reward'] > self.best_reward:
                self.best_reward = metrics['reward']
                self._save_checkpoint(episode + 1, is_best=True)

            # Print progress
            if (episode + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                fps = total_steps / elapsed
                print(
                    f"Ep {episode+1:4d}/{self.cfg.experiment.num_episodes} | "
                    f"R={metrics['reward']:7.2f} | "
                    f"Cost={metrics['cost']:4.0f} | "
                    f"FormErr={metrics['formation_error']:.4f} | "
                    f"Coll={metrics['collisions']:2d} | "
                    f"FPS={fps:.0f}"
                )

        # Save final model
        self._save_checkpoint(self.cfg.experiment.num_episodes, is_final=True)

        # Save metrics
        self._save_metrics()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        if self.use_wandb:
            import wandb
            wandb.finish()

    def _train_episode(self, episode: int) -> Dict[str, float]:
        """Run one training episode."""
        # Reset
        obs, share_obs, _ = self.env.reset(seed=self.cfg.experiment.seed + episode)
        constraint_info = self.env.get_constraint_info()
        self.safety.reset(constraint_info)
        self.safety.update(constraint_info)
        self.buffer.set_first_obs(obs, share_obs)

        # Episode metrics
        ep_reward, ep_cost, ep_form_err = 0.0, 0.0, 0.0
        ep_min_dist, ep_collisions = float('inf'), 0

        for step in range(self.cfg.experiment.max_steps):
            # Get actions
            alphas, log_probs = self.algo.get_actions(obs)
            values = self.algo.get_values(share_obs)

            # Safety projection
            constraint_info = self.env.get_constraint_info()
            safe_actions = self.safety.project(
                alphas, constraint_info, dt=self.cfg.env.dt
            )

            # Step environment
            next_obs, next_share_obs, rewards, costs, dones, infos, _ = \
                self.env.step(safe_actions)

            # Store experience
            masks = (~dones).astype(np.float32).reshape(-1, 1)
            self.buffer.insert(
                next_obs, next_share_obs, alphas, log_probs,
                values, rewards, costs, masks
            )

            obs, share_obs = next_obs, next_share_obs

            # Accumulate metrics
            ep_reward += rewards[0, 0]
            ep_cost += costs[0, 0]
            ep_form_err += infos[0]["formation_error"]
            ep_min_dist = min(ep_min_dist, infos[0]["min_inter_dist"])
            ep_collisions += infos[0]["collisions"]

            if dones.all():
                break

        # Update policy
        last_values = self.algo.get_values(share_obs)
        self.buffer.compute_returns_and_advantages(last_values)
        self.algo.update(self.buffer)
        self.buffer.after_update()

        # Return metrics
        return {
            'reward': ep_reward,
            'cost': ep_cost,
            'formation_error': ep_form_err / (step + 1),
            'min_dist': ep_min_dist,
            'collisions': ep_collisions,
            'steps': step + 1
        }

    def _log_metrics(self, episode: int, metrics: Dict, total_steps: int, start_time: float):
        """Log metrics to history and WandB."""
        self.metrics['episode'].append(episode + 1)
        self.metrics['reward'].append(metrics['reward'])
        self.metrics['cost'].append(metrics['cost'])
        self.metrics['formation_error'].append(metrics['formation_error'])
        self.metrics['min_dist'].append(metrics['min_dist'])
        self.metrics['collisions'].append(metrics['collisions'])

        if self.use_wandb:
            import wandb
            wandb.log({
                "train/reward": metrics['reward'],
                "train/cost": metrics['cost'],
                "train/formation_error": metrics['formation_error'],
                "train/min_dist": metrics['min_dist'],
                "train/collisions": metrics['collisions'],
                "train/episode_length": metrics['steps'],
                "episode": episode + 1,
            })

    def _save_checkpoint(self, episode: int, is_best: bool = False, is_final: bool = False):
        """Save checkpoint."""
        if is_best:
            path = os.path.join(self.checkpoint_dir, "best_model.pt")
        elif is_final:
            path = os.path.join(self.checkpoint_dir, "final_model.pt")
        else:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_ep{episode}.pt")

        checkpoint = {
            'episode': episode,
            'actor': self.algo.actor.state_dict(),
            'critic': self.algo.critic.state_dict(),
            'metrics': self.metrics,
        }
        torch.save(checkpoint, path)

        if self.use_wandb and (is_best or is_final):
            import wandb
            artifact = wandb.Artifact(
                f"model_{'best' if is_best else 'final'}",
                type="model"
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def _save_metrics(self):
        """Save metrics to file."""
        metrics_path = os.path.join(self.results_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def load_checkpoint(self, path: str):
        """Load checkpoint to resume training."""
        checkpoint = torch.load(path, map_location=self.device)
        self.algo.actor.load_state_dict(checkpoint['actor'])
        self.algo.critic.load_state_dict(checkpoint['critic'])
        self.start_episode = checkpoint.get('episode', 0)
        self.metrics = checkpoint.get('metrics', self.metrics)
        print(f"Loaded checkpoint from {path} (episode {self.start_episode})")

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policy."""
        self.algo.eval_mode()
        results = []

        for ep in range(num_episodes):
            obs, share_obs, _ = self.env.reset(
                seed=self.cfg.experiment.seed + 10000 + ep
            )
            constraint_info = self.env.get_constraint_info()
            self.safety.reset(constraint_info)

            ep_reward, ep_collisions = 0.0, 0

            for step in range(self.cfg.experiment.max_steps):
                alphas, _ = self.algo.get_actions(obs, deterministic=True)
                constraint_info = self.env.get_constraint_info()
                safe_actions = self.safety.project(
                    alphas, constraint_info, dt=self.cfg.env.dt
                )
                obs, share_obs, rewards, _, dones, infos, _ = self.env.step(safe_actions)

                ep_reward += rewards[0, 0]
                ep_collisions += infos[0]["collisions"]

                if dones.all():
                    break

            results.append({'reward': ep_reward, 'collisions': ep_collisions})

        self.algo.train_mode()

        return {
            'avg_reward': np.mean([r['reward'] for r in results]),
            'avg_collisions': np.mean([r['collisions'] for r in results]),
            'total_collisions': sum([r['collisions'] for r in results]),
        }
