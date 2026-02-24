#!/usr/bin/env python3
"""
COSMOS Training Entry Point

Main entry point for training with Hydra configuration.

Usage:
    # Default training
    python -m cosmos.train

    # Change algorithm
    python -m cosmos.train algo=qmix

    # Change environment
    python -m cosmos.train env=mpe

    # Override parameters
    python -m cosmos.train experiment.num_episodes=500 env.num_agents=6

    # Disable safety filter
    python -m cosmos.train safety=none

    # Enable WandB
    python -m cosmos.train logging.use_wandb=true

    # Multi-run sweep
    python -m cosmos.train -m algo=mappo,qmix env.num_agents=4,6
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig, OmegaConf

# Import registries to trigger component registration
from cosmos.registry import ENV_REGISTRY, ALGO_REGISTRY, SAFETY_REGISTRY


def print_banner():
    """Print COSMOS banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗ ██████╗ ███████╗███╗   ███╗ ██████╗ ███████╗                        ║
║  ██╔════╝██╔═══██╗██╔════╝████╗ ████║██╔═══██╗██╔════╝                        ║
║  ██║     ██║   ██║███████╗██╔████╔██║██║   ██║███████╗                        ║
║  ██║     ██║   ██║╚════██║██║╚██╔╝██║██║   ██║╚════██║                        ║
║  ╚██████╗╚██████╔╝███████║██║ ╚═╝ ██║╚██████╔╝███████║                        ║
║   ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝                        ║
║                                                                               ║
║   COordinated Safety On Manifold for multi-agent Systems                     ║
║   Safe Multi-Agent Reinforcement Learning Framework                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_available_components():
    """Print available components."""
    # Trigger registration by importing modules
    try:
        from cosmos.envs import formation_nav
        from cosmos.algos import mappo
        from cosmos.safety import cosmos_filter
    except ImportError:
        pass

    print("\nAvailable Components:")
    print(f"  Environments:    {ENV_REGISTRY.list()}")
    print(f"  Algorithms:      {ALGO_REGISTRY.list()}")
    print(f"  Safety Filters:  {SAFETY_REGISTRY.list()}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    print_banner()

    # Print config
    print("\nConfiguration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Import here to avoid circular imports
    from cosmos.trainer import Trainer

    # Create trainer
    trainer = Trainer(cfg)

    # Load checkpoint if resuming
    if cfg.checkpoint.resume_from:
        trainer.load_checkpoint(cfg.checkpoint.resume_from)

    # Train
    trainer.train()

    # Evaluate
    print("\nRunning Evaluation...")
    eval_results = trainer.evaluate(num_episodes=10)
    print(f"  Average Reward:     {eval_results['avg_reward']:.2f}")
    print(f"  Average Collisions: {eval_results['avg_collisions']:.2f}")
    print(f"  Total Collisions:   {eval_results['total_collisions']}")


if __name__ == "__main__":
    main()
