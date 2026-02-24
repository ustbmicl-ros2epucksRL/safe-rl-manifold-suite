"""
Checkpoint utilities for saving and loading models.
"""

from typing import Dict, Any, Optional
import os
import torch


def save_checkpoint(
    path: str,
    episode: int,
    algo: Any,
    metrics: Optional[Dict] = None,
    config: Optional[Dict] = None,
    **extra
) -> str:
    """
    Save training checkpoint.

    Args:
        path: Path to save checkpoint.
        episode: Current episode number.
        algo: Algorithm with actor/critic networks.
        metrics: Training metrics history.
        config: Configuration dict.
        **extra: Additional data to save.

    Returns:
        Path to saved checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'episode': episode,
        'actor': algo.actor.state_dict(),
        'critic': algo.critic.state_dict(),
    }

    # Add cost critic if available
    if hasattr(algo, 'cost_critic'):
        checkpoint['cost_critic'] = algo.cost_critic.state_dict()

    # Add optional data
    if metrics is not None:
        checkpoint['metrics'] = metrics
    if config is not None:
        checkpoint['config'] = config

    checkpoint.update(extra)

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    path: str,
    algo: Any,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint.
        algo: Algorithm to load weights into.
        device: Device to load tensors to.

    Returns:
        Checkpoint dict with episode, metrics, etc.
    """
    checkpoint = torch.load(path, map_location=device)

    # Load network weights
    algo.actor.load_state_dict(checkpoint['actor'])
    algo.critic.load_state_dict(checkpoint['critic'])

    if 'cost_critic' in checkpoint and hasattr(algo, 'cost_critic'):
        algo.cost_critic.load_state_dict(checkpoint['cost_critic'])

    return checkpoint
