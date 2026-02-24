#!/usr/bin/env python3
"""
COSMOS - Test All Environments

Quick test to verify all environments work correctly.

Usage:
    python test_all_envs.py
"""

import numpy as np
import sys

def test_environment(env_name: str, env_class, env_kwargs: dict = None):
    """Test a single environment."""
    env_kwargs = env_kwargs or {}

    print(f"\n{'='*60}")
    print(f"Testing: {env_name}")
    print('='*60)

    try:
        # Create environment
        env = env_class(**env_kwargs)
        print(f"  ✓ Created environment")
        print(f"    - num_agents: {env.num_agents}")
        print(f"    - obs_dim: {env.get_obs_dim()}")
        print(f"    - act_dim: {env.get_act_dim()}")
        print(f"    - share_obs_dim: {env.get_share_obs_dim()}")

        # Reset
        obs, share_obs, info = env.reset(seed=42)
        print(f"  ✓ Reset environment")
        print(f"    - obs shape: {obs.shape}")
        print(f"    - share_obs shape: {share_obs.shape}")

        # Step
        actions = np.random.uniform(-1, 1, (env.num_agents, env.get_act_dim()))
        next_obs, next_share, rewards, costs, dones, infos, truncated = env.step(actions)
        print(f"  ✓ Step environment")
        print(f"    - rewards: {rewards.flatten()}")
        print(f"    - costs: {costs.flatten()}")

        # Get constraint info
        constraint_info = env.get_constraint_info()
        print(f"  ✓ Get constraint info")
        print(f"    - positions shape: {constraint_info['positions'].shape}")

        # Run a few more steps
        for _ in range(10):
            actions = np.random.uniform(-1, 1, (env.num_agents, env.get_act_dim()))
            env.step(actions)
        print(f"  ✓ Ran 10 additional steps")

        # Close
        env.close()
        print(f"  ✓ Closed environment")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  COSMOS - Environment Test Suite")
    print("=" * 60)

    results = {}

    # ==========================================================================
    # Test built-in environments
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  BUILT-IN ENVIRONMENTS")
    print("=" * 60)

    # Formation Navigation
    try:
        from cosmos.envs.formation_nav import FormationNavEnv
        results['formation_nav'] = test_environment(
            'formation_nav',
            FormationNavEnv,
            {'cfg': {'num_agents': 4, 'arena_size': 5.0}}
        )
    except ImportError as e:
        print(f"\n✗ formation_nav: Import error - {e}")
        results['formation_nav'] = False

    # E-puck Simulation
    try:
        from cosmos.envs.webots_wrapper import EpuckSimEnv
        results['epuck_sim'] = test_environment(
            'epuck_sim',
            EpuckSimEnv,
            {'num_agents': 4}
        )
    except ImportError as e:
        print(f"\n✗ epuck_sim: Import error - {e}")
        results['epuck_sim'] = False

    # ==========================================================================
    # Test optional environments
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  OPTIONAL ENVIRONMENTS")
    print("=" * 60)

    # Safety-Gym
    try:
        import safety_gymnasium
        from cosmos.envs.safety_gym_wrapper import SafetyGymWrapper
        results['safety_gym'] = test_environment(
            'safety_gym',
            SafetyGymWrapper,
            {'env_id': 'SafetyPointGoal1-v0'}
        )
    except ImportError as e:
        print(f"\n⚠ safety_gym: Not installed - {e}")
        results['safety_gym'] = None

    # VMAS
    try:
        import vmas
        from cosmos.envs.vmas_wrapper import VMASWrapper
        results['vmas'] = test_environment(
            'vmas',
            VMASWrapper,
            {'scenario': 'navigation', 'num_agents': 4}
        )
    except ImportError as e:
        print(f"\n⚠ vmas: Not installed - {e}")
        results['vmas'] = None

    # MuJoCo
    try:
        import mujoco
        from cosmos.envs.mujoco_wrapper import MuJoCoWrapper
        results['mujoco'] = test_environment(
            'mujoco',
            MuJoCoWrapper,
            {'env_id': 'Ant-v4', 'num_agents': 2}
        )
    except ImportError as e:
        print(f"\n⚠ mujoco: Not installed - {e}")
        results['mujoco'] = None

    # ==========================================================================
    # Test algorithms
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  ALGORITHMS")
    print("=" * 60)

    algo_results = {}

    # MAPPO
    try:
        from cosmos.algos.mappo import MAPPO
        algo = MAPPO(obs_dim=10, share_obs_dim=20, act_dim=2, num_agents=4)
        obs = np.random.randn(4, 10)
        actions, log_probs = algo.get_actions(obs)
        print(f"\n✓ MAPPO: actions shape {actions.shape}")
        algo_results['mappo'] = True
    except Exception as e:
        print(f"\n✗ MAPPO: {e}")
        algo_results['mappo'] = False

    # QMIX
    try:
        from cosmos.algos.qmix import QMIX
        algo = QMIX(obs_dim=10, share_obs_dim=20, act_dim=2, num_agents=4, n_actions=5)
        obs = np.random.randn(4, 10)
        actions, q_values = algo.get_actions(obs)
        print(f"✓ QMIX: actions shape {actions.shape}")
        algo_results['qmix'] = True
    except Exception as e:
        print(f"✗ QMIX: {e}")
        algo_results['qmix'] = False

    # MADDPG
    try:
        from cosmos.algos.maddpg import MADDPG
        algo = MADDPG(obs_dim=10, share_obs_dim=20, act_dim=2, num_agents=4)
        obs = np.random.randn(4, 10)
        actions, _ = algo.get_actions(obs)
        print(f"✓ MADDPG: actions shape {actions.shape}")
        algo_results['maddpg'] = True
    except Exception as e:
        print(f"✗ MADDPG: {e}")
        algo_results['maddpg'] = False

    # ==========================================================================
    # Test safety filters
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  SAFETY FILTERS")
    print("=" * 60)

    safety_results = {}

    # CBF
    try:
        from cosmos.safety.cosmos_filter import CBFFilter
        cbf = CBFFilter(
            env_cfg={'arena_size': 5.0, 'num_agents': 4},
            safety_cfg=None
        )
        constraint_info = {
            'positions': np.random.randn(4, 3),
            'velocities': np.random.randn(4, 3),
        }
        actions = np.random.randn(4, 2)
        safe_actions = cbf.project(actions, constraint_info)
        print(f"\n✓ CBF: safe_actions shape {safe_actions.shape}")
        safety_results['cbf'] = True
    except Exception as e:
        print(f"\n✗ CBF: {e}")
        safety_results['cbf'] = False

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print("\nEnvironments:")
    for name, status in results.items():
        if status is True:
            print(f"  ✓ {name}")
        elif status is False:
            print(f"  ✗ {name}")
        else:
            print(f"  - {name} (not installed)")

    print("\nAlgorithms:")
    for name, status in algo_results.items():
        print(f"  {'✓' if status else '✗'} {name}")

    print("\nSafety Filters:")
    for name, status in safety_results.items():
        print(f"  {'✓' if status else '✗'} {name}")

    # Check if core components work
    core_ok = (
        results.get('formation_nav', False) and
        results.get('epuck_sim', False) and
        algo_results.get('mappo', False) and
        safety_results.get('cbf', False)
    )

    print("\n" + "=" * 60)
    if core_ok:
        print("  ✓ All core components working!")
        print("  Ready to run experiments.")
    else:
        print("  ✗ Some core components failed.")
        print("  Please check errors above.")
    print("=" * 60)

    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
