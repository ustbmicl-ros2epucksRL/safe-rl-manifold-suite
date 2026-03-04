#!/usr/bin/env python3
"""
Generate Webots experiment figures from real trajectory data.

Produces:
  - figures/ros_setup.png: Arena setup with 6 obstacles
  - figures/trajectory.png: Trajectory comparison (safe vs unsafe)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
RESULTS_PATH = os.path.join(SCRIPT_DIR, '..', 'results_webots_real', 'webots_results_with_traj.json')

OBSTACLE_RADIUS = 0.10
ROBOT_RADIUS = 0.035


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def plot_setup(obstacles):
    """Plot E-puck arena setup with 6 obstacles."""
    fig, ax = plt.subplots(figsize=(5, 5))

    # Arena boundary
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False,
                               edgecolor='black', linewidth=2.5))

    # Grid
    for i in np.arange(-1, 1.1, 0.5):
        ax.axhline(i, color='gray', alpha=0.15, linewidth=0.5)
        ax.axvline(i, color='gray', alpha=0.15, linewidth=0.5)

    # Obstacles
    for idx, obs in enumerate(obstacles):
        ox, oy = obs[0], obs[1]
        ax.add_patch(Circle((ox, oy), OBSTACLE_RADIUS, fill=True,
                            color='#d32f2f', alpha=0.85, edgecolor='#8b0000',
                            linewidth=1.5))
        ax.text(ox, oy - OBSTACLE_RADIUS - 0.07, f'r={OBSTACLE_RADIUS}m',
                ha='center', fontsize=10, color='#8b0000')

    # Robot (example position)
    robot_pos = (-0.5, -0.5)
    ax.add_patch(Circle(robot_pos, ROBOT_RADIUS * 2.5, fill=True,
                        color='#1976d2', alpha=0.9, edgecolor='black', linewidth=1))
    ax.annotate('', xy=(robot_pos[0] + 0.07, robot_pos[1]),
                xytext=robot_pos,
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax.text(robot_pos[0], robot_pos[1] - 0.14, 'E-puck', ha='center',
            fontsize=13, fontweight='bold')

    # Goal
    goal = (0.5, 0.5)
    ax.scatter(*goal, color='gold', s=300, marker='*', edgecolors='black',
              linewidths=0.8, zorder=10)
    ax.text(goal[0] + 0.09, goal[1], 'Goal', fontsize=13, fontweight='bold')

    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_title('E-puck Arena (2m × 2m)', fontsize=15, fontweight='bold')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=12)

    # Info box
    ax.annotate(f'Obstacles: {len(obstacles)}\n'
                f'Robot: E-puck (⌀70mm)\nSensors: Odom + IMU',
                xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=11, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                          alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'ros_setup.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_trajectory(data):
    """Plot trajectory comparison from real Webots data."""
    obstacles = data['obstacles']
    configs = data['trial_configs']
    safe_results = data['safe']
    unsafe_results = data['unsafe']

    # Select representative trials to plot (not all 20, for readability)
    # Pick: a few normal, one that shows safety filter detour, one with collision
    trial_indices = select_trials(safe_results, unsafe_results)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trial_indices)))

    for ax_idx, (ax, mode, results) in enumerate([
        (axes[0], 'safe', safe_results),
        (axes[1], 'unsafe', unsafe_results),
    ]):
        # Arena
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False,
                                   edgecolor='black', linewidth=2))

        # Obstacles
        for obs in obstacles:
            ax.add_patch(Circle((obs[0], obs[1]), OBSTACLE_RADIUS, fill=True,
                               color='#d32f2f', alpha=0.7, edgecolor='#8b0000',
                               linewidth=1))

        # Trajectories
        for i, tidx in enumerate(trial_indices):
            if tidx >= len(results):
                continue
            r = results[tidx]
            traj_gt = np.array(r['traj_gt'])
            start = configs[tidx][0]
            goal = configs[tidx][1]
            c = colors[i]

            # Ground truth trajectory
            ax.plot(traj_gt[:, 0], traj_gt[:, 1], '-', color=c,
                    linewidth=1.8, alpha=0.8)

            # EKF estimate (safe mode only)
            if mode == 'safe' and 'traj_est' in r:
                traj_est = np.array(r['traj_est'])
                ax.plot(traj_est[:, 0], traj_est[:, 1], '--', color=c,
                        linewidth=0.8, alpha=0.35)

            # Start marker
            ax.plot(start[0], start[1], 'o', color=c, markersize=6,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)

            # Goal marker
            ax.scatter(goal[0], goal[1], color='gold', s=80, marker='*',
                       edgecolors='black', linewidths=0.5, zorder=6)

            # Collision marker
            if r['collisions'] > 0:
                # Find closest point to any obstacle
                for pt in traj_gt:
                    for obs in obstacles:
                        d = np.linalg.norm(pt - np.array(obs[:2]))
                        if d < OBSTACLE_RADIUS + ROBOT_RADIUS + 0.01:
                            ax.plot(pt[0], pt[1], 'x', color='red',
                                    markersize=10, markeredgewidth=2.5, zorder=7)
                            break

        title = '(a) With Safety Filter + EKF' if mode == 'safe' else '(b) Without Safety Filter'
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=13)
        ax.set_ylabel('Y (m)', fontsize=13)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=11)

    # Legend on left panel
    axes[0].plot([], [], 'k-', linewidth=1.8, label='Ground Truth')
    axes[0].plot([], [], 'k--', linewidth=0.8, alpha=0.4, label='EKF Estimate')
    axes[0].scatter([], [], color='gold', s=60, marker='*',
                    edgecolors='black', label='Goal')
    axes[0].legend(loc='upper left', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'trajectory.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def select_trials(safe_results, unsafe_results):
    """Select 6 representative trials for visualization."""
    indices = []

    # 1. A trial where both modes succeed normally (short path)
    for i in range(len(safe_results)):
        if (safe_results[i]['success'] and safe_results[i]['path_length'] < 1.2
                and i < len(unsafe_results) and unsafe_results[i]['success']):
            indices.append(i)
            if len(indices) >= 2:
                break

    # 2. A trial where safe mode takes a longer (detour) path
    for i in range(len(safe_results)):
        if (safe_results[i]['success'] and safe_results[i]['path_length'] > 1.8
                and i not in indices):
            indices.append(i)
            if len(indices) >= 4:
                break

    # 3. A trial with collision in unsafe mode
    for i in range(len(unsafe_results)):
        if unsafe_results[i]['collisions'] > 0 and i not in indices:
            indices.append(i)
            break

    # 4. Fill to 6 with varied trials
    for i in range(len(safe_results)):
        if i not in indices and safe_results[i]['success']:
            indices.append(i)
            if len(indices) >= 6:
                break

    return sorted(indices[:6])


def main():
    data = load_data()
    print(f"Loaded data: {len(data['safe'])} safe, {len(data['unsafe'])} unsafe trials")
    print(f"Obstacles: {len(data['obstacles'])}")

    plot_setup(data['obstacles'])
    plot_trajectory(data)


if __name__ == '__main__':
    main()
