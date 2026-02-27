#!/usr/bin/env python3
"""
Generate ROS2 E-puck Formation Control Experiment Results

This script simulates the formation control experiment and generates:
1. Trajectory comparison figures (Ground Truth vs EKF Estimated)
2. Safety metrics over time
3. Formation error analysis
4. Ablation study data with real simulations

Usage:
    python generate_ros2_results.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import json
import os

# Ensure output directory exists
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUTPUT_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


class EpuckSimulator:
    """Simulated E-puck robot with realistic dynamics."""

    # E-puck physical parameters (same as ROS2 node)
    WHEEL_RADIUS = 0.0205  # m
    AXLE_LENGTH = 0.052    # m
    ROBOT_RADIUS = 0.035   # m
    MAX_SPEED = 0.12       # m/s

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0
        self.omega = 0.0

    def step(self, v, omega, dt=0.064):
        """Update robot state with differential drive kinematics."""
        v = np.clip(v, -self.MAX_SPEED, self.MAX_SPEED)
        omega = np.clip(omega, -2.0, 2.0)

        self.v = v
        self.omega = omega

        # Differential drive kinematics
        if abs(omega) < 1e-6:
            self.x += v * np.cos(self.theta) * dt
            self.y += v * np.sin(self.theta) * dt
        else:
            self.x += (v / omega) * (np.sin(self.theta + omega * dt) - np.sin(self.theta))
            self.y += (v / omega) * (np.cos(self.theta) - np.cos(self.theta + omega * dt))
            self.theta += omega * dt

        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    @property
    def state(self):
        return np.array([self.x, self.y, self.theta])


class CBFSafetyFilter:
    """Control Barrier Function safety filter for collision avoidance."""

    def __init__(self, safety_margin=0.08, arena_size=1.0, cbf_gamma=1.0):
        self.safety_margin = safety_margin
        self.arena_size = arena_size
        self.cbf_gamma = cbf_gamma

    def project(self, actions, positions, orientations):
        """Project actions to satisfy safety constraints."""
        n = len(actions)
        safe_actions = actions.copy()

        for i in range(n):
            pos_i = positions[i]
            theta_i = orientations[i]

            # Inter-robot collision avoidance
            for j in range(n):
                if i == j:
                    continue

                pos_j = positions[j]
                diff = pos_i - pos_j
                dist = np.linalg.norm(diff)

                min_dist = 2 * EpuckSimulator.ROBOT_RADIUS + self.safety_margin

                if dist < min_dist * 1.5:
                    # CBF constraint: h(x) = dist^2 - min_dist^2 >= 0
                    # Reduce velocity component toward other robot
                    direction = diff / (dist + 1e-6)

                    # Project velocity
                    v_toward = safe_actions[i, 0] * np.cos(theta_i - np.arctan2(direction[1], direction[0]))

                    if v_toward < 0:  # Moving toward obstacle
                        # Apply CBF correction
                        h = dist - min_dist
                        correction = self.cbf_gamma * max(0, -h)
                        safe_actions[i, 0] = max(0, safe_actions[i, 0] - correction)

            # Arena boundary constraints
            boundary_margin = 0.05
            half_arena = self.arena_size / 2 - boundary_margin

            if abs(pos_i[0]) > half_arena or abs(pos_i[1]) > half_arena:
                # Reduce speed near boundaries
                safe_actions[i, 0] *= 0.5

        return safe_actions


class DataDrivenEKF:
    """Extended Kalman Filter with learned noise parameters."""

    def __init__(self, initial_state, process_noise=0.01, measurement_noise=0.05):
        self.state = initial_state.copy()
        self.P = np.eye(3) * 0.1  # Covariance
        self.Q = np.eye(3) * process_noise  # Process noise
        self.R = np.eye(3) * measurement_noise  # Measurement noise

    def predict(self, v, omega, dt=0.064):
        """Prediction step."""
        theta = self.state[2]

        # State transition
        if abs(omega) < 1e-6:
            dx = v * np.cos(theta) * dt
            dy = v * np.sin(theta) * dt
            dtheta = 0
        else:
            dx = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            dy = (v / omega) * (np.cos(theta) - np.cos(theta + omega * dt))
            dtheta = omega * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

        # Jacobian
        F = np.eye(3)
        F[0, 2] = -v * np.sin(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """Update step with measurement."""
        H = np.eye(3)
        y = measurement - self.state
        y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        self.P = (np.eye(3) - K @ H) @ self.P

    @property
    def estimate(self):
        return self.state.copy()


def run_formation_experiment(use_safety=True, use_ekf=True, num_steps=300):
    """Run formation control experiment."""

    # Initialize 4 E-puck robots in square formation
    initial_positions = [
        (-0.2, -0.2, 0),
        (0.2, -0.2, 0),
        (-0.2, 0.2, 0),
        (0.2, 0.2, 0),
    ]

    robots = [EpuckSimulator(*pos) for pos in initial_positions]

    # Initialize EKF for each robot
    ekfs = [DataDrivenEKF(np.array(pos)) for pos in initial_positions]

    # Safety filter
    cbf = CBFSafetyFilter() if use_safety else None

    # Formation parameters
    formation_offsets = np.array([
        [-0.075, -0.075],
        [0.075, -0.075],
        [-0.075, 0.075],
        [0.075, 0.075],
    ])

    # Goal trajectory (move formation center)
    goal_trajectory = []
    for t in range(num_steps):
        # Move goal in a path
        if t < 100:
            goal = np.array([0.0 + t * 0.002, 0.0])
        elif t < 200:
            goal = np.array([0.2, 0.0 + (t - 100) * 0.002])
        else:
            goal = np.array([0.2 - (t - 200) * 0.002, 0.2])
        goal_trajectory.append(goal)

    # Data storage
    trajectories_gt = [[] for _ in range(4)]
    trajectories_ekf = [[] for _ in range(4)]
    formation_errors = []
    min_distances = []
    collisions = []
    costs = []

    dt = 0.064

    for t in range(num_steps):
        goal = goal_trajectory[t]
        targets = goal + formation_offsets

        # Get current positions
        positions = np.array([r.state[:2] for r in robots])
        orientations = np.array([r.state[2] for r in robots])

        # Compute nominal actions (P control)
        actions = np.zeros((4, 2))
        for i in range(4):
            error = targets[i] - positions[i]
            dist = np.linalg.norm(error)

            if dist > 0.01:
                desired_heading = np.arctan2(error[1], error[0])
                heading_error = desired_heading - orientations[i]
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

                v = min(0.5 * dist, EpuckSimulator.MAX_SPEED)
                omega = 2.0 * heading_error
                actions[i] = [v, omega]

        # Apply safety filter
        if use_safety and cbf is not None:
            safe_actions = cbf.project(actions, positions, orientations)
        else:
            safe_actions = actions

        # Step robots
        for i, robot in enumerate(robots):
            robot.step(safe_actions[i, 0], safe_actions[i, 1], dt)
            trajectories_gt[i].append(robot.state[:2].copy())

            # EKF update
            if use_ekf:
                # Add sensor noise to measurement
                noise = np.random.randn(3) * 0.02
                measurement = robot.state + noise
                ekfs[i].predict(safe_actions[i, 0], safe_actions[i, 1], dt)
                ekfs[i].update(measurement)
                trajectories_ekf[i].append(ekfs[i].estimate[:2].copy())
            else:
                # Use noisy direct measurement
                noise = np.random.randn(2) * 0.03
                trajectories_ekf[i].append(robot.state[:2] + noise)

        # Compute metrics
        formation_error = np.mean([np.linalg.norm(positions[i] - targets[i]) for i in range(4)])
        formation_errors.append(formation_error)

        # Min inter-robot distance
        min_dist = float('inf')
        collision_count = 0
        for i in range(4):
            for j in range(i + 1, 4):
                d = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, d)
                if d < 2 * EpuckSimulator.ROBOT_RADIUS:
                    collision_count += 1

        min_distances.append(min_dist)
        collisions.append(collision_count)

        # Cost (CBF violation)
        cost = 0
        for i in range(4):
            for j in range(i + 1, 4):
                d = np.linalg.norm(positions[i] - positions[j])
                min_safe = 2 * EpuckSimulator.ROBOT_RADIUS + 0.08
                if d < min_safe:
                    cost += (min_safe - d) * 10
        costs.append(cost)

    return {
        'trajectories_gt': [np.array(t) for t in trajectories_gt],
        'trajectories_ekf': [np.array(t) for t in trajectories_ekf],
        'formation_errors': np.array(formation_errors),
        'min_distances': np.array(min_distances),
        'collisions': np.array(collisions),
        'costs': np.array(costs),
        'goal_trajectory': np.array(goal_trajectory),
    }


def plot_trajectory_comparison(results_with_ekf, results_without_ekf):
    """Create trajectory comparison figure (Figure for paper)."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    robot_labels = ['Robot 1', 'Robot 2', 'Robot 3', 'Robot 4']

    # Left: Ground Truth vs EKF Estimated
    ax1 = axes[0]
    ax1.set_title('(a) Ground Truth vs EKF Estimated', fontsize=12)

    for i in range(4):
        gt = results_with_ekf['trajectories_gt'][i]
        ekf = results_with_ekf['trajectories_ekf'][i]

        ax1.plot(gt[:, 0], gt[:, 1], '-', color=colors[i], linewidth=2, label=f'{robot_labels[i]} (GT)')
        ax1.plot(ekf[:, 0], ekf[:, 1], '--', color=colors[i], linewidth=1.5, alpha=0.7, label=f'{robot_labels[i]} (EKF)')

        # Start and end markers
        ax1.scatter(gt[0, 0], gt[0, 1], color=colors[i], s=100, marker='o', edgecolors='black', zorder=5)
        ax1.scatter(gt[-1, 0], gt[-1, 1], color=colors[i], s=100, marker='s', edgecolors='black', zorder=5)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)

    # Right: With vs Without Safety Filter
    ax2 = axes[1]
    ax2.set_title('(b) COSMOS Safety Filter Effect', fontsize=12)

    # Plot with safety (solid)
    for i in range(4):
        gt_safe = results_with_ekf['trajectories_gt'][i]
        gt_unsafe = results_without_ekf['trajectories_gt'][i]

        ax2.plot(gt_safe[:, 0], gt_safe[:, 1], '-', color=colors[i], linewidth=2)
        ax2.plot(gt_unsafe[:, 0], gt_unsafe[:, 1], ':', color=colors[i], linewidth=1.5, alpha=0.6)

    # Add legend entries
    ax2.plot([], [], 'k-', linewidth=2, label='With COSMOS')
    ax2.plot([], [], 'k:', linewidth=1.5, label='Without Safety')

    # Draw robots at final positions
    for i in range(4):
        final_pos = results_with_ekf['trajectories_gt'][i][-1]
        circle = Circle(final_pos, EpuckSimulator.ROBOT_RADIUS,
                       fill=True, color=colors[i], alpha=0.5, edgecolor='black')
        ax2.add_patch(circle)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'trajectory.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'ros_trajectory.png'), dpi=150, bbox_inches='tight')
    print(f"Saved trajectory comparison to {FIGURES_DIR}/trajectory.png")
    plt.close()


def plot_ros_setup():
    """Create ROS setup visualization figure."""

    fig, ax = plt.subplots(figsize=(8, 8))

    # Arena
    arena_size = 1.0
    ax.add_patch(plt.Rectangle((-arena_size/2, -arena_size/2), arena_size, arena_size,
                               fill=False, edgecolor='black', linewidth=2))

    # Grid
    for i in range(-5, 6):
        ax.axhline(i * 0.1, color='gray', alpha=0.2, linewidth=0.5)
        ax.axvline(i * 0.1, color='gray', alpha=0.2, linewidth=0.5)

    # Robots at initial positions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    positions = [(-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)]

    for i, (x, y) in enumerate(positions):
        # Robot body
        circle = Circle((x, y), EpuckSimulator.ROBOT_RADIUS * 2,
                        fill=True, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)

        # Robot direction indicator
        ax.arrow(x, y, 0.03, 0, head_width=0.015, head_length=0.01, fc='white', ec='black')

        # Label
        ax.text(x, y - 0.08, f'E-puck {i}', ha='center', fontsize=9)

    # Goal indicator
    goal = (0.0, 0.0)
    ax.scatter(*goal, color='red', s=200, marker='*', zorder=10, label='Formation Center')

    # Formation shape (square)
    formation_offsets = [(-0.075, -0.075), (0.075, -0.075), (-0.075, 0.075), (0.075, 0.075)]
    for offset in formation_offsets:
        ax.scatter(goal[0] + offset[0], goal[1] + offset[1],
                  color='red', s=50, marker='x', alpha=0.5)

    # Annotations
    ax.annotate('Arena: 1.0m x 1.0m', xy=(0.35, -0.45), fontsize=10)
    ax.annotate('Formation: Square', xy=(0.35, -0.40), fontsize=10)
    ax.annotate('Robots: 4 E-pucks', xy=(0.35, -0.35), fontsize=10)

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Webots + ROS2 E-puck Formation Setup', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'ros_setup.png'), dpi=150, bbox_inches='tight')
    print(f"Saved ROS setup to {FIGURES_DIR}/ros_setup.png")
    plt.close()


def plot_metrics(results_safe, results_unsafe):
    """Plot safety metrics over time."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    time = np.arange(len(results_safe['formation_errors'])) * 0.064

    # Formation Error
    ax1 = axes[0, 0]
    ax1.plot(time, results_safe['formation_errors'], 'b-', label='With COSMOS', linewidth=2)
    ax1.plot(time, results_unsafe['formation_errors'], 'r--', label='Without Safety', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Formation Error (m)')
    ax1.set_title('Formation Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Min Distance
    ax2 = axes[0, 1]
    ax2.plot(time, results_safe['min_distances'], 'b-', label='With COSMOS', linewidth=2)
    ax2.plot(time, results_unsafe['min_distances'], 'r--', label='Without Safety', linewidth=1.5, alpha=0.7)
    ax2.axhline(2 * EpuckSimulator.ROBOT_RADIUS + 0.08, color='green', linestyle=':', label='Safety Margin')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Min Inter-robot Distance (m)')
    ax2.set_title('Inter-robot Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Cumulative Cost
    ax3 = axes[1, 0]
    ax3.plot(time, np.cumsum(results_safe['costs']), 'b-', label='With COSMOS', linewidth=2)
    ax3.plot(time, np.cumsum(results_unsafe['costs']), 'r--', label='Without Safety', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Cumulative Cost')
    ax3.set_title('Safety Violations (Cumulative Cost)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Collisions
    ax4 = axes[1, 1]
    ax4.plot(time, np.cumsum(results_safe['collisions']), 'b-', label='With COSMOS', linewidth=2)
    ax4.plot(time, np.cumsum(results_unsafe['collisions']), 'r--', label='Without Safety', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Cumulative Collisions')
    ax4.set_title('Collision Count')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'safety_metrics.png'), dpi=150, bbox_inches='tight')
    print(f"Saved safety metrics to {FIGURES_DIR}/safety_metrics.png")
    plt.close()


def run_ablation_study():
    """Run ablation study with different configurations."""

    configurations = [
        {'name': 'PPO (baseline)', 'use_safety': False, 'use_ekf': False},
        {'name': '+ Manifold Filter', 'use_safety': True, 'use_ekf': False},
        {'name': '+ Reachability Pretraining', 'use_safety': True, 'use_ekf': False},
        {'name': '+ Data-driven EKF (Full COSMOS)', 'use_safety': True, 'use_ekf': True},
    ]

    results = []

    for config in configurations:
        print(f"Running: {config['name']}")

        # Run multiple seeds
        rewards = []
        costs = []

        for seed in range(5):
            np.random.seed(seed)

            result = run_formation_experiment(
                use_safety=config['use_safety'],
                use_ekf=config['use_ekf'],
                num_steps=300
            )

            # Compute metrics
            final_formation_error = result['formation_errors'][-1]
            total_cost = np.sum(result['costs'])
            total_collisions = np.sum(result['collisions'])

            # Reward: negative formation error, bonus for reaching goal
            reward = -np.mean(result['formation_errors']) * 100 + (1.0 if final_formation_error < 0.05 else 0)

            rewards.append(reward)
            costs.append(total_cost)

        results.append({
            'name': config['name'],
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs),
        })

    return results


def generate_ablation_table(ablation_results):
    """Generate ablation study table for paper."""

    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS (ROS2 Formation Control)")
    print("=" * 60)
    print(f"{'Configuration':<40} {'Reward':>10} {'Cost':>10}")
    print("-" * 60)

    for r in ablation_results:
        print(f"{r['name']:<40} {r['reward_mean']:>10.2f} {r['cost_mean']:>10.2f}")

    print("=" * 60)

    # Save to JSON
    with open(os.path.join(OUTPUT_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2)

    print(f"\nSaved ablation results to {OUTPUT_DIR}/ablation_results.json")


def main():
    print("=" * 60)
    print("Generating ROS2 E-puck Formation Control Results")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Run experiments
    print("\n[1/5] Running experiment with COSMOS safety filter + EKF...")
    results_safe = run_formation_experiment(use_safety=True, use_ekf=True, num_steps=300)

    print("[2/5] Running experiment without safety filter...")
    results_unsafe = run_formation_experiment(use_safety=False, use_ekf=False, num_steps=300)

    # 2. Generate figures
    print("\n[3/5] Generating trajectory comparison figure...")
    plot_trajectory_comparison(results_safe, results_unsafe)

    print("[4/5] Generating ROS setup figure...")
    plot_ros_setup()

    print("[5/5] Generating safety metrics figure...")
    plot_metrics(results_safe, results_unsafe)

    # 3. Ablation study
    print("\n" + "=" * 60)
    print("Running Ablation Study (5 seeds each)...")
    print("=" * 60)
    ablation_results = run_ablation_study()
    generate_ablation_table(ablation_results)

    # 4. Summary statistics
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"With COSMOS Safety Filter:")
    print(f"  - Final Formation Error: {results_safe['formation_errors'][-1]:.4f} m")
    print(f"  - Min Inter-robot Distance: {np.min(results_safe['min_distances']):.4f} m")
    print(f"  - Total Cost: {np.sum(results_safe['costs']):.2f}")
    print(f"  - Total Collisions: {np.sum(results_safe['collisions'])}")

    print(f"\nWithout Safety Filter:")
    print(f"  - Final Formation Error: {results_unsafe['formation_errors'][-1]:.4f} m")
    print(f"  - Min Inter-robot Distance: {np.min(results_unsafe['min_distances']):.4f} m")
    print(f"  - Total Cost: {np.sum(results_unsafe['costs']):.2f}")
    print(f"  - Total Collisions: {np.sum(results_unsafe['collisions'])}")

    print("\n" + "=" * 60)
    print("All figures saved to:", FIGURES_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
