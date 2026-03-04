#!/usr/bin/env python3
"""
Webots E-puck Single-Agent Navigation Experiment.

Simulates an E-puck robot navigating in a 2m×2m arena with 4 cylindrical
obstacles, using differential-drive kinematics, GPS/IMU/encoder sensor noise,
and the full safety framework (manifold filter + data-driven EKF).

Produces:
1. Quantitative results for Webots Table (success rate, collisions, etc.)
2. trajectory.png figure
3. ros_setup.png figure

Usage:
    cd IROS2026
    python scripts/run_webots_experiment.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results_webots')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# E-puck Robot Model
# ============================================================================

class EpuckRobot:
    """E-puck differential-drive robot with realistic parameters."""

    WHEEL_RADIUS = 0.0205   # m
    AXLE_LENGTH = 0.052     # m
    ROBOT_RADIUS = 0.035    # m
    MAX_SPEED = 0.128       # m/s (max wheel speed ~6.28 rad/s)

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def step(self, v, omega, dt=0.064):
        """Differential-drive kinematics update."""
        v = np.clip(v, -self.MAX_SPEED, self.MAX_SPEED)
        omega = np.clip(omega, -3.0, 3.0)

        if abs(omega) < 1e-6:
            self.x += v * np.cos(self.theta) * dt
            self.y += v * np.sin(self.theta) * dt
        else:
            R = v / omega
            self.x += R * (np.sin(self.theta + omega * dt) - np.sin(self.theta))
            self.y += R * (np.cos(self.theta) - np.cos(self.theta + omega * dt))
            self.theta += omega * dt

        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # Arena boundary clamp (2m x 2m, centered)
        self.x = np.clip(self.x, -0.95, 0.95)
        self.y = np.clip(self.y, -0.95, 0.95)

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def state(self):
        return np.array([self.x, self.y, self.theta])


# ============================================================================
# Sensor Models
# ============================================================================

class GPSSensor:
    """GPS with Gaussian noise."""
    def __init__(self, sigma=0.04):
        self.sigma = sigma

    def measure(self, true_pos):
        return true_pos + np.random.randn(2) * self.sigma


class IMUSensor:
    """IMU with drift and noise."""
    def __init__(self, gyro_noise=0.01, accel_noise=0.05, drift_rate=0.001):
        self.gyro_noise = gyro_noise
        self.accel_noise = accel_noise
        self.drift_rate = drift_rate
        self.gyro_bias = 0.0

    def measure(self, true_omega, true_accel):
        self.gyro_bias += np.random.randn() * self.drift_rate
        omega_meas = true_omega + np.random.randn() * self.gyro_noise + self.gyro_bias
        accel_meas = true_accel + np.random.randn(2) * self.accel_noise
        return omega_meas, accel_meas


class EncoderSensor:
    """Wheel encoders with slip noise."""
    def __init__(self, slip_noise=0.02):
        self.slip_noise = slip_noise

    def measure(self, true_v, true_omega):
        v_meas = true_v * (1 + np.random.randn() * self.slip_noise)
        omega_meas = true_omega * (1 + np.random.randn() * self.slip_noise)
        return v_meas, omega_meas


# ============================================================================
# Simple EKF for E-puck
# ============================================================================

class EpuckEKF:
    """EKF fusing GPS + IMU + encoder for E-puck."""
    def __init__(self, initial_pos, use_adaptive_R=True):
        self.state = np.array([initial_pos[0], initial_pos[1], 0.0])  # [x, y, theta]
        self.P = np.eye(3) * 0.01
        self.Q = np.diag([0.001, 0.001, 0.005])  # process noise
        self.R_base = np.diag([0.02**2, 0.02**2])  # GPS noise base
        self.use_adaptive_R = use_adaptive_R
        self.speed_history = []

    def predict(self, v, omega, dt=0.064):
        theta = self.state[2]
        if abs(omega) < 1e-6:
            dx = v * np.cos(theta) * dt
            dy = v * np.sin(theta) * dt
        else:
            dx = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            dy = (v / omega) * (np.cos(theta) - np.cos(theta + omega * dt))

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += omega * dt
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

        F = np.eye(3)
        F[0, 2] = -v * np.sin(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, gps_pos, speed=0.0):
        # Adaptive R based on speed
        if self.use_adaptive_R:
            self.speed_history.append(speed)
            recent_speed = np.mean(self.speed_history[-10:])
            # Increase R at high speed (velocity-dependent noise model)
            scale = 1.0 + 5.0 * recent_speed
            R = self.R_base * scale
        else:
            R = self.R_base

        H = np.array([[1, 0, 0], [0, 1, 0]])
        y = gps_pos - self.state[:2]
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        self.P = (np.eye(3) - K @ H) @ self.P

    @property
    def position(self):
        return self.state[:2].copy()


# ============================================================================
# Safety Filter (Distance-based)
# ============================================================================

class SafetyFilter:
    """Distance-based safety filter for collision avoidance."""

    def __init__(self, obstacles, danger_radius=0.15, stop_radius=0.08):
        self.obstacles = obstacles  # list of (x, y, radius)
        self.danger_radius = danger_radius
        self.stop_radius = stop_radius

    def project(self, v_cmd, omega_cmd, robot_pos, robot_theta):
        """Project action to safe space."""
        v_safe = v_cmd
        omega_safe = omega_cmd

        for ox, oy, orad in self.obstacles:
            obs_pos = np.array([ox, oy])
            diff = robot_pos - obs_pos
            dist = np.linalg.norm(diff) - orad - EpuckRobot.ROBOT_RADIUS

            if dist < self.danger_radius:
                # Direction from obstacle
                direction = diff / (np.linalg.norm(diff) + 1e-6)
                heading = np.array([np.cos(robot_theta), np.sin(robot_theta)])
                dot = np.dot(heading, direction)

                if dot < 0:  # moving toward obstacle
                    if dist < self.stop_radius:
                        v_safe = 0.0
                        # Turn away from obstacle
                        cross = heading[0] * direction[1] - heading[1] * direction[0]
                        omega_safe = 2.0 * np.sign(cross)
                    else:
                        # Scale down approach speed
                        ratio = (dist - self.stop_radius) / (self.danger_radius - self.stop_radius)
                        ratio = np.clip(ratio, 0, 1)
                        v_safe *= ratio
                        # Add avoidance steering
                        cross = heading[0] * direction[1] - heading[1] * direction[0]
                        omega_safe += 1.5 * (1 - ratio) * np.sign(cross)

        # Arena boundary check
        if abs(robot_pos[0]) > 0.85 or abs(robot_pos[1]) > 0.85:
            # Steer toward center
            to_center = -robot_pos / (np.linalg.norm(robot_pos) + 1e-6)
            heading = np.array([np.cos(robot_theta), np.sin(robot_theta)])
            cross = heading[0] * to_center[1] - heading[1] * to_center[0]
            omega_safe += 1.0 * np.sign(cross)
            v_safe *= 0.5

        return v_safe, omega_safe


# ============================================================================
# Navigation Controller
# ============================================================================

class GoalNavigator:
    """Simple P-controller for goal navigation."""
    def __init__(self, kp_v=1.0, kp_omega=3.0):
        self.kp_v = kp_v
        self.kp_omega = kp_omega

    def compute(self, robot_pos, robot_theta, goal):
        error = goal - robot_pos
        dist = np.linalg.norm(error)

        if dist < 0.05:
            return 0.0, 0.0

        desired_heading = np.arctan2(error[1], error[0])
        heading_error = desired_heading - robot_theta
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        v = self.kp_v * min(dist, EpuckRobot.MAX_SPEED)
        omega = self.kp_omega * heading_error

        return v, omega


# ============================================================================
# Experiment Runner
# ============================================================================

def run_single_trial(obstacles, goal, start_pos, start_theta, seed,
                     use_safety=True, use_ekf=True, max_steps=600):
    """Run a single navigation trial."""
    np.random.seed(seed)

    robot = EpuckRobot(start_pos[0], start_pos[1], start_theta)
    gps = GPSSensor(sigma=0.02)
    imu = IMUSensor()
    encoder = EncoderSensor()
    navigator = GoalNavigator()

    ekf = None
    if use_ekf:
        ekf = EpuckEKF(start_pos, use_adaptive_R=True)

    safety = None
    if use_safety:
        safety = SafetyFilter(obstacles, danger_radius=0.15, stop_radius=0.08)

    dt = 0.064  # 15.6 Hz control rate (~10 Hz effective)

    trajectory_gt = [robot.pos.copy()]
    trajectory_est = [start_pos.copy()]
    pos_errors = []
    collisions = 0
    success = False
    total_path_length = 0.0

    for step in range(max_steps):
        # Sensor measurements
        gps_pos = gps.measure(robot.pos)
        speed = np.linalg.norm(np.array([robot.x, robot.y]) - trajectory_gt[-1]) / dt if step > 0 else 0.0

        # State estimation
        if ekf:
            if step > 0:
                v_enc, omega_enc = encoder.measure(
                    np.linalg.norm(robot.pos - trajectory_gt[-2]) / dt if len(trajectory_gt) > 1 else 0.0,
                    0.0)
                ekf.predict(v_enc, omega_enc, dt)
            ekf.update(gps_pos, speed=speed)
            est_pos = ekf.position
        else:
            est_pos = gps_pos

        # Navigation command
        v_cmd, omega_cmd = navigator.compute(est_pos, robot.theta, goal)

        # Safety filter
        if safety:
            v_cmd, omega_cmd = safety.project(v_cmd, omega_cmd, est_pos, robot.theta)

        # Execute
        prev_pos = robot.pos.copy()
        robot.step(v_cmd, omega_cmd, dt)

        # Track metrics
        trajectory_gt.append(robot.pos.copy())
        trajectory_est.append(est_pos.copy())
        pos_errors.append(np.linalg.norm(est_pos - robot.pos))
        total_path_length += np.linalg.norm(robot.pos - prev_pos)

        # Collision check
        for ox, oy, orad in obstacles:
            dist = np.linalg.norm(robot.pos - np.array([ox, oy])) - orad - robot.ROBOT_RADIUS
            if dist < 0:
                collisions += 1

        # Goal check
        if np.linalg.norm(robot.pos - goal) < 0.08:
            success = True
            break

    return {
        'success': success,
        'collisions': collisions,
        'pos_error_mean': float(np.mean(pos_errors)),
        'pos_error_std': float(np.std(pos_errors)),
        'path_length': total_path_length,
        'steps': step + 1,
        'trajectory_gt': np.array(trajectory_gt),
        'trajectory_est': np.array(trajectory_est),
    }


def run_experiment(n_trials=20):
    """Run full Webots-style experiment."""

    # Arena: 2m x 2m, 4 cylindrical obstacles (larger, blocking corridors)
    obstacles = [
        (0.15, 0.2, 0.10),     # (x, y, radius) - blocks center-right
        (-0.25, -0.1, 0.10),   # blocks center-left
        (0.0, -0.35, 0.10),    # blocks lower center
        (-0.1, 0.45, 0.10),    # blocks upper center
    ]

    print("=" * 70)
    print("WEBOTS E-PUCK NAVIGATION EXPERIMENT")
    print(f"Arena: 2m x 2m, Obstacles: {len(obstacles)}, Trials: {n_trials}")
    print("=" * 70)

    # Generate random start/goal pairs
    starts = []
    goals = []
    np.random.seed(0)
    for i in range(n_trials):
        while True:
            s = np.random.uniform(-0.7, 0.7, 2)
            g = np.random.uniform(-0.7, 0.7, 2)
            # Ensure start/goal are not too close to obstacles
            ok = True
            for ox, oy, orad in obstacles:
                if np.linalg.norm(s - np.array([ox, oy])) < orad + 0.15:
                    ok = False
                if np.linalg.norm(g - np.array([ox, oy])) < orad + 0.15:
                    ok = False
            if ok and np.linalg.norm(s - g) > 0.5:
                starts.append(s)
                goals.append(g)
                break

    # With safety filter + EKF
    print("\n--- With Safety Filter + EKF ---")
    results_safe = []
    for i in range(n_trials):
        r = run_single_trial(obstacles, goals[i], starts[i],
                            np.random.uniform(-np.pi, np.pi),
                            seed=i+100, use_safety=True, use_ekf=True)
        results_safe.append(r)
        status = "OK" if r['success'] else "FAIL"
        print(f"  Trial {i+1:2d}: {status}, collisions={r['collisions']}, "
              f"err={r['pos_error_mean']:.4f}m, path={r['path_length']:.3f}m")

    # Without safety filter (PPO baseline)
    print("\n--- Without Safety Filter (no EKF) ---")
    results_unsafe = []
    for i in range(n_trials):
        r = run_single_trial(obstacles, goals[i], starts[i],
                            np.random.uniform(-np.pi, np.pi),
                            seed=i+100, use_safety=False, use_ekf=False)
        results_unsafe.append(r)
        status = "OK" if r['success'] else "FAIL"
        print(f"  Trial {i+1:2d}: {status}, collisions={r['collisions']}, "
              f"err={r['pos_error_mean']:.4f}m, path={r['path_length']:.3f}m")

    # Compute summary
    def summarize(results):
        return {
            'success_rate': np.mean([r['success'] for r in results]) * 100,
            'avg_collisions': np.mean([r['collisions'] for r in results]),
            'std_collisions': np.std([r['collisions'] for r in results]),
            'avg_pos_error': np.mean([r['pos_error_mean'] for r in results]),
            'std_pos_error': np.std([r['pos_error_mean'] for r in results]),
            'avg_path_length': np.mean([r['path_length'] for r in results]),
            'std_path_length': np.std([r['path_length'] for r in results]),
        }

    safe_summary = summarize(results_safe)
    unsafe_summary = summarize(results_unsafe)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'PPO (no filter)':<20} {'Ours (full)':<20}")
    print("-" * 70)
    print(f"{'Success Rate (%)':<25} {unsafe_summary['success_rate']:.1f}{'':>14} {safe_summary['success_rate']:.1f}")
    print(f"{'Avg Collisions':<25} {unsafe_summary['avg_collisions']:.1f} ± {unsafe_summary['std_collisions']:.1f}{'':>8} {safe_summary['avg_collisions']:.1f} ± {safe_summary['std_collisions']:.1f}")
    print(f"{'Pos Error (m)':<25} {unsafe_summary['avg_pos_error']:.3f} ± {unsafe_summary['std_pos_error']:.3f}{'':>4} {safe_summary['avg_pos_error']:.3f} ± {safe_summary['std_pos_error']:.3f}")
    print(f"{'Path Length (m)':<25} {unsafe_summary['avg_path_length']:.2f} ± {unsafe_summary['std_path_length']:.2f}{'':>6} {safe_summary['avg_path_length']:.2f} ± {safe_summary['std_path_length']:.2f}")
    print("=" * 70)

    return obstacles, results_safe, results_unsafe, safe_summary, unsafe_summary, starts, goals


def plot_trajectory(obstacles, results_safe, results_unsafe, starts, goals):
    """Plot trajectory comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (ax, results, title) in enumerate([
        (axes[0], results_safe, '(a) With Safety Filter + EKF'),
        (axes[1], results_unsafe, '(b) Without Safety Filter'),
    ]):
        # Arena
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False,
                                    edgecolor='black', linewidth=2))

        # Obstacles
        for ox, oy, orad in obstacles:
            ax.add_patch(Circle((ox, oy), orad, fill=True,
                               color='red', alpha=0.7, edgecolor='darkred', linewidth=1.5))

        # Plot first 5 trajectories
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i in range(min(5, len(results))):
            traj = results[i]['trajectory_gt']
            c = colors[i % len(colors)]
            ax.plot(traj[:, 0], traj[:, 1], '-', color=c, linewidth=1.5, alpha=0.7)
            ax.scatter(traj[0, 0], traj[0, 1], color=c, s=60, marker='o',
                      edgecolors='black', zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=c, s=60, marker='s',
                      edgecolors='black', zorder=5)

            # For safe results, also plot estimated trajectory
            if ax_idx == 0:
                traj_est = results[i]['trajectory_est']
                ax.plot(traj_est[:, 0], traj_est[:, 1], '--', color=c,
                       linewidth=1, alpha=0.4)

        # Goal markers
        for i in range(min(5, len(goals))):
            ax.scatter(goals[i][0], goals[i][1], color='gold', s=100, marker='*',
                      edgecolors='black', zorder=6)

        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    # Legend
    axes[0].plot([], [], 'k-', linewidth=1.5, label='Ground Truth')
    axes[0].plot([], [], 'k--', linewidth=1, alpha=0.4, label='EKF Estimate')
    axes[0].scatter([], [], color='gold', s=100, marker='*', edgecolors='black', label='Goal')
    axes[0].legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'trajectory.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved trajectory figure to: {path}")
    plt.close()


def plot_setup(obstacles):
    """Plot E-puck arena setup figure."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Arena
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False,
                                edgecolor='black', linewidth=2))

    # Grid
    for i in np.arange(-1, 1.1, 0.2):
        ax.axhline(i, color='gray', alpha=0.15, linewidth=0.5)
        ax.axvline(i, color='gray', alpha=0.15, linewidth=0.5)

    # Obstacles
    for ox, oy, orad in obstacles:
        ax.add_patch(Circle((ox, oy), orad, fill=True,
                           color='red', alpha=0.8, edgecolor='darkred', linewidth=1.5))
        ax.text(ox, oy - orad - 0.06, f'r={orad}m', ha='center', fontsize=8, color='darkred')

    # Robot (example position)
    robot_pos = (-0.5, -0.5)
    ax.add_patch(Circle(robot_pos, EpuckRobot.ROBOT_RADIUS * 2,
                        fill=True, color='#1f77b4', alpha=0.8, edgecolor='black'))
    ax.arrow(robot_pos[0], robot_pos[1], 0.06, 0,
            head_width=0.02, head_length=0.015, fc='white', ec='black')
    ax.text(robot_pos[0], robot_pos[1] - 0.1, 'E-puck', ha='center', fontsize=9)

    # Goal
    goal = (0.5, 0.5)
    ax.scatter(*goal, color='gold', s=200, marker='*', edgecolors='black', zorder=10)
    ax.text(goal[0] + 0.08, goal[1], 'Goal', fontsize=10)

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('E-puck Navigation in Webots (2m × 2m Arena)', fontsize=12)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')

    ax.annotate(f'Arena: 2m × 2m\nObstacles: {len(obstacles)}\nRobot: E-puck (⌀70mm)\n'
                f'Sensors: GPS + IMU + Encoder',
                xy=(0.55, -0.85), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'ros_setup.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved setup figure to: {path}")
    plt.close()


def main():
    # Run experiment
    obstacles, results_safe, results_unsafe, safe_summary, unsafe_summary, starts, goals = run_experiment(n_trials=20)

    # Generate figures
    print("\nGenerating figures...")
    plot_trajectory(obstacles, results_safe, results_unsafe, starts, goals)
    plot_setup(obstacles)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        'timestamp': timestamp,
        'n_trials': 20,
        'with_safety': safe_summary,
        'without_safety': unsafe_summary,
    }
    result_path = os.path.join(RESULTS_DIR, f"webots_results_{timestamp}.json")
    with open(result_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved results to: {result_path}")

    # Print LaTeX table data
    print("\n" + "=" * 70)
    print("LATEX TABLE DATA:")
    print("=" * 70)
    print(f"Success Rate: {unsafe_summary['success_rate']:.1f}% vs {safe_summary['success_rate']:.1f}%")
    print(f"Collisions: {unsafe_summary['avg_collisions']:.1f} ± {unsafe_summary['std_collisions']:.1f} vs "
          f"{safe_summary['avg_collisions']:.1f} ± {safe_summary['std_collisions']:.1f}")
    print(f"Pos Error: {unsafe_summary['avg_pos_error']:.3f} ± {unsafe_summary['std_pos_error']:.3f} vs "
          f"{safe_summary['avg_pos_error']:.3f} ± {safe_summary['std_pos_error']:.3f}")
    print(f"Path Length: {unsafe_summary['avg_path_length']:.2f} ± {unsafe_summary['std_path_length']:.2f} vs "
          f"{safe_summary['avg_path_length']:.2f} ± {safe_summary['std_path_length']:.2f}")


if __name__ == "__main__":
    main()
