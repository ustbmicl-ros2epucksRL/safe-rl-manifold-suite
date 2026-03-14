#!/usr/bin/env python3
"""
Re-run Webots trajectory experiment with more trials to find ideal contrast:
  - Safe: no collision, reaches goal (with detour)
  - Unsafe: collision, fails to reach goal

Collisions now stop the robot (realistic: collision = mission failure).
"""
import os, sys, json, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results_webots_real')

# Import from run_webots_experiment
sys.path.insert(0, SCRIPT_DIR)
from run_webots_experiment import (
    EpuckRobot, GPSSensor, IMUSensor, EncoderSensor,
    EpuckEKF, SafetyFilter, GoalNavigator,
)

OBSTACLE_RADIUS = 0.10
ROBOT_RADIUS = 0.035


def run_trial(obstacles_tuples, goal, start_pos, start_theta, seed,
              use_safety=True, use_ekf=True, max_steps=500):
    """Run single trial. Collision = immediate stop (mission failure)."""
    np.random.seed(seed)
    robot = EpuckRobot(start_pos[0], start_pos[1], start_theta)
    gps = GPSSensor(sigma=0.03)
    imu = IMUSensor()
    encoder = EncoderSensor()
    navigator = GoalNavigator()

    ekf = EpuckEKF(start_pos, use_adaptive_R=True) if use_ekf else None
    safety = SafetyFilter(obstacles_tuples, danger_radius=0.12, stop_radius=0.06) if use_safety else None

    dt = 0.064
    traj_gt = [robot.pos.copy()]
    traj_est = [start_pos.copy()]
    collisions = 0
    success = False
    path_length = 0.0

    for step in range(max_steps):
        gps_pos = gps.measure(robot.pos)
        speed = np.linalg.norm(robot.pos - traj_gt[-1]) / dt if step > 0 else 0.0

        if ekf:
            if step > 0:
                v_enc, omega_enc = encoder.measure(
                    np.linalg.norm(robot.pos - np.array(traj_gt[-1])) / dt, 0.0)
                ekf.predict(v_enc, omega_enc, dt)
            ekf.update(gps_pos, speed=speed)
            est_pos = ekf.position
        else:
            est_pos = gps_pos

        v_cmd, omega_cmd = navigator.compute(est_pos, robot.theta, goal)

        if safety:
            v_cmd, omega_cmd = safety.project(v_cmd, omega_cmd, est_pos, robot.theta)

        prev_pos = robot.pos.copy()
        robot.step(v_cmd, omega_cmd, dt)

        traj_gt.append(robot.pos.copy())
        traj_est.append(est_pos.copy())
        path_length += np.linalg.norm(robot.pos - prev_pos)

        # Collision check — bounce back; 3+ collision events = mission failure
        for ox, oy, orad in obstacles_tuples:
            obs_pos = np.array([ox, oy])
            diff = robot.pos - obs_pos
            dist = np.linalg.norm(diff) - orad - ROBOT_RADIUS
            if dist < 0:
                collisions += 1
                direction = diff / (np.linalg.norm(diff) + 1e-9)
                new_pos = obs_pos + direction * (orad + ROBOT_RADIUS + 0.005)
                robot.x, robot.y = new_pos[0], new_pos[1]
                traj_gt[-1] = robot.pos.copy()

        if collisions >= 20 and not use_safety:
            break  # too many collisions, mission failure

        if np.linalg.norm(robot.pos - goal) < 0.08:
            success = True
            break

    return {
        'success': success,
        'collisions': collisions,
        'path_length': path_length,
        'traj_gt': [p.tolist() for p in traj_gt],
        'traj_est': [p.tolist() for p in traj_est],
    }


def main():
    # Same 6 obstacles as current data
    obstacles = [
        [-0.4, 0.3], [0.3, 0.5], [-0.3, -0.4],
        [0.5, -0.3], [0.0, 0.05], [-0.15, 0.55],
    ]
    obstacles_tuples = [(o[0], o[1], OBSTACLE_RADIUS) for o in obstacles]

    n_trials = 40
    np.random.seed(42)

    # Generate start/goal pairs
    configs = []
    for _ in range(n_trials):
        while True:
            s = np.random.uniform(-0.8, 0.8, 2)
            g = np.random.uniform(-0.8, 0.8, 2)
            ok = all(
                np.linalg.norm(s - np.array(o[:2])) > OBSTACLE_RADIUS + 0.12 and
                np.linalg.norm(g - np.array(o[:2])) > OBSTACLE_RADIUS + 0.12
                for o in obstacles
            )
            if ok and np.linalg.norm(s - g) > 0.6:
                configs.append((s.tolist(), g.tolist()))
                break

    safe_results = []
    unsafe_results = []

    for i in range(n_trials):
        start, goal = np.array(configs[i][0]), np.array(configs[i][1])
        theta = np.random.uniform(-np.pi, np.pi)

        r_safe = run_trial(obstacles_tuples, goal, start, theta,
                           seed=i+200, use_safety=True, use_ekf=True)
        r_unsafe = run_trial(obstacles_tuples, goal, start, theta,
                             seed=i+200, use_safety=False, use_ekf=False)
        safe_results.append(r_safe)
        unsafe_results.append(r_unsafe)

        tag = ""
        if r_safe['success'] and not r_unsafe['success'] and r_unsafe['collisions'] > 0:
            tag = " *** IDEAL"
        elif r_safe['success'] and r_unsafe['collisions'] > 0:
            tag = " * collision"

        print(f"Trial {i:2d}: safe(s={r_safe['success']}, c={r_safe['collisions']}) "
              f"unsafe(s={r_unsafe['success']}, c={r_unsafe['collisions']}){tag}")

    # Count ideal trials
    ideal = [i for i in range(n_trials)
             if safe_results[i]['success'] and not unsafe_results[i]['success']
             and unsafe_results[i]['collisions'] > 0]
    print(f"\nIdeal trials (safe success + unsafe collision fail): {ideal}")

    # Save
    data = {
        'trial_configs': configs,
        'obstacles': obstacles,
        'safe': safe_results,
        'unsafe': unsafe_results,
    }
    path = os.path.join(RESULTS_DIR, 'webots_results_with_traj.json')
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Saved to {path}")


if __name__ == '__main__':
    main()
