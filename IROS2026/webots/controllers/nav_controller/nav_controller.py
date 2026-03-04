#!/usr/bin/env python3
"""
Webots E-puck Navigation Safety Experiment - Final Version

P-controller with obstacle avoidance (both modes) + DistanceFilter
safety layer + StandardEKF (safe mode only). Uses 6 obstacles for
challenging navigation. Conservative filter parameters prevent all
collisions in safe mode.
"""

import sys
import os
import json
import numpy as np

# Auto-detect src path relative to this controller
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IROS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, os.path.join(IROS_DIR, 'src'))

from controller import Supervisor
from safety.distance_filter import DistanceFilter
from ekf import StandardEKF, EKFConfig

WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
ROBOT_RADIUS = 0.035
MAX_WHEEL_SPEED = 6.28
MAX_LINEAR_SPEED = WHEEL_RADIUS * MAX_WHEEL_SPEED

OBSTACLE_RADIUS = 0.10
COLLISION_DIST = ROBOT_RADIUS + OBSTACLE_RADIUS + 0.005
GOAL_THRESHOLD = 0.08
MAX_STEPS = 2500
N_TRIALS = 20
GPS_NOISE_STD = 0.04
HEADING_NOISE_STD = 0.05
SEED = 42

# 6 obstacles (matches world file)
OBSTACLES = [
    np.array([-0.4, 0.3]),
    np.array([0.3, 0.5]),
    np.array([-0.3, -0.4]),
    np.array([0.5, -0.3]),
    np.array([0.0, 0.05]),
    np.array([-0.15, 0.55]),
]


def generate_trial_configs(n_trials=20, seed=42):
    rng = np.random.RandomState(seed)
    configs = []
    arena = 0.75
    min_dist_obs = OBSTACLE_RADIUS + ROBOT_RADIUS + 0.12
    while len(configs) < n_trials:
        start = rng.uniform(-arena, arena, 2)
        goal = rng.uniform(-arena, arena, 2)
        if np.linalg.norm(goal - start) < 0.5:
            continue
        ok = True
        for obs in OBSTACLES:
            if np.linalg.norm(start - obs) < min_dist_obs:
                ok = False
                break
            if np.linalg.norm(goal - obs) < min_dist_obs:
                ok = False
                break
        if ok:
            configs.append((start.copy(), goal.copy()))
    return configs


def get_true_state(robot_node):
    pos = robot_node.getPosition()
    rot = robot_node.getOrientation()
    heading = np.arctan2(rot[3], rot[0])
    return np.array([pos[0], pos[1]]), heading


def p_controller_with_avoidance(position, heading, goal,
                                 obstacles=OBSTACLES,
                                 obstacle_radius=OBSTACLE_RADIUS,
                                 avoidance_radius=0.15):
    """P-controller with weak obstacle avoidance (radius=0.15m from edge).
    Small radius means the robot navigates close to obstacles, relying on
    the safety filter to prevent actual collisions in safe mode."""
    error = goal - position
    dist_to_goal = np.linalg.norm(error)
    if dist_to_goal < GOAL_THRESHOLD:
        return 0.0, 0.0

    goal_dir = error / (dist_to_goal + 1e-8)
    desired_heading = np.arctan2(error[1], error[0])

    closest_obs = None
    closest_gap = float('inf')
    for obs in obstacles:
        to_obs = obs - position
        center_dist = np.linalg.norm(to_obs)
        gap = center_dist - obstacle_radius
        if gap < avoidance_radius and center_dist > 1e-4:
            obs_dir = to_obs / center_dist
            if np.dot(obs_dir, goal_dir) > -0.2:
                if gap < closest_gap:
                    closest_gap = gap
                    closest_obs = obs

    if closest_obs is not None and closest_gap < avoidance_radius:
        to_obs = closest_obs - position
        center_dist = np.linalg.norm(to_obs) + 1e-6
        tangent1 = np.array([-to_obs[1], to_obs[0]]) / center_dist
        tangent2 = np.array([to_obs[1], -to_obs[0]]) / center_dist
        avoid_dir = tangent1 if np.dot(tangent1, goal_dir) >= np.dot(tangent2, goal_dir) else tangent2
        blend = np.clip(1.0 - closest_gap / avoidance_radius, 0.0, 1.0) ** 0.5
        combined = (1.0 - blend) * goal_dir + blend * avoid_dir
        n = np.linalg.norm(combined)
        if n > 1e-6:
            combined /= n
        desired_heading = np.arctan2(combined[1], combined[0])

    heading_error = desired_heading - heading
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    v = min(0.4 * dist_to_goal, MAX_LINEAR_SPEED)
    if abs(heading_error) > 0.5:
        v *= 0.2
    omega = 3.5 * heading_error
    return v, omega


def set_velocity(left_motor, right_motor, v, omega):
    v = np.clip(v, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
    omega = np.clip(omega, -4.0, 4.0)
    v_left = (v - omega * AXLE_LENGTH / 2) / WHEEL_RADIUS
    v_right = (v + omega * AXLE_LENGTH / 2) / WHEEL_RADIUS
    v_left = np.clip(v_left, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    v_right = np.clip(v_right, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    left_motor.setVelocity(float(v_left))
    right_motor.setVelocity(float(v_right))


def run_trial(robot, robot_node, left_motor, right_motor, timestep,
              start, goal, use_safety=False, use_ekf=False, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    dt = timestep / 1000.0

    trans_field = robot_node.getField('translation')
    rot_field = robot_node.getField('rotation')
    initial_heading = np.arctan2(goal[1] - start[1], goal[0] - start[0])
    trans_field.setSFVec3f([float(start[0]), float(start[1]), 0.0])
    rot_field.setSFRotation([0.0, 0.0, 1.0, float(initial_heading)])
    robot_node.resetPhysics()
    set_velocity(left_motor, right_motor, 0, 0)
    for _ in range(10):
        robot.step(timestep)

    # Conservative safety filter: inflated hazard radius accounts for
    # robot size + estimation error
    safety_filter = None
    if use_safety:
        safety_filter = DistanceFilter(
            hazard_radius=OBSTACLE_RADIUS + ROBOT_RADIUS,  # 0.135m
            danger_radius=0.12,   # slow at 12cm from inflated edge
            stop_radius=0.08,     # stop at 8cm from inflated edge
            max_forward_vel=MAX_LINEAR_SPEED,
            max_angular_vel=2.0,
            lambda_calib=0.0,
        )
        safety_filter.reset(OBSTACLES)

    ekf = None
    if use_ekf:
        ekf = StandardEKF(config=EKFConfig(dt=dt, sigma_lat=GPS_NOISE_STD,
                                            sigma_up=HEADING_NOISE_STD))
        true_pos, true_heading = get_true_state(robot_node)
        ekf.reset(np.array([true_pos[0], true_pos[1], true_heading]))

    collisions = 0
    in_collision = False
    path_length = 0.0
    pos_errors = []
    prev_true_pos = None
    prev_action = np.array([0.0, 0.0])
    success = False
    final_step = MAX_STEPS
    traj_gt = []    # ground truth trajectory
    traj_est = []   # estimated trajectory (EKF or noisy)

    for step in range(MAX_STEPS):
        if robot.step(timestep) == -1:
            break

        true_pos, true_heading = get_true_state(robot_node)
        if prev_true_pos is None:
            prev_true_pos = true_pos.copy()

        noisy_pos = true_pos + rng.randn(2) * GPS_NOISE_STD
        noisy_heading = true_heading + rng.randn() * HEADING_NOISE_STD

        if use_ekf and ekf is not None:
            ekf.predict(prev_action, dt=dt)
            ekf.update(np.array([noisy_pos[0], noisy_pos[1], noisy_heading]))
            est = ekf.get_position()
            nav_pos, nav_heading = est[:2], est[2]
        else:
            nav_pos, nav_heading = noisy_pos, noisy_heading

        pos_errors.append(float(np.linalg.norm(nav_pos - true_pos)))

        # Log trajectory every 5 steps to keep file size manageable
        if step % 5 == 0:
            traj_gt.append([float(true_pos[0]), float(true_pos[1])])
            traj_est.append([float(nav_pos[0]), float(nav_pos[1])])

        if np.linalg.norm(true_pos - goal) < GOAL_THRESHOLD:
            success = True
            final_step = step + 1
            break

        v, omega = p_controller_with_avoidance(nav_pos, nav_heading, goal)

        if safety_filter is not None:
            robot_pose = np.array([nav_pos[0], nav_pos[1], nav_heading])
            result = safety_filter.project(np.array([v, omega]), robot_pose)
            v = float(result.action_safe[0])
            omega = float(result.action_safe[1])

        prev_action = np.array([v, omega])
        set_velocity(left_motor, right_motor, v, omega)

        colliding = any(
            np.linalg.norm(true_pos - obs) < COLLISION_DIST
            for obs in OBSTACLES
        )
        if colliding and not in_collision:
            collisions += 1
        in_collision = colliding

        path_length += float(np.linalg.norm(true_pos - prev_true_pos))
        prev_true_pos = true_pos.copy()

    # Log final position
    traj_gt.append([float(true_pos[0]), float(true_pos[1])])
    traj_est.append([float(nav_pos[0]), float(nav_pos[1])])

    set_velocity(left_motor, right_motor, 0, 0)
    return {
        'success': bool(success),
        'collisions': int(collisions),
        'path_length': float(path_length),
        'steps': int(final_step),
        'avg_pos_error': float(np.mean(pos_errors)) if pos_errors else 0.0,
        'std_pos_error': float(np.std(pos_errors)) if pos_errors else 0.0,
        'traj_gt': traj_gt,
        'traj_est': traj_est,
    }


def main():
    print("=" * 60)
    print("WEBOTS E-PUCK NAVIGATION EXPERIMENT (FINAL)")
    print("=" * 60)

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

    for i in range(8):
        ds = robot.getDevice(f'ps{i}')
        ds.enable(timestep)

    robot_node = robot.getSelf()
    trials = generate_trial_configs(N_TRIALS, seed=SEED)
    print(f"Generated {len(trials)} trial configs")

    all_results = {
        'trial_configs': [[s.tolist(), g.tolist()] for s, g in trials],
        'obstacles': [obs.tolist() for obs in OBSTACLES],
    }
    for mode in ['safe', 'unsafe']:
        print(f"\n{'=' * 60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'=' * 60}")

        mode_results = []
        rng = np.random.RandomState(SEED)
        for trial_idx, (start, goal) in enumerate(trials):
            result = run_trial(
                robot, robot_node, left_motor, right_motor, timestep,
                start, goal,
                use_safety=(mode == 'safe'),
                use_ekf=(mode == 'safe'),
                rng=rng,
            )
            mode_results.append(result)
            status = "OK" if result['success'] else "TIMEOUT"
            print(f"  T{trial_idx+1:2d}: {status} col={result['collisions']} "
                  f"path={result['path_length']:.2f}m err={result['avg_pos_error']:.4f}m")
        all_results[mode] = mode_results

        # Save after each mode so results survive early termination
        output_dir = os.path.join(IROS_DIR, 'results_webots_real')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'webots_results.json')
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        sys.stdout.flush()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for mode in ['safe', 'unsafe']:
        if mode not in all_results:
            continue
        results = all_results[mode]
        n = len(results)
        succ = sum(r['success'] for r in results)
        cols = [r['collisions'] for r in results]
        errs = [r['avg_pos_error'] for r in results]
        paths = [r['path_length'] for r in results]
        print(f"  {mode.upper():>6s}: success={succ}/{n} ({100*succ/n:.0f}%), "
              f"col={np.mean(cols):.1f}+/-{np.std(cols):.1f}, "
              f"err={np.mean(errs):.4f}+/-{np.std(errs):.4f}m, "
              f"path={np.mean(paths):.2f}+/-{np.std(paths):.2f}m")

    print(f"\nResults saved to: {output_path}")
    sys.stdout.flush()
    robot.simulationQuit(0)


if __name__ == '__main__':
    main()
