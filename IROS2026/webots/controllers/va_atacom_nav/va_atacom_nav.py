#!/usr/bin/env python3
"""
Webots E-puck VA-ATACOM Safety Filter Inference (AAAI 2027).

Drives the E-puck in the corridor world with a P-controller and the
**VA-ATACOM** velocity-augmented braking-distance manifold filter
(safe-rl-2027/safe_rl/filters/brake_manifold.py), instead of the
distance-based filter used by the IROS2026 nav_controller.

Modes:
  - safe   : P-controller + BrakeManifoldFilter + StandardEKF
  - unsafe : P-controller only (no filter, no EKF)

Outputs: results_webots_va_atacom/webots_va_atacom_results.json with
per-trial success/collisions/path_length/avg_pos_error plus filter
intervention statistics (n_corrected, mean_barrier_min).

Filter params calibrated for E-puck:
  hazard_radius = 0.10   (cylinder radius from .wbt)
  safety_margin = 0.035  (robot body radius)
  a_max         = 0.082  (calibrated deceleration; from epuck_transfer.py)
  alpha0        = 1.0
  action_scale  = a_max  (maps normalised fwd in [-1,1] -> a_max accel)

Place: IROS2026/webots/controllers/va_atacom_nav/va_atacom_nav.py
Referenced by:  IROS2026/webots/worlds/epuck_corridor_va_atacom.wbt
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np

# --- Paths ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IROS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
REPO_ROOT = os.path.abspath(os.path.join(IROS_DIR, '..'))
sys.path.insert(0, os.path.join(IROS_DIR, 'src'))  # for StandardEKF
sys.path.insert(0, os.path.join(REPO_ROOT, 'safe-rl-2027'))  # for brake_manifold

from controller import Supervisor                          # noqa: E402
from safe_rl.filters.brake_manifold import BrakeManifoldFilter  # noqa: E402
try:
    from ekf import StandardEKF, EKFConfig                 # noqa: E402
    HAVE_EKF = True
except Exception:
    HAVE_EKF = False

# --- Constants --------------------------------------------------------------
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
ROBOT_RADIUS = 0.035
MAX_WHEEL_SPEED = 6.28
MAX_LINEAR_SPEED = WHEEL_RADIUS * MAX_WHEEL_SPEED   # ~0.1287 m/s
MAX_ANGULAR_SPEED = 3.0                              # rad/s (for normalisation)

OBSTACLE_RADIUS = 0.10
COLLISION_DIST = ROBOT_RADIUS + OBSTACLE_RADIUS + 0.005
GOAL_THRESHOLD = 0.08
MAX_STEPS = int(os.environ.get('VA_ATACOM_MAX_STEPS', '4000'))
N_TRIALS = int(os.environ.get('VA_ATACOM_N_TRIALS', '20'))
GPS_NOISE_STD = 0.04
HEADING_NOISE_STD = 0.05
SEED = 42

# E-puck calibrated dynamics (from safe-rl-2027/experiments/transfer/epuck_transfer.py)
A_MAX = 0.082          # max forward deceleration (m/s^2)
ALPHA0 = 1.0
D_SAFE = ROBOT_RADIUS  # 0.035; r_base = OBSTACLE_RADIUS + D_SAFE = 0.135

# Obstacles in the epuck_corridor_va_atacom.wbt world (5 cylinders, r=0.10).
# Matches the original epuck_demo_corridor.wbt obstacle layout exactly.
OBSTACLES = [
    np.array([-0.45, -0.25]),
    np.array([-0.05, -0.55]),
    np.array([0.00, 0.00]),
    np.array([0.10, 0.40]),
    np.array([0.55, 0.15]),
]


def generate_trial_configs(n_trials=N_TRIALS, seed=SEED):
    """Random (start, goal) pairs spanning the 2x2 arena, avoiding obstacles."""
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
                ok = False; break
            if np.linalg.norm(goal - obs) < min_dist_obs:
                ok = False; break
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
    """P-controller with weak (non-safety) obstacle steering.
    The VA-ATACOM filter is the hard guarantee on top; this just gives the
    policy a sane heuristic so trials don't get stuck head-on at obstacles.
    """
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
            if np.dot(obs_dir, goal_dir) > -0.2 and gap < closest_gap:
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


def apply_va_atacom_filter(brake_filter, v_cmd, omega_cmd, nav_pos, nav_heading,
                            prev_v_safe, prev_heading):
    """Adapter: (v, omega) physical command -> normalised [fwd, turn] for the
    BrakeManifoldFilter -> filtered [fwd_safe, turn_safe] -> physical (v, omega).

    Filter `current_velocity` is the *actual* world-frame velocity the robot
    has on entry to this step. For velocity-controlled diff-drive wheels (E-puck),
    this equals last step's filtered commanded velocity projected on last step's
    heading -- much cleaner than GPS-finite-differencing (the GPS noise σ=0.04 m
    over dt=0.064 s injects ~0.6 m/s noise into the derived velocity, dwarfing
    the e-puck's 0.13 m/s top speed and over-inflating the braking term).
    """
    fwd_n = np.clip(v_cmd / MAX_LINEAR_SPEED, -1.0, 1.0)
    turn_n = np.clip(omega_cmd / MAX_ANGULAR_SPEED, -1.0, 1.0)
    vel_world = np.array([prev_v_safe * np.cos(prev_heading),
                          prev_v_safe * np.sin(prev_heading)])
    result = brake_filter.project(
        np.array([fwd_n, turn_n]),
        np.array([nav_pos[0], nav_pos[1], nav_heading]),
        current_velocity=vel_world,
    )
    fwd_safe, turn_safe = float(result.action_safe[0]), float(result.action_safe[1])
    v_safe = fwd_safe * MAX_LINEAR_SPEED
    omega_safe = turn_safe * MAX_ANGULAR_SPEED
    return v_safe, omega_safe, result


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

    # VA-ATACOM filter
    brake_filter = None
    if use_safety:
        brake_filter = BrakeManifoldFilter(
            hazard_radius=OBSTACLE_RADIUS,
            safety_margin=D_SAFE,
            a_max=A_MAX,
            alpha0=ALPHA0,
            action_scale=A_MAX,
            action_form="diff_drive",
        )
        brake_filter.reset(OBSTACLES)

    ekf = None
    if use_ekf and HAVE_EKF:
        ekf = StandardEKF(config=EKFConfig(dt=dt, sigma_lat=GPS_NOISE_STD,
                                            sigma_up=HEADING_NOISE_STD))
        tp, th = get_true_state(robot_node)
        ekf.reset(np.array([tp[0], tp[1], th]))

    collisions = 0
    in_collision = False
    path_length = 0.0
    pos_errors = []
    prev_true_pos = None
    prev_v_safe = 0.0           # robot's actual fwd velocity entering this step
    prev_heading = 0.0
    prev_action = np.array([0.0, 0.0])
    success = False
    final_step = MAX_STEPS

    n_filter_calls = 0
    n_filter_corrected = 0
    barrier_min_log = []
    fwd_correction_abs_log = []
    traj_gt = []
    traj_est = []

    for step in range(MAX_STEPS):
        if robot.step(timestep) == -1:
            break
        true_pos, true_heading = get_true_state(robot_node)
        if prev_true_pos is None:
            prev_true_pos = true_pos.copy()
        noisy_pos = true_pos + rng.randn(2) * GPS_NOISE_STD
        noisy_heading = true_heading + rng.randn() * HEADING_NOISE_STD

        if ekf is not None:
            ekf.predict(prev_action, dt=dt)
            ekf.update(np.array([noisy_pos[0], noisy_pos[1], noisy_heading]))
            est = ekf.get_position()
            nav_pos, nav_heading = est[:2], est[2]
        else:
            nav_pos, nav_heading = noisy_pos, noisy_heading

        pos_errors.append(float(np.linalg.norm(nav_pos - true_pos)))
        if step % 5 == 0:
            traj_gt.append([float(true_pos[0]), float(true_pos[1])])
            traj_est.append([float(nav_pos[0]), float(nav_pos[1])])

        if np.linalg.norm(true_pos - goal) < GOAL_THRESHOLD:
            success = True
            final_step = step + 1
            break

        avoid_r = 0.15 if use_safety else 0.08
        v, omega = p_controller_with_avoidance(nav_pos, nav_heading, goal,
                                               avoidance_radius=avoid_r)
        if brake_filter is not None:
            v_safe, omega_safe, result = apply_va_atacom_filter(
                brake_filter, v, omega, nav_pos, nav_heading,
                prev_v_safe, prev_heading)
            n_filter_calls += 1
            if bool(result.is_corrected):
                n_filter_corrected += 1
                fwd_correction_abs_log.append(abs(float(result.correction[0])))
            barrier_min_log.append(float(result.barrier_min))
            v, omega = v_safe, omega_safe

        prev_action = np.array([v, omega])
        prev_v_safe = float(v)
        prev_heading = float(nav_heading)
        set_velocity(left_motor, right_motor, v, omega)

        colliding = any(np.linalg.norm(true_pos - o) < COLLISION_DIST for o in OBSTACLES)
        if colliding and not in_collision:
            collisions += 1
        in_collision = colliding
        path_length += float(np.linalg.norm(true_pos - prev_true_pos))
        prev_true_pos = true_pos.copy()

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
        # VA-ATACOM intervention stats (only meaningful in safe mode)
        'n_filter_calls': n_filter_calls,
        'n_filter_corrected': n_filter_corrected,
        'filter_correction_rate': (n_filter_corrected / n_filter_calls) if n_filter_calls else 0.0,
        'mean_barrier_min': float(np.mean(barrier_min_log)) if barrier_min_log else float('nan'),
        'min_barrier_min': float(np.min(barrier_min_log)) if barrier_min_log else float('nan'),
        'mean_fwd_correction_abs': float(np.mean(fwd_correction_abs_log)) if fwd_correction_abs_log else 0.0,
        'traj_gt': traj_gt,
        'traj_est': traj_est,
    }


def main():
    print("=" * 64, flush=True)
    print("WEBOTS E-PUCK VA-ATACOM INFERENCE (corridor world, AAAI 2027)", flush=True)
    print("=" * 64, flush=True)
    print(f"  HAVE_EKF: {HAVE_EKF}", flush=True)
    print(f"  Filter params: r={OBSTACLE_RADIUS}, d_safe={D_SAFE}, "
          f"a_max={A_MAX}, alpha0={ALPHA0}", flush=True)

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    print(f"  basicTimeStep={timestep}ms  (dt={timestep/1000.0}s)", flush=True)

    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf')); left_motor.setVelocity(0)
    right_motor.setPosition(float('inf')); right_motor.setVelocity(0)
    for i in range(8):
        ds = robot.getDevice(f'ps{i}'); ds.enable(timestep)
    robot_node = robot.getSelf()

    trials = generate_trial_configs(N_TRIALS, seed=SEED)
    print(f"  generated {len(trials)} trials", flush=True)

    all_results = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'world': 'epuck_corridor_va_atacom.wbt',
            'filter': 'BrakeManifoldFilter (VA-ATACOM)',
            'filter_params': {
                'hazard_radius': OBSTACLE_RADIUS, 'safety_margin': D_SAFE,
                'a_max': A_MAX, 'alpha0': ALPHA0, 'action_scale': A_MAX,
                'action_form': 'diff_drive',
            },
            'gps_noise_std': GPS_NOISE_STD, 'heading_noise_std': HEADING_NOISE_STD,
            'n_trials': len(trials), 'max_steps': MAX_STEPS,
        },
        'trial_configs': [[s.tolist(), g.tolist()] for s, g in trials],
        'obstacles': [obs.tolist() for obs in OBSTACLES],
    }

    for mode in ['safe', 'unsafe']:
        print(f"\n{'=' * 64}\nMODE: {mode.upper()}\n{'=' * 64}", flush=True)
        mode_results = []
        rng = np.random.RandomState(SEED)
        for i, (start, goal) in enumerate(trials):
            r = run_trial(robot, robot_node, left_motor, right_motor, timestep,
                          start, goal,
                          use_safety=(mode == 'safe'),
                          use_ekf=(mode == 'safe'), rng=rng)
            mode_results.append(r)
            extra = ''
            if mode == 'safe' and r['n_filter_calls']:
                extra = f"  filt={r['filter_correction_rate']*100:.0f}%  bmin={r['min_barrier_min']:.3f}"
            print(f"  T{i+1:2d}: {'OK' if r['success'] else 'TMO'}  col={r['collisions']}  "
                  f"path={r['path_length']:.2f}m{extra}", flush=True)
        all_results[mode] = mode_results

        out_dir = os.path.join(IROS_DIR, 'results_webots_va_atacom')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'webots_va_atacom_results.json')
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        sys.stdout.flush()

    print(f"\n{'=' * 64}\nSUMMARY\n{'=' * 64}", flush=True)
    for mode in ['safe', 'unsafe']:
        rs = all_results[mode]; n = len(rs)
        succ = sum(r['success'] for r in rs)
        cols = [r['collisions'] for r in rs]
        ptot = sum(c for c in cols)
        ftc = [r['filter_correction_rate'] for r in rs if r['n_filter_calls']]
        print(f"  {mode.upper():>6s}: success={succ}/{n} ({100*succ/n:.0f}%)  "
              f"total_collisions={ptot}  mean_col={np.mean(cols):.2f}±{np.std(cols):.2f}"
              + (f"  mean_filter_correction_rate={np.mean(ftc)*100:.1f}%" if ftc else ''), flush=True)

    print(f"\nResults: {out_path}", flush=True)
    sys.stdout.flush()
    robot.simulationQuit(0)


if __name__ == '__main__':
    main()
