#!/usr/bin/env python3
"""
Safety navigation demo controller for E-puck.
Auto-detects obstacles from world file via Supervisor API.
Uses waypoint navigation + APF avoidance + safety filter + EKF.
"""

import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IROS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, os.path.join(IROS_DIR, 'src'))

from controller import Supervisor
from safety.distance_filter import DistanceFilter
from ekf import StandardEKF, EKFConfig

# E-puck physical parameters
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
ROBOT_RADIUS = 0.037
MAX_WHEEL_SPEED = 6.28
MAX_LINEAR_SPEED = WHEEL_RADIUS * MAX_WHEEL_SPEED  # ~0.1287 m/s
OBSTACLE_RADIUS = 0.10
GOAL_THRESHOLD = 0.15

START = np.array([-0.7, -0.7])
DEFAULT_GOAL = np.array([0.5, 0.5])
GPS_NOISE_STD = 0.03
HEADING_NOISE_STD = 0.04

WP_REACH_DIST = 0.12

# Movie recording settings
RECORD_MOVIE = True
MOVIE_WIDTH = 1280
MOVIE_HEIGHT = 720
MOVIE_QUALITY = 100


def get_true_state(robot_node):
    pos = robot_node.getPosition()
    rot = robot_node.getOrientation()
    heading = np.arctan2(rot[3], rot[0])
    return np.array([pos[0], pos[1]]), heading


def set_velocity(left_motor, right_motor, v, omega):
    v = np.clip(v, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
    omega = np.clip(omega, -5.0, 5.0)
    vl = (v - omega * AXLE_LENGTH / 2) / WHEEL_RADIUS
    vr = (v + omega * AXLE_LENGTH / 2) / WHEEL_RADIUS
    left_motor.setVelocity(float(np.clip(vl, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)))
    right_motor.setVelocity(float(np.clip(vr, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)))


def compute_waypoints(start, goal, obstacles):
    """Compute waypoints that route around each obstacle.

    Places waypoints on the path axis but offset perpendicular to avoid
    each obstacle. Waypoints are placed BETWEEN obstacles so the robot
    approaches each from the correct side.
    """
    if not obstacles:
        return [goal]

    path_vec = goal - start
    path_len = np.linalg.norm(path_vec)
    path_dir = path_vec / (path_len + 1e-8)
    perp_right = np.array([path_dir[1], -path_dir[0]])

    obs_sorted = sorted(obstacles,
                        key=lambda o: np.dot(o - start, path_dir))

    path_obs = []
    for obs in obs_sorted:
        proj = np.dot(obs - start, path_dir)
        perp_dist = abs(np.dot(obs - start, perp_right))
        if 0.1 < proj < path_len - 0.1 and perp_dist < 0.5:
            path_obs.append(obs)

    waypoints = []
    offset = 0.10  # perpendicular offset from path centerline
    forward = 0.10  # push waypoint forward past obstacle

    for obs in path_obs:
        # Project obstacle onto path to find the path point
        obs_proj = np.dot(obs - start, path_dir)
        path_point = start + (obs_proj + forward) * path_dir

        # Detect which side of path the obstacle is on
        obs_perp = np.dot(obs - start, perp_right)
        if obs_perp >= 0:
            side = -1  # obstacle on right of path, waypoint on left
        else:
            side = 1   # obstacle on left of path, waypoint on right

        wp = path_point + side * offset * perp_right
        waypoints.append(wp)

    waypoints.append(goal)
    return waypoints


def apf_controller(position, heading, target, obstacles):
    """APF toward target with obstacle avoidance."""
    to_target = target - position
    dist_target = np.linalg.norm(to_target)
    if dist_target < 0.05:
        return 0.0, 0.0

    target_dir = to_target / (dist_target + 1e-8)
    f_att = target_dir * min(dist_target, 0.5)

    f_obs = np.zeros(2)
    avoidance_range = 0.22

    for obs in obstacles:
        to_obs = obs - position
        center_dist = np.linalg.norm(to_obs)
        gap = center_dist - OBSTACLE_RADIUS

        if gap < avoidance_range and center_dist > 0.01:
            eta = 1.5
            safe_gap = max(gap, 0.02)
            rep_strength = eta * (1.0 / safe_gap - 1.0 / avoidance_range) / (safe_gap ** 2)

            away_dir = -to_obs / center_dist
            tang1 = np.array([-to_obs[1], to_obs[0]]) / center_dist
            tang2 = -tang1
            tang = tang1 if np.dot(tang1, target_dir) > np.dot(tang2, target_dir) else tang2

            f_obs += rep_strength * (0.3 * away_dir + 0.7 * tang)

    obs_norm = np.linalg.norm(f_obs)
    if obs_norm > 5.0:
        f_obs = f_obs / obs_norm * 5.0

    if dist_target < 0.3:
        f_obs *= max(0.1, dist_target / 0.3)

    f_total = f_att + f_obs
    f_mag = np.linalg.norm(f_total)

    if f_mag < 1e-6:
        desired_heading = np.arctan2(target_dir[1], target_dir[0])
    else:
        desired_heading = np.arctan2(f_total[1], f_total[0])

    heading_error = np.arctan2(np.sin(desired_heading - heading),
                               np.cos(desired_heading - heading))

    v = min(0.6 * dist_target, MAX_LINEAR_SPEED * 0.9)
    turn_factor = max(0.2, 1.0 - abs(heading_error) / np.pi)
    v *= turn_factor

    omega = 4.5 * heading_error
    return v, omega


def detect_obstacles(robot):
    obstacles = []
    i = 0
    while True:
        node = robot.getFromDef(f'OBS{i}')
        if node is None:
            break
        pos = node.getField('translation').getSFVec3f()
        obstacles.append(np.array([pos[0], pos[1]]))
        i += 1
    return obstacles


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    obstacles = detect_obstacles(robot)
    # Auto-detect goal position from world file
    goal_node = robot.getFromDef('GOAL')
    if goal_node:
        gp = goal_node.getField('translation').getSFVec3f()
        GOAL = np.array([gp[0], gp[1]])
    else:
        GOAL = DEFAULT_GOAL
    waypoints = compute_waypoints(START, GOAL, obstacles)

    print("=" * 50)
    print("SAFETY NAVIGATION DEMO")
    print(f"Start: {START}, Goal: {GOAL}")
    print(f"Obstacles: {len(obstacles)}, Waypoints: {len(waypoints)}")
    for i, obs in enumerate(obstacles):
        print(f"  OBS{i}: ({obs[0]:.2f}, {obs[1]:.2f})")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i}: ({wp[0]:.2f}, {wp[1]:.2f})")
    print("=" * 50)

    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

    ps = []
    for i in range(8):
        sensor = robot.getDevice(f'ps{i}')
        sensor.enable(timestep)
        ps.append(sensor)

    robot_node = robot.getSelf()
    rng = np.random.RandomState(42)

    trans_field = robot_node.getField('translation')
    rot_field = robot_node.getField('rotation')
    initial_heading = np.arctan2(GOAL[1] - START[1], GOAL[0] - START[0])
    trans_field.setSFVec3f([float(START[0]), float(START[1]), 0.0])
    rot_field.setSFRotation([0.0, 0.0, 1.0, float(initial_heading)])
    robot_node.resetPhysics()
    set_velocity(left_motor, right_motor, 0, 0)
    for _ in range(10):
        robot.step(timestep)

    # Auto movie recording via Supervisor API
    movie_path = None
    if RECORD_MOVIE:
        # Derive scene name from world file
        world_file = robot.getWorldPath()
        scene_name = os.path.splitext(os.path.basename(world_file))[0]
        movie_path = os.path.join(IROS_DIR, f'webots_demo_{scene_name}.mp4')
        robot.movieStartRecording(movie_path, MOVIE_WIDTH, MOVIE_HEIGHT,
                                   0, MOVIE_QUALITY, 1, False)
        print(f"Recording movie to: {movie_path}")
        # Let a few frames render before robot moves
        for _ in range(30):
            robot.step(timestep)

    safety_filter = DistanceFilter(
        hazard_radius=OBSTACLE_RADIUS + ROBOT_RADIUS,
        danger_radius=0.08,
        stop_radius=0.03,
        max_forward_vel=MAX_LINEAR_SPEED,
        max_angular_vel=4.0,
        lambda_calib=0.0,
    )
    safety_filter.reset(obstacles)

    ekf = StandardEKF(config=EKFConfig(dt=dt, sigma_lat=GPS_NOISE_STD,
                                        sigma_up=HEADING_NOISE_STD))
    true_pos, true_heading = get_true_state(robot_node)
    ekf.reset(np.array([true_pos[0], true_pos[1], true_heading]))

    prev_action = np.array([0.0, 0.0])
    max_steps = 4000
    current_wp = 0
    stuck_counter = 0
    prev_pos = true_pos.copy()

    for step in range(max_steps):
        if robot.step(timestep) == -1:
            break

        true_pos, true_heading = get_true_state(robot_node)
        noisy_pos = true_pos + rng.randn(2) * GPS_NOISE_STD
        noisy_heading = true_heading + rng.randn() * HEADING_NOISE_STD

        ekf.predict(prev_action, dt=dt)
        ekf.update(np.array([noisy_pos[0], noisy_pos[1], noisy_heading]))
        est = ekf.get_position()
        nav_pos, nav_heading = est[:2], est[2]

        if np.linalg.norm(true_pos - GOAL) < GOAL_THRESHOLD:
            print(f"GOAL REACHED at step {step}!")
            set_velocity(left_motor, right_motor, 0, 0)
            # Linger at goal for a few seconds so viewer can see arrival
            for _ in range(150):
                robot.step(timestep)
            break

        # Advance waypoint
        while (current_wp < len(waypoints) - 1 and
               np.linalg.norm(nav_pos - waypoints[current_wp]) < WP_REACH_DIST):
            current_wp += 1
            print(f"  -> WP{current_wp} ({waypoints[current_wp][0]:.2f},{waypoints[current_wp][1]:.2f})")

        target = waypoints[current_wp]
        v, omega = apf_controller(nav_pos, nav_heading, target, obstacles)

        # Reactive sensor avoidance
        sensor_vals = [ps[i].getValue() for i in range(8)]
        front_intensity = sensor_vals[0] + sensor_vals[7]
        if front_intensity > 160.0:
            right_intensity = sensor_vals[0] + sensor_vals[1] + 0.5 * sensor_vals[2]
            left_intensity = sensor_vals[7] + sensor_vals[6] + 0.5 * sensor_vals[5]
            v = min(v, MAX_LINEAR_SPEED * 0.3)
            if right_intensity > left_intensity:
                omega = max(omega, 3.0)
            else:
                omega = min(omega, -3.0)

        # Stuck detection (every 60 steps, skip after 5)
        if step % 60 == 0 and step > 0:
            displacement = np.linalg.norm(true_pos - prev_pos)
            if displacement < 0.01:
                stuck_counter += 1
                if stuck_counter >= 5 and current_wp < len(waypoints) - 1:
                    current_wp += 1
                    print(f"  -> skip to WP{current_wp}")
                    stuck_counter = 0
                else:
                    omega = 4.0 * (1 if stuck_counter % 2 == 0 else -1)
                    v = MAX_LINEAR_SPEED * 0.5
            else:
                stuck_counter = 0
            prev_pos = true_pos.copy()

        # Safety filter
        robot_pose = np.array([nav_pos[0], nav_pos[1], nav_heading])
        result = safety_filter.project(np.array([v, omega]), robot_pose)
        v_safe = float(result.action_safe[0])
        omega_safe = float(result.action_safe[1])

        prev_action = np.array([v_safe, omega_safe])
        set_velocity(left_motor, right_motor, v_safe, omega_safe)

        if step % 100 == 0:
            d = np.linalg.norm(true_pos - GOAL)
            corr = np.linalg.norm(result.correction)
            print(f"  step {step:4d}: pos=({true_pos[0]:.2f},{true_pos[1]:.2f}) "
                  f"d_goal={d:.3f}m wp={current_wp} corr={corr:.4f}")

    # Stop movie recording
    if RECORD_MOVIE and movie_path:
        robot.movieStopRecording()
        # Wait for movie to finish writing
        while not robot.movieIsReady():
            robot.step(timestep)
        print(f"Movie saved to: {movie_path}")

    print("Demo complete.")
    robot.simulationQuit(0)


if __name__ == '__main__':
    main()
