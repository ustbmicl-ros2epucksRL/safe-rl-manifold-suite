"""
Distance-based Safety Filter (Practical Version)

A simpler safety filter that scales velocity based on distance to obstacles.
More robust for Point Robot in Safety-Gymnasium than the manifold filter
due to the robot's momentum and discrete time steps.

Key features:
    - Progressive velocity scaling in danger zone
    - Hard stop when too close to obstacles
    - Escape logic when inside obstacle
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class DistanceFilterResult:
    """Result of distance-based safety filter."""
    action_safe: np.ndarray      # Safe action
    action_original: np.ndarray  # Original action
    correction: np.ndarray       # Correction vector
    min_distance: float          # Minimum distance to obstacle
    scale_factor: float          # Velocity scale factor applied
    is_corrected: bool           # Whether action was modified


class DistanceFilter:
    """
    Distance-based velocity scaling safety filter.

    Zones:
        - Safe zone (d > danger_radius): No modification
        - Danger zone (stop_radius < d < danger_radius): Progressive scaling
        - Stop zone (d < stop_radius): Force backward movement
    """

    def __init__(
        self,
        danger_radius: float = 0.5,
        stop_radius: float = 0.25,
        hazard_radius: float = 0.2,
        max_forward_vel: float = 1.0,
        max_angular_vel: float = 1.0,
        lambda_calib: float = 0.1,
    ):
        """
        Args:
            danger_radius: Start scaling velocity at this distance
            stop_radius: Force backward at this distance
            hazard_radius: Actual hazard collision radius
            max_forward_vel: Maximum forward velocity
            max_angular_vel: Maximum angular velocity
            lambda_calib: Reward calibration weight
        """
        self.danger_radius = danger_radius
        self.stop_radius = stop_radius
        self.hazard_radius = hazard_radius
        self.max_forward_vel = max_forward_vel
        self.max_angular_vel = max_angular_vel
        self.lambda_calib = lambda_calib

        self._obstacles: List[np.ndarray] = []

    def reset(self, obstacles: List[np.ndarray]):
        """Reset with new obstacle positions."""
        self._obstacles = [np.array(o[:2]) for o in obstacles]

    def compute_min_distance(
        self,
        robot_pos: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute minimum distance to any obstacle.

        Returns:
            min_dist: Minimum distance (negative if inside obstacle)
            away_direction: Unit vector pointing away from nearest obstacle
        """
        if len(self._obstacles) == 0:
            return float('inf'), np.zeros(2)

        pos = robot_pos[:2]
        min_dist = float('inf')
        away_dir = np.zeros(2)

        for obs in self._obstacles:
            diff = pos - obs
            dist = np.linalg.norm(diff) - self.hazard_radius

            if dist < min_dist:
                min_dist = dist
                if np.linalg.norm(diff) > 1e-6:
                    away_dir = diff / np.linalg.norm(diff)
                else:
                    # At center of obstacle, pick random direction
                    away_dir = np.array([1.0, 0.0])

        return min_dist, away_dir

    def project(
        self,
        action: np.ndarray,
        robot_pos: np.ndarray,
        current_velocity: np.ndarray = None,
    ) -> DistanceFilterResult:
        """
        Project action to safe action based on distance.

        Args:
            action: [v, omega] forward velocity and angular velocity
            robot_pos: [x, y, theta] robot pose
            current_velocity: [vx, vy, omega] current velocity (optional)

        Returns:
            DistanceFilterResult with safe action
        """
        action = np.array(action, dtype=np.float64)
        v, omega = action[0], action[1]

        # Get distance to nearest obstacle
        min_dist, away_dir = self.compute_min_distance(robot_pos)

        # Default: no modification
        v_safe = v
        omega_safe = omega
        scale = 1.0

        theta = robot_pos[2]
        heading = np.array([np.cos(theta), np.sin(theta)])

        if min_dist <= 0:
            # Inside obstacle - emergency escape
            heading_away = np.dot(heading, away_dir) > 0.3

            if heading_away and v > 0:
                # Already heading away, allow movement
                v_safe = max(v, 0.3 * self.max_forward_vel)
            else:
                # Force backward
                v_safe = -0.5 * self.max_forward_vel
                # Turn toward escape direction
                cross = heading[0] * away_dir[1] - heading[1] * away_dir[0]
                omega_safe = np.sign(cross) * 0.5 * self.max_angular_vel

            scale = 0.0

        elif min_dist < self.stop_radius:
            # Very close - minimal forward movement
            heading_toward = np.dot(heading, -away_dir) > 0.5

            if heading_toward and v > 0:
                # Moving toward obstacle, stop
                v_safe = 0.0
            else:
                # Allow slow movement away
                v_safe = np.clip(v, -0.3, 0.2) * self.max_forward_vel

            scale = 0.1

        elif min_dist < self.danger_radius:
            # Danger zone - progressive scaling
            t = (min_dist - self.stop_radius) / (self.danger_radius - self.stop_radius)
            scale = 0.2 + 0.8 * t  # Scale from 0.2 to 1.0

            heading_toward = np.dot(heading, -away_dir) > 0.3

            if heading_toward and v > 0:
                # Moving toward obstacle, reduce speed
                v_safe = v * scale * 0.5
            else:
                v_safe = v * scale

        # Clip to limits
        v_safe = np.clip(v_safe, -self.max_forward_vel, self.max_forward_vel)
        omega_safe = np.clip(omega_safe, -self.max_angular_vel, self.max_angular_vel)

        action_safe = np.array([v_safe, omega_safe])
        correction = action_safe - action

        return DistanceFilterResult(
            action_safe=action_safe,
            action_original=action.copy(),
            correction=correction,
            min_distance=min_dist,
            scale_factor=scale,
            is_corrected=np.linalg.norm(correction) > 1e-6,
        )

    def calibrate_reward(
        self,
        reward: float,
        result: DistanceFilterResult,
    ) -> float:
        """
        Calibrate reward based on correction magnitude.

        R_calibrated = R - lambda * ||a_unsafe - a_safe||^2
        """
        correction_norm_sq = np.sum(result.correction ** 2)
        return reward - self.lambda_calib * correction_norm_sq


class AdaptiveDistanceFilter(DistanceFilter):
    """
    Distance filter with adaptive parameters based on velocity.

    Increases safety margin when moving fast.
    """

    def __init__(
        self,
        base_danger_radius: float = 0.5,
        base_stop_radius: float = 0.25,
        velocity_scaling: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            danger_radius=base_danger_radius,
            stop_radius=base_stop_radius,
            **kwargs,
        )
        self.base_danger_radius = base_danger_radius
        self.base_stop_radius = base_stop_radius
        self.velocity_scaling = velocity_scaling

    def project(
        self,
        action: np.ndarray,
        robot_pos: np.ndarray,
        current_velocity: np.ndarray = None,
    ) -> DistanceFilterResult:
        """Project with velocity-adaptive safety margins."""
        # Adjust safety margins based on current velocity
        if current_velocity is not None:
            speed = np.linalg.norm(current_velocity[:2])
            velocity_factor = 1.0 + self.velocity_scaling * speed

            self.danger_radius = self.base_danger_radius * velocity_factor
            self.stop_radius = self.base_stop_radius * velocity_factor

        return super().project(action, robot_pos, current_velocity)
