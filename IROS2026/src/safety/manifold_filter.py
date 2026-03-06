"""
Constraint Manifold Filter (Section III-A)

Projects RL actions onto the tangent space of the constraint manifold
using null-space projection, guaranteeing hard constraint satisfaction.

Key equations from paper:
    - Slack variable: c_bar(s, mu) = c(s) + mu = 0,  mu >= 0
    - Safe action: a_safe = N_c @ alpha - K_c @ J_c^+ @ c_bar - J_c^+ @ J_s @ f(s)
    - Null-space projector: N_c = I - J_c^+ @ J_c
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FilterResult:
    """Result of safety filter projection."""
    action_safe: np.ndarray      # Projected safe action
    action_original: np.ndarray  # Original action
    correction: np.ndarray       # Correction vector (a_safe - a_original)
    constraint_value: float      # Current constraint value
    is_corrected: bool           # Whether action was modified


class ManifoldFilter:
    """
    Constraint Manifold Safety Filter.

    Projects actions onto the null-space of the constraint Jacobian,
    ensuring the system stays on the constraint manifold.

    For Point Robot with differential drive:
        - State: [x, y, theta]
        - Action: [v, omega] (forward velocity, angular velocity)
        - Constraint: distance to obstacles >= d_safe
    """

    def __init__(
        self,
        n_constraints: int = 8,
        K_c: float = 10.0,
        epsilon: float = 1e-4,
        d_safe: float = 0.3,
        lambda_calib: float = 0.1,
    ):
        """
        Args:
            n_constraints: Number of obstacle constraints
            K_c: Constraint correction gain (exponential convergence rate)
            epsilon: Regularization for pseudoinverse
            d_safe: Minimum safe distance to obstacles
            lambda_calib: Reward calibration weight
        """
        self.n_constraints = n_constraints
        self.K_c = K_c
        self.epsilon = epsilon
        self.d_safe = d_safe
        self.lambda_calib = lambda_calib

        # State
        self._obstacles: List[np.ndarray] = []
        self._mu: Optional[np.ndarray] = None  # Slack variables

    def reset(self, obstacles: List[np.ndarray]):
        """
        Reset filter with new obstacle positions.

        Args:
            obstacles: List of obstacle positions [x, y]
        """
        self._obstacles = obstacles
        n = len(obstacles)
        # Initialize slack variables: mu = -c so that c + mu = 0
        # mu >= 0 guaranteed since c <= 0 for feasible states
        self._mu = np.ones(n) * 0.5

    def compute_constraint(
        self,
        robot_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Compute constraint values c(s) for all obstacles.

        c_i(s) = d_safe - ||p - o_i|| <= 0  (safe when negative)

        Args:
            robot_pos: Robot position [x, y, theta]

        Returns:
            Constraint values for each obstacle
        """
        pos = robot_pos[:2]
        c = np.zeros(len(self._obstacles))

        for i, obs in enumerate(self._obstacles):
            dist = np.linalg.norm(pos - obs)
            c[i] = self.d_safe - dist  # Negative when safe

        return c

    def compute_constraint_jacobian(
        self,
        robot_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Compute constraint Jacobian J_s = dc/ds.

        For distance constraint c_i = d_safe - ||p - o_i||:
            dc_i/dp = (o_i - p) / ||p - o_i||

        Args:
            robot_pos: Robot position [x, y, theta]

        Returns:
            Jacobian matrix [n_constraints, 2] (only position part)
        """
        pos = robot_pos[:2]
        n = len(self._obstacles)
        J_s = np.zeros((n, 2))

        for i, obs in enumerate(self._obstacles):
            diff = pos - obs
            dist = np.linalg.norm(diff)

            if dist > 1e-6:
                # Gradient of -||p - o||
                J_s[i] = -diff / dist
            else:
                # At obstacle center, gradient undefined
                J_s[i] = np.zeros(2)

        return J_s

    def project(
        self,
        action: np.ndarray,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray = None,
    ) -> FilterResult:
        """
        Project action onto constraint manifold.

        Implements equation (8) from paper:
            a_safe = N_c @ alpha - K_c @ J_c^+ @ c_bar - J_c^+ @ J_s @ f(s)

        Args:
            action: Original action [v, omega]
            robot_pos: Robot position [x, y, theta]
            robot_vel: Robot velocity (for drift compensation)

        Returns:
            FilterResult with safe action and correction info
        """
        if len(self._obstacles) == 0:
            return FilterResult(
                action_safe=action.copy(),
                action_original=action.copy(),
                correction=np.zeros_like(action),
                constraint_value=0.0,
                is_corrected=False,
            )

        # Compute constraint values
        c = self.compute_constraint(robot_pos)

        # Update slack variables
        c_bar = c + self._mu

        # Compute Jacobians
        J_s = self.compute_constraint_jacobian(robot_pos)  # [n, 2]
        J_mu = np.eye(len(self._mu))  # d(c_bar)/d(mu) = I

        # Transform action Jacobian for differential drive
        # Action [v, omega] -> Cartesian velocity [vx, vy]
        theta = robot_pos[2]
        G = np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
        ])  # [2, 2] maps [v, omega] to [vx, vy]

        # Combined Jacobian J_c = [J_s @ G, J_mu]
        J_c = np.hstack([J_s @ G, J_mu])  # [n, 2 + n]

        # Construct augmented reference: u_ref = [a_unsafe, 0]
        u_ref = np.concatenate([action, np.zeros(len(self._obstacles))])

        # Damped pseudoinverse (Remark in paper: introduces O(epsilon) residual)
        J_c_pinv = self._damped_pinv(J_c)

        # Null-space projector
        N_c = np.eye(J_c.shape[1]) - J_c_pinv @ J_c

        # Safe action: u_safe = N_c @ u_ref - J_c^+ @ (J_s @ f(s) + K_c * c_bar)
        # For Point robot: f(s) = 0, so drift term vanishes
        action_safe_aug = N_c @ u_ref - J_c_pinv @ (self.K_c * c_bar)

        # Extract action and slack variable updates
        action_safe = action_safe_aug[:2]
        mu_dot = action_safe_aug[2:]

        # Update slack variables
        dt = 0.1
        self._mu = np.clip(self._mu + mu_dot * dt, 0.0, 10.0)

        # Compute correction
        correction = action_safe - action

        return FilterResult(
            action_safe=action_safe,
            action_original=action.copy(),
            correction=correction,
            constraint_value=float(np.max(c)),
            is_corrected=np.linalg.norm(correction) > 1e-6,
        )

    def calibrate_reward(
        self,
        reward: float,
        result: FilterResult,
    ) -> float:
        """
        Calibrate reward based on correction magnitude (Eq. 11).

        R_calibrated = R - lambda * ||a_unsafe - a_safe||^2
        """
        correction_norm_sq = np.sum(result.correction ** 2)
        return reward - self.lambda_calib * correction_norm_sq

    def _damped_pinv(self, J: np.ndarray) -> np.ndarray:
        """Compute damped pseudoinverse."""
        JJT = J @ J.T
        n = JJT.shape[0]
        return J.T @ np.linalg.inv(JJT + self.epsilon * np.eye(n))


class CartesianManifoldFilter(ManifoldFilter):
    """
    Manifold filter for Cartesian action space.

    Use this when the robot takes [vx, vy] actions directly
    instead of [v, omega].
    """

    def project(
        self,
        action: np.ndarray,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray = None,
    ) -> FilterResult:
        """Project Cartesian action onto constraint manifold."""
        if len(self._obstacles) == 0:
            return FilterResult(
                action_safe=action.copy(),
                action_original=action.copy(),
                correction=np.zeros_like(action),
                constraint_value=0.0,
                is_corrected=False,
            )

        # Compute constraint
        c = self.compute_constraint(robot_pos)
        c_bar = c + self._mu

        # Jacobians
        J_s = self.compute_constraint_jacobian(robot_pos)  # [n, 2]
        J_mu = np.eye(len(self._mu))  # d(c_bar)/d(mu) = I

        # For Cartesian: G = I
        J_c = np.hstack([J_s, J_mu])  # [n, 2 + n]

        # Construct augmented reference: u_ref = [a_unsafe, 0]
        u_ref = np.concatenate([action, np.zeros(len(self._obstacles))])

        # Damped pseudoinverse
        J_c_pinv = self._damped_pinv(J_c)

        # Null-space projector
        N_c = np.eye(J_c.shape[1]) - J_c_pinv @ J_c

        # Safe action: u_safe = N_c @ u_ref - J_c^+ @ (K_c * c_bar)
        action_safe_aug = N_c @ u_ref - J_c_pinv @ (self.K_c * c_bar)

        # Extract
        action_safe = action_safe_aug[:2]
        mu_dot = action_safe_aug[2:]

        # Update slack
        dt = 0.1
        self._mu = np.clip(self._mu + mu_dot * dt, 0.0, 10.0)

        correction = action_safe - action

        return FilterResult(
            action_safe=action_safe,
            action_original=action.copy(),
            correction=correction,
            constraint_value=float(np.max(c)),
            is_corrected=np.linalg.norm(correction) > 1e-6,
        )
