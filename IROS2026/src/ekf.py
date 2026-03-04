"""
Data-Driven Extended Kalman Filter (Section III-C)

EKF with CNN-learned noise parameters for robust state estimation
under sensor uncertainty.

Key features:
    - CNN processes IMU window to predict noise covariance
    - Automatic adaptation to varying conditions
    - More accurate than fixed-parameter EKF
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class EKFConfig:
    """EKF configuration."""
    state_dim: int = 6           # [x, y, theta, vx, vy, omega]
    measurement_dim: int = 3     # [x, y, theta] from sensors
    window_length: int = 10      # IMU history window
    sigma_lat: float = 0.1       # Initial lateral noise std
    sigma_up: float = 0.05       # Initial vertical noise std
    beta: float = 2.0            # Noise adaptation range
    dt: float = 0.1              # Time step


class NoiseAdapter(nn.Module if TORCH_AVAILABLE else object):
    """
    CNN for learning measurement noise covariance from IMU data.

    Input: [batch, window_length, 6] (angular_vel + linear_accel)
    Output: [batch, 2] (z_lat, z_up) -> used to compute N
    """

    def __init__(
        self,
        window_length: int = 10,
        hidden_dim: int = 64,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for NoiseAdapter")

        super().__init__()

        self.conv1 = nn.Conv1d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, imu_window: torch.Tensor) -> torch.Tensor:
        """
        Predict noise parameters from IMU window.

        Args:
            imu_window: [batch, window_length, 6]

        Returns:
            z: [batch, 2] noise parameters
        """
        # [batch, 6, window_length]
        x = imu_window.transpose(1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # [batch, 64]

        z = self.fc(x)  # [batch, 2]

        return z


class DataDrivenEKF:
    """
    Extended Kalman Filter with learned noise parameters.

    State: [x, y, theta, vx, vy, omega]
    Measurement: [x, y, theta] (e.g., from GPS + compass)

    The key innovation is learning N (measurement noise covariance)
    from recent IMU data using a CNN.
    """

    def __init__(
        self,
        config: EKFConfig = None,
        noise_adapter: NoiseAdapter = None,
        device: str = "cpu",
    ):
        self.config = config or EKFConfig()
        self.device = torch.device(device) if TORCH_AVAILABLE else None

        # Noise adapter (CNN)
        if noise_adapter is not None:
            self.noise_adapter = noise_adapter
        elif TORCH_AVAILABLE:
            self.noise_adapter = NoiseAdapter(
                self.config.window_length
            ).to(self.device)
        else:
            self.noise_adapter = None

        # State dimension
        self.n = self.config.state_dim
        self.m = self.config.measurement_dim

        # State estimate and covariance
        self.x_hat = np.zeros(self.n)
        self.P = np.eye(self.n) * 0.1

        # Process noise (constant)
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])

        # Default measurement noise
        self.R = np.diag([
            self.config.sigma_lat ** 2,
            self.config.sigma_lat ** 2,
            self.config.sigma_up ** 2,
        ])

        # IMU history for noise adaptation
        self.imu_history: List[np.ndarray] = []

        # Observation matrix (measure position only)
        self.H = np.zeros((self.m, self.n))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # theta

    def reset(self, initial_state: np.ndarray = None):
        """Reset filter state."""
        if initial_state is not None:
            self.x_hat = np.zeros(self.n)
            self.x_hat[:len(initial_state)] = initial_state
        else:
            self.x_hat = np.zeros(self.n)

        self.P = np.eye(self.n) * 0.1
        self.imu_history = []

    def predict(self, action: np.ndarray, dt: float = None):
        """
        EKF prediction step.

        State transition: x_{t+1} = f(x_t, u_t)
        For differential drive:
            x' = x + v*cos(theta)*dt
            y' = y + v*sin(theta)*dt
            theta' = theta + omega*dt
            vx' = v*cos(theta)
            vy' = v*sin(theta)
            omega' = omega_cmd
        """
        dt = dt or self.config.dt

        x, y, theta, vx, vy, omega = self.x_hat
        v_cmd, omega_cmd = action[0], action[1]

        # State prediction
        x_pred = np.array([
            x + vx * dt,
            y + vy * dt,
            theta + omega * dt,
            v_cmd * np.cos(theta),
            v_cmd * np.sin(theta),
            omega_cmd,
        ])

        # Jacobian of state transition
        F = np.eye(self.n)
        F[0, 3] = dt  # dx/dvx
        F[1, 4] = dt  # dy/dvy
        F[2, 5] = dt  # dtheta/domega
        F[3, 2] = -v_cmd * np.sin(theta)  # dvx/dtheta
        F[4, 2] = v_cmd * np.cos(theta)   # dvy/dtheta

        # Covariance prediction
        P_pred = F @ self.P @ F.T + self.Q

        self.x_hat = x_pred
        self.P = P_pred

    def update(
        self,
        measurement: np.ndarray,
        imu_data: np.ndarray = None,
    ):
        """
        EKF update step with learned noise covariance.

        Args:
            measurement: [x, y, theta] sensor measurement
            imu_data: [6] IMU reading (omega, accel) for noise adaptation
        """
        # Update IMU history
        if imu_data is not None:
            self.imu_history.append(imu_data)
            if len(self.imu_history) > self.config.window_length:
                self.imu_history.pop(0)

        # Adapt measurement noise
        R = self._compute_noise_covariance()

        # Innovation
        y = measurement - self.H @ self.x_hat

        # Normalize angle
        y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x_hat = self.x_hat + K @ y

        # Covariance update
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P

    def _compute_noise_covariance(self) -> np.ndarray:
        """
        Compute measurement noise covariance.

        Uses CNN if IMU history is available, otherwise default.
        """
        if (self.noise_adapter is None or
            len(self.imu_history) < self.config.window_length):
            return self.R

        if not TORCH_AVAILABLE:
            return self.R

        # Prepare IMU window
        imu_window = np.array(self.imu_history[-self.config.window_length:])
        imu_tensor = torch.FloatTensor(imu_window).unsqueeze(0).to(self.device)

        # Get noise parameters
        with torch.no_grad():
            z = self.noise_adapter(imu_tensor).cpu().numpy()[0]

        z_lat, z_up = z[0], z[1]

        # Compute adapted noise (Eq. 17)
        sigma_lat = self.config.sigma_lat * (10 ** (self.config.beta * np.tanh(z_lat)))
        sigma_up = self.config.sigma_up * (10 ** (self.config.beta * np.tanh(z_up)))

        return np.diag([sigma_lat ** 2, sigma_lat ** 2, sigma_up ** 2])

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.x_hat.copy()

    def get_position(self) -> np.ndarray:
        """Get estimated position [x, y, theta]."""
        return self.x_hat[:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Get estimated velocity [vx, vy, omega]."""
        return self.x_hat[3:].copy()


class StandardEKF(DataDrivenEKF):
    """
    Standard EKF with fixed noise parameters.

    Used as baseline comparison for Table V.
    """

    def __init__(
        self,
        config: EKFConfig = None,
        R: np.ndarray = None,
    ):
        super().__init__(config=config, noise_adapter=None)

        # Fixed measurement noise
        if R is not None:
            self.R = R
        else:
            self.R = np.diag([0.1 ** 2, 0.1 ** 2, 0.05 ** 2])

    def _compute_noise_covariance(self) -> np.ndarray:
        """Return fixed noise covariance."""
        return self.R


def train_noise_adapter(
    noise_adapter: NoiseAdapter,
    train_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
) -> List[float]:
    """
    Train noise adapter network using mini-batch gradient descent.

    Args:
        noise_adapter: NoiseAdapter network
        train_data: List of (imu_window, measurement, ground_truth) tuples
        n_epochs: Training epochs
        learning_rate: Learning rate
        batch_size: Mini-batch size

    Returns:
        List of training losses
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for training")

    # Pre-convert to tensors for efficient batching
    imu_all = torch.FloatTensor(np.array([d[0] for d in train_data]))    # [N, W, 6]
    meas_all = torch.FloatTensor(np.array([d[1] for d in train_data]))   # [N, 3]
    gt_all = torch.FloatTensor(np.array([d[2] for d in train_data]))     # [N, 3]

    n_samples = len(train_data)
    optimizer = torch.optim.Adam(noise_adapter.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = indices[start:end]

            imu_batch = imu_all[idx]     # [B, W, 6]
            meas_batch = meas_all[idx]   # [B, 3]
            gt_batch = gt_all[idx]       # [B, 3]

            # Forward pass
            z = noise_adapter(imu_batch)  # [B, 2]

            # Compute predicted noise per sample
            sigma = 0.1 * (10 ** (2.0 * torch.tanh(z)))  # [B, 2]

            # Measurement error
            error = meas_batch - gt_batch  # [B, 3]

            # NLL: 0.5 * (log(sigma^2) + error^2/sigma^2), summed over dims
            nll = 0.5 * (
                2 * torch.log(sigma[:, 0]) + 2 * torch.log(sigma[:, 0])  # x,y share sigma_lat
                + torch.log(sigma[:, 1] ** 2)                             # theta uses sigma_up
                + (error[:, 0] ** 2 + error[:, 1] ** 2) / (sigma[:, 0] ** 2 + 1e-8)
                + error[:, 2] ** 2 / (sigma[:, 1] ** 2 + 1e-8)
            )
            loss = nll.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")

    return losses
