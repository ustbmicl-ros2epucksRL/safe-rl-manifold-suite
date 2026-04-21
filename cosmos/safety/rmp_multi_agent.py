"""
MultiAgentRMPAdapter — single-process multi-agent wrapper for chap4 GCPL's RMPCorrector.

The underlying RMPCorrector (safepo.multi_agent.rmp_corrector) is designed for
safety-gymnasium's batched vectorized env (num_envs rollout threads, torch tensors,
specific env.task.agent state layout). For Webots supervisors, ROS2 nodes, or any
other single-process multi-agent deployment, we want a thinner API:

    adapter = MultiAgentRMPAdapter(
        num_agents=3, formation_shape='wedge',
        formation_target_distance=0.5, collision_safety_radius=0.3,
        fusion_mode='leaf',      # or 'additive'
        rl_leaf_weight=10.0,      # 0.01 for rmp-only baseline
        # ablation toggles (default: all True = full GCPL):
        use_formation_leaf=True, use_orientation_leaf=True, use_collision_leaf=True,
    )

    # every control step:
    cmd = adapter.step(
        positions=np.array([[x0,y0], [x1,y1], [x2,y2]]),       # shape (N, 2)
        velocities=np.array([[vx0,vy0], ...]),                  # shape (N, 2)
        headings=np.array([theta0, theta1, theta2]),            # shape (N,) radians
        rl_actions=np.zeros((3, 2)),                             # shape (N, 2); zeros = rmp-only
    )
    # cmd: np.ndarray (N, 2) of (v, omega) per agent, clipped to [-1, 1]

Use cases:
    * chap5 Webots supervisor  — collect positions from Supervisor, send wheel speeds
    * chap5 E-puck ROS2 node   — /agent_i/odom in, /agent_i/cmd_vel out
    * chap4 analysis scripts   — trajectory rollouts with custom dynamics
    * Quick sanity tests       — no safety-gymnasium env needed

Dependencies:
    safepo (editable install in `safe-rl-manifold-suite/algorithms/safe-po/`)
    rmp_leaf / rmp (MULTI_ROBOT_RMPFLOW_PATH env var must point to
                    `safe-rl-manifold-suite/algorithms/multi-robot-rmpflow/`)

Notes:
    * torch import is done lazily inside RMPCorrector; we stay pure-numpy at the API.
    * Differential-drive mapping (ax, ay) -> (v, omega) is handled by the corrector
      when fusion_mode='leaf' and use_diff_drive_mapping=True (default).
    * For fusion_mode='additive' the return of step() is already world-frame (ax, ay)
      scaled like a_RL + rmp_weight * a_RMP; caller must apply their own diff-drive map.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np


def _ensure_rmpflow_on_path() -> None:
    """Make sure MULTI_ROBOT_RMPFLOW_PATH (or suite default) is on sys.path."""
    p = os.environ.get("MULTI_ROBOT_RMPFLOW_PATH")
    if not p:
        # Best-effort default to suite layout
        suite = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        candidate = os.path.join(suite, "algorithms", "multi-robot-rmpflow")
        if os.path.isdir(candidate):
            p = candidate
            os.environ["MULTI_ROBOT_RMPFLOW_PATH"] = p
    if p and p not in sys.path:
        sys.path.insert(0, p)


_ensure_rmpflow_on_path()


class MultiAgentRMPAdapter:
    """Thin single-env wrapper over safepo's RMPCorrector.

    Parameters mirror chap4 config keys; see chap4.tex §4.3-4.5 and chap4-sweep-guide.md.

    The adapter builds one RMP tree internally (num_envs=1) and exposes a plain
    numpy step() that returns per-agent (v, omega) if fusion_mode='leaf' (the default,
    which includes the diff-drive mapping), or per-agent (ax, ay) if 'additive'.
    """

    def __init__(
        self,
        num_agents: int,
        formation_shape: str = "wedge",
        formation_target_distance: float = 0.5,
        formation_line_axis: str = "x",
        formation_wedge_half_angle_deg: float = 35.0,
        formation_desired_direction: Optional[list] = None,
        formation_orientation_alpha: float = 1.0,
        formation_orientation_eta: float = 2.0,
        formation_orientation_c_metric: float = 10.0,
        collision_safety_radius: float = 0.3,
        sigwall_centers: Optional[list] = None,
        fusion_mode: str = "leaf",
        rl_leaf_weight: float = 10.0,
        use_diff_drive_mapping: bool = True,
        diff_drive_k_theta: float = 1.0,
        use_formation_leaf: bool = True,
        use_orientation_leaf: bool = True,
        use_collision_leaf: bool = True,
        device: str = "cpu",
    ):
        from safepo.multi_agent.rmp_corrector import RMPCorrector

        self.num_agents = num_agents
        self.fusion_mode = fusion_mode.strip().lower()

        cfg = {
            "use_rmp": True,
            "formation_shape": formation_shape,
            "formation_target_distance": formation_target_distance,
            "formation_line_axis": formation_line_axis,
            "formation_wedge_half_angle_deg": formation_wedge_half_angle_deg,
            "formation_desired_direction": (
                formation_desired_direction if formation_desired_direction is not None
                else [0.0, 1.0]
            ),
            "formation_orientation_alpha": formation_orientation_alpha,
            "formation_orientation_eta": formation_orientation_eta,
            "formation_orientation_c_metric": formation_orientation_c_metric,
            "collision_safety_radius": collision_safety_radius,
            "fusion_mode": self.fusion_mode,
            "rl_leaf_weight": rl_leaf_weight,
            "use_diff_drive_mapping": use_diff_drive_mapping,
            "diff_drive_k_theta": diff_drive_k_theta,
            "use_formation_leaf": use_formation_leaf,
            "use_orientation_leaf": use_orientation_leaf,
            "use_collision_leaf": use_collision_leaf,
        }
        if sigwall_centers is not None:
            cfg["sigwall_centers"] = sigwall_centers

        self._corrector = RMPCorrector(
            num_agents=num_agents,
            num_envs=1,
            device=device,
            config=cfg,
        )

    # ---- shape validation helpers ----
    def _prep_state(self, arr, shape, name):
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")
        return a

    # ---- main API ----
    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        headings: Optional[np.ndarray] = None,
        rl_actions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run one RMP correction step.

        Args:
            positions:  shape (N, 2)   world-frame (x, y) in meters
            velocities: shape (N, 2)   world-frame (vx, vy) in m/s
            headings:   shape (N,)     yaw θ in radians (optional but recommended
                        in fusion_mode='leaf' so diff-drive map uses real heading;
                        if None, RMPCorrector falls back to velocity direction)
            rl_actions: shape (N, 2)   RL policy's expected acceleration per agent.
                        For pure RMPflow baseline pass zeros (default).

        Returns:
            np.ndarray shape (N, 2)
                * if fusion_mode='leaf' and use_diff_drive_mapping=True:
                  per agent (v_cmd, omega_cmd) clipped to [-1, 1]
                * otherwise: per agent (ax, ay) world-frame acceleration
        """
        N = self.num_agents
        positions = self._prep_state(positions, (N, 2), "positions")
        velocities = self._prep_state(velocities, (N, 2), "velocities")
        if rl_actions is None:
            rl_actions = np.zeros((N, 2), dtype=np.float64)
        else:
            rl_actions = self._prep_state(rl_actions, (N, 2), "rl_actions")
        if headings is not None:
            h = np.asarray(headings, dtype=np.float64)
            if h.shape != (N,):
                raise ValueError(f"headings: expected shape ({N},), got {h.shape}")

        # RMPCorrector expects List[per-agent] of arrays shape [n_envs, ...]
        # With n_envs=1 we just wrap each (N, d) into list of length N with [1, d] arrays.
        actions_list = [rl_actions[i : i + 1] for i in range(N)]
        pos_list = [positions[i : i + 1] for i in range(N)]
        vel_list = [velocities[i : i + 1] for i in range(N)]
        head_list = None
        if headings is not None:
            head_list = [np.asarray(headings[i : i + 1], dtype=np.float64) for i in range(N)]

        corrected = self._corrector.apply_correction(
            actions=actions_list,
            agent_positions=pos_list,
            agent_velocities=vel_list,
            agent_headings=head_list,
        )
        # corrected is List[per-agent] of arrays/tensors shape [1, 2]; pack back.
        out = np.zeros((N, 2), dtype=np.float64)
        for i in range(N):
            a = corrected[i][0]
            if hasattr(a, "detach"):
                a = a.detach().cpu().numpy()
            out[i] = np.asarray(a, dtype=np.float64).reshape(-1)[:2]
        return out

    # ---- convenience factories ----
    @classmethod
    def gcpl(cls, num_agents: int, **kw) -> "MultiAgentRMPAdapter":
        """Factory for full GCPL (default config)."""
        return cls(num_agents=num_agents, **kw)

    @classmethod
    def rmp_only(cls, num_agents: int, **kw) -> "MultiAgentRMPAdapter":
        """Factory for RMPflow-standalone baseline: rl_leaf_weight ~ 0."""
        kw.setdefault("rl_leaf_weight", 0.01)
        kw.setdefault("fusion_mode", "leaf")
        return cls(num_agents=num_agents, **kw)
