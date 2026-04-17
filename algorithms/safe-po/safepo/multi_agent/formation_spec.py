# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================
"""Formation edge specifications for decentralized RMP leaves.

Each edge is (i, j, d_ij, r_star): RMP hub attaches to robot *i*; the leaf encodes
vector from *i* toward *j* (x_j - x_i) with desired length d_ij and unit direction
r_star in the world frame.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _unit2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = np.linalg.norm(v)
    if n < _EPS:
        raise ValueError("formation_spec: zero-length direction vector")
    return v / n


def _add_pair(
    edges: list[tuple[int, int, float, np.ndarray]],
    i: int,
    j: int,
    d: float,
    r_ij: np.ndarray,
    bidirectional: bool,
) -> None:
    """Append (i,j,d,r_ij); optionally append (j,i,d,r_ji) with r_ji = -r_ij."""
    r_ij = _unit2(r_ij)
    edges.append((i, j, float(d), r_ij.copy()))
    if bidirectional:
        edges.append((j, i, float(d), (-r_ij).copy()))


def build_formation_edges(
    shape: str,
    num_agents: int,
    spacing: float,
    global_direction: np.ndarray | None = None,
    *,
    line_axis: str = "x",
    wedge_half_angle_rad: float | None = None,
) -> list[tuple[int, int, float, np.ndarray]]:
    """
    Build the list of directed formation edges.

    Args:
        shape: ``mesh`` | ``line`` | ``wedge`` | ``circle`` (aliases: full/default for mesh;
            row/line_horizontal for line; v/vee for wedge; ring/polygon for circle).
        num_agents: Number of robots.
        spacing: Characteristic edge length (neighbor spacing along polyline / to apex /
            **chord length** between consecutive agents on the circle for ``circle``).
        global_direction: For ``mesh``: shared desired edge orientation. For ``circle``:
            unit direction from the formation center to agent 0 (sets rotation of the ring).
        line_axis: For ``line``: ``x`` (horizontal row, i→i+1 along +x) or ``y``
            (column along +y).
        wedge_half_angle_rad: Half of the V opening at the apex (apex = agent 0),
            in radians. Default 35° if None.

    Returns:
        List of ``(i, j, distance, r_star)``.

    Notes:
        - ``mesh`` matches legacy behavior: one leaf per unordered pair, only on the
          lower-index robot ``i`` (no mirrored ``j``→``i`` leaf).
        - ``line``, ``wedge``, and ``circle`` use **bidirectional** edges along the ring/chain.
        - ``circle``: regular ``N``-gon on the plane; circumradius
          ``R = spacing / (2 sin(π/N))`` so consecutive vertices are ``spacing`` apart (chord).
    """
    shape = (shape or "mesh").strip().lower()
    spacing = float(spacing)
    if num_agents < 2:
        return []
    if spacing <= 0:
        raise ValueError("formation spacing (e.g. formation_target_distance) must be > 0")

    gd = np.asarray(
        global_direction if global_direction is not None else [0.0, 1.0], dtype=np.float64
    ).reshape(2)
    gdu = _unit2(gd)
    edges: list[tuple[int, int, float, np.ndarray]] = []

    if shape in ("mesh", "full", "default", "complete"):
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                edges.append((i, j, spacing, gdu.copy()))
        return edges

    if shape in ("line", "line_horizontal", "row"):
        axis = (line_axis or "x").strip().lower()
        if axis in ("x", "horizontal", "h"):
            r_star = np.array([1.0, 0.0], dtype=np.float64)
        elif axis in ("y", "vertical", "v"):
            r_star = np.array([0.0, 1.0], dtype=np.float64)
        else:
            raise ValueError(f"Unknown formation_line_axis: {line_axis!r}")
        for i in range(num_agents - 1):
            _add_pair(edges, i, i + 1, spacing, r_star, bidirectional=True)
        return edges

    if shape in ("wedge", "v", "vee"):
        if num_agents == 2:
            r_star = np.array([1.0, 0.0], dtype=np.float64)
            _add_pair(edges, 0, 1, spacing, r_star, bidirectional=True)
            return edges

        ha = float(wedge_half_angle_rad if wedge_half_angle_rad is not None else np.deg2rad(35.0))
        if not (0 < ha < np.pi / 2):
            raise ValueError("formation wedge half-angle must lie in (0, pi/2) radians")

        # Apex = 0, march / goal roughly along +y: followers behind (negative y component).
        nfol = num_agents - 1
        positions: dict[int, np.ndarray] = {0: np.zeros(2, dtype=np.float64)}
        for k in range(1, num_agents):
            theta = -ha + (k - 1) * (2.0 * ha / max(nfol - 1, 1))
            positions[k] = spacing * np.array([np.sin(theta), -np.cos(theta)], dtype=np.float64)

        for k in range(1, num_agents):
            v = positions[k] - positions[0]
            d = float(np.linalg.norm(v))
            _add_pair(edges, 0, k, d, v, bidirectional=True)

        for k in range(1, num_agents - 1):
            v = positions[k + 1] - positions[k]
            d = float(np.linalg.norm(v))
            if d < _EPS:
                continue
            _add_pair(edges, k, k + 1, d, v, bidirectional=True)

        return edges

    if shape in ("circle", "ring", "polygon"):
        # Regular N-gon: spacing = chord length between consecutive vertices.
        chord = spacing
        R_circ = chord / (2.0 * np.sin(np.pi / float(num_agents)))
        phi = float(np.arctan2(gdu[1], gdu[0]))
        positions = np.zeros((num_agents, 2), dtype=np.float64)
        for k in range(num_agents):
            theta = phi + 2.0 * np.pi * k / float(num_agents)
            positions[k] = R_circ * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        for i in range(num_agents):
            j = (i + 1) % num_agents
            v = positions[j] - positions[i]
            d = float(np.linalg.norm(v))
            if d < _EPS:
                continue
            _add_pair(edges, i, j, d, v, bidirectional=True)
        return edges

    raise ValueError(
        f"Unknown formation_shape {shape!r}. "
        "Expected one of: mesh, line, wedge, circle."
    )
