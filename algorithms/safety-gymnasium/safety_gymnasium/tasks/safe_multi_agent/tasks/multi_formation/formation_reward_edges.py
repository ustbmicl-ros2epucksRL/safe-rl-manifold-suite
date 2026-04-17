# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
# ==============================================================================
"""Nominal pairwise distance constraints for formation reward (topology matches RMP)."""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _unit2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        raise ValueError('formation_reward_edges: zero-length direction vector')
    return v / n


def _dedupe_undirected(
    segments: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Merge duplicate undirected edges; keep one target distance (they match by construction)."""
    best: dict[tuple[int, int], float] = {}
    for i, j, d in segments:
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        d = float(d)
        if (a, b) not in best:
            best[(a, b)] = d
    return [(a, b, best[(a, b)]) for a, b in sorted(best.keys())]


def _add_both(
    out: list[tuple[int, int, float]],
    i: int,
    j: int,
    d: float,
    r_ij: np.ndarray,
) -> None:
    r_ij = _unit2(r_ij)
    out.append((i, j, float(d)))
    out.append((j, i, float(d)))


def build_formation_distance_constraints(
    shape: str,
    num_agents: int,
    spacing: float,
    global_direction: np.ndarray | None = None,
    *,
    line_axis: str = 'x',
    wedge_half_angle_rad: float | None = None,
) -> tuple[list[tuple[int, int, float]], np.ndarray | None]:
    """
    Returns:
        unique_edges: list of (i, j, d_target) with i < j.
        mesh_r_star: unit preferred edge direction for mesh alignment reward; else None.
    """
    shape = (shape or 'mesh').strip().lower()
    spacing = float(spacing)
    if num_agents < 2 or spacing <= 0:
        return [], None

    gd = np.asarray(
        global_direction if global_direction is not None else [0.0, 1.0],
        dtype=np.float64,
    ).reshape(2)
    gdu = _unit2(gd)

    if shape in ('mesh', 'full', 'default', 'complete'):
        edges = [(i, j, spacing) for i in range(num_agents) for j in range(i + 1, num_agents)]
        return edges, gdu.copy()

    if shape in ('line', 'line_horizontal', 'row'):
        axis = (line_axis or 'x').strip().lower()
        if axis in ('x', 'horizontal', 'h'):
            r_star = np.array([1.0, 0.0], dtype=np.float64)
        elif axis in ('y', 'vertical', 'v'):
            r_star = np.array([0.0, 1.0], dtype=np.float64)
        else:
            raise ValueError(f'Unknown formation_line_axis: {line_axis!r}')
        raw: list[tuple[int, int, float]] = []
        for i in range(num_agents - 1):
            _add_both(raw, i, i + 1, spacing, r_star)
        return _dedupe_undirected(raw), None

    if shape in ('wedge', 'v', 'vee'):
        if num_agents == 2:
            r_star = np.array([1.0, 0.0], dtype=np.float64)
            raw = []
            _add_both(raw, 0, 1, spacing, r_star)
            return _dedupe_undirected(raw), None

        ha = float(
            wedge_half_angle_rad if wedge_half_angle_rad is not None else np.deg2rad(35.0),
        )
        if not (0 < ha < np.pi / 2):
            raise ValueError('formation wedge half-angle must lie in (0, pi/2) radians')

        nfol = num_agents - 1
        positions: dict[int, np.ndarray] = {0: np.zeros(2, dtype=np.float64)}
        for k in range(1, num_agents):
            theta = -ha + (k - 1) * (2.0 * ha / max(nfol - 1, 1))
            positions[k] = spacing * np.array([np.sin(theta), -np.cos(theta)], dtype=np.float64)

        raw = []
        for k in range(1, num_agents):
            v = positions[k] - positions[0]
            d = float(np.linalg.norm(v))
            _add_both(raw, 0, k, d, v)

        for k in range(1, num_agents - 1):
            v = positions[k + 1] - positions[k]
            d = float(np.linalg.norm(v))
            if d < _EPS:
                continue
            _add_both(raw, k, k + 1, d, v)

        return _dedupe_undirected(raw), None

    if shape in ('circle', 'ring', 'polygon'):
        chord = spacing
        r_circ = chord / (2.0 * np.sin(np.pi / float(num_agents)))
        phi = float(np.arctan2(gdu[1], gdu[0]))
        positions = np.zeros((num_agents, 2), dtype=np.float64)
        for k in range(num_agents):
            theta = phi + 2.0 * np.pi * k / float(num_agents)
            positions[k] = r_circ * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        raw = []
        for i in range(num_agents):
            j = (i + 1) % num_agents
            v = positions[j] - positions[i]
            d = float(np.linalg.norm(v))
            if d < _EPS:
                continue
            _add_both(raw, i, j, d, v)
        return _dedupe_undirected(raw), None

    raise ValueError(
        f'Unknown formation_shape {shape!r}. Expected mesh, line, wedge, or circle.',
    )
