import numpy as np
from typing import List, Tuple


class FormationShape:
    """Compute formation positions and desired inter-agent distance matrix."""

    @staticmethod
    def polygon(n: int, radius: float = 1.0, center: np.ndarray = None) -> np.ndarray:
        """Regular polygon formation. Returns (n, 2) positions."""
        if center is None:
            center = np.zeros(2)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
        return positions + center

    @staticmethod
    def line(n: int, spacing: float = 1.0, center: np.ndarray = None) -> np.ndarray:
        """Line formation along x-axis. Returns (n, 2) positions."""
        if center is None:
            center = np.zeros(2)
        x = np.linspace(-(n - 1) / 2 * spacing, (n - 1) / 2 * spacing, n)
        positions = np.stack([x, np.zeros(n)], axis=1)
        return positions + center

    @staticmethod
    def v_shape(n: int, spacing: float = 1.0, angle: float = np.pi / 4,
                center: np.ndarray = None) -> np.ndarray:
        """V-formation. Agent 0 is leader at the tip. Returns (n, 2) positions."""
        if center is None:
            center = np.zeros(2)
        positions = [np.zeros(2)]
        for i in range(1, n):
            side = 1 if i % 2 == 1 else -1
            rank = (i + 1) // 2
            dx = -rank * spacing * np.cos(angle)
            dy = side * rank * spacing * np.sin(angle)
            positions.append(np.array([dx, dy]))
        return np.array(positions) + center

    @staticmethod
    def get_shape(name: str, n: int, radius: float = 1.0,
                  center: np.ndarray = None) -> np.ndarray:
        """Get formation positions by name."""
        if name in ("polygon", "square", "triangle", "hexagon", "circle"):
            return FormationShape.polygon(n, radius, center)
        elif name == "line":
            return FormationShape.line(n, radius, center)
        elif name == "v":
            return FormationShape.v_shape(n, radius, center=center)
        else:
            raise ValueError(f"Unknown formation shape: {name}")

    @staticmethod
    def desired_distance_matrix(positions: np.ndarray) -> np.ndarray:
        """Compute pairwise desired distance matrix from formation positions.
        Returns (n, n) symmetric matrix."""
        n = positions.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = np.linalg.norm(positions[i] - positions[j])
        return D


class FormationTopology:
    """Adjacency matrix and edge/neighbor queries for formation graphs."""

    def __init__(self, n: int, topology: str = "complete"):
        """
        Args:
            n: number of agents
            topology: "complete", "chain", or "star"
        """
        self.n = n
        self.topology = topology
        self.adj = self._build_adjacency(n, topology)

    @staticmethod
    def _build_adjacency(n: int, topology: str) -> np.ndarray:
        adj = np.zeros((n, n), dtype=bool)
        if topology == "complete":
            adj = np.ones((n, n), dtype=bool)
            np.fill_diagonal(adj, False)
        elif topology == "chain":
            for i in range(n - 1):
                adj[i, i + 1] = True
                adj[i + 1, i] = True
        elif topology == "star":
            for i in range(1, n):
                adj[0, i] = True
                adj[i, 0] = True
        else:
            raise ValueError(f"Unknown topology: {topology}")
        return adj

    def edges(self) -> List[Tuple[int, int]]:
        """Return list of undirected edges (i, j) with i < j."""
        edge_list = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adj[i, j]:
                    edge_list.append((i, j))
        return edge_list

    def neighbors(self, i: int) -> List[int]:
        """Return list of neighbors of agent i."""
        return list(np.where(self.adj[i])[0])
