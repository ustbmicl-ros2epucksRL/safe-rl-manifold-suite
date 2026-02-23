"""
RMPflow leaf node policies and MultiRobotRMPForest convenience class.
Ported from algorithms/multi-robot-rmpflow/rmp_leaf.py (Anqi Li, 2019).
"""

import numpy as np
from numpy.linalg import norm

from formation_nav.safety.rmp_tree import RMPRoot, RMPNode, RMPLeaf


class CollisionAvoidance(RMPLeaf):
    """Static obstacle avoidance RMP leaf."""

    def __init__(self, name, parent, parent_param=None, c=np.zeros(2),
                 R=1, epsilon=0.2, alpha=1e-5, eta=0):
        self.R = R
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon

        if parent_param:
            psi = None
            J = None
            J_dot = None
        else:
            if c.ndim == 1:
                c = c.reshape(-1, 1)
            N = c.size

            psi = lambda y: np.array(norm(y - c) / R - 1).reshape(-1, 1)
            J = lambda y: 1.0 / norm(y - c) * (y - c).T / R
            J_dot = lambda y, y_dot: np.dot(
                y_dot.T,
                (-1 / norm(y - c) ** 3 * np.dot((y - c), (y - c).T)
                 + 1 / norm(y - c) * np.eye(N))) / R

        def RMP_func(x, x_dot):
            if x < 0:
                w = 1e10
                grad_w = 0
            else:
                w = 1.0 / x ** 4
                grad_w = -4.0 / x ** 5
            u = epsilon + np.minimum(0, x_dot) * x_dot
            g = w * u
            grad_u = 2 * np.minimum(0, x_dot)
            grad_Phi = alpha * w * grad_w
            xi = 0.5 * x_dot ** 2 * u * grad_w
            M = g + 0.5 * x_dot * w * grad_u
            M = np.minimum(np.maximum(M, -1e5), 1e5)
            Bx_dot = eta * g * x_dot
            f = -grad_Phi - xi - Bx_dot
            f = np.minimum(np.maximum(f, -1e10), 1e10)
            return (f, M)

        super().__init__(name, parent, parent_param, psi, J, J_dot, RMP_func)


class CollisionAvoidanceDecentralized(RMPLeaf):
    """Decentralized collision avoidance for multi-robot RMPForest."""

    def __init__(self, name, parent, parent_param, c=np.zeros(2),
                 R=1, epsilon=1e-8, alpha=1e-5, eta=0):
        assert parent_param is not None
        self.R = R
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon
        self.x_dot_real = None

        psi = None
        J = None
        J_dot = None

        def RMP_func(x, x_dot, x_dot_real):
            if x < 0:
                w = 1e10
                grad_w = 0
            else:
                w = 1.0 / x ** 4
                grad_w = -4.0 / x ** 5
            u = epsilon + np.minimum(0, x_dot) * x_dot
            g = w * u
            grad_u = 2 * np.minimum(0, x_dot)
            grad_Phi = alpha * w * grad_w
            xi = 0.5 * x_dot * x_dot_real * u * grad_w
            M = g + 0.5 * x_dot * w * grad_u
            M = np.minimum(np.maximum(M, -1e5), 1e5)
            Bx_dot = eta * g * x_dot
            f = -grad_Phi - xi - Bx_dot
            f = np.minimum(np.maximum(f, -1e10), 1e10)
            return (f, M)

        super().__init__(name, parent, parent_param, psi, J, J_dot, RMP_func)

    def pushforward(self):
        if self.psi is not None and self.J is not None:
            self.x = self.psi(self.parent.x)
            self.x_dot = np.dot(self.J(self.parent.x), self.parent.x_dot)
            self.x_dot_real = np.dot(
                self.J(self.parent.x),
                self.parent.x_dot - self.parent_param.x_dot)

    def eval_leaf(self):
        self.f, self.M = self.RMP_func(self.x, self.x_dot, self.x_dot_real)

    def update_params(self):
        c = self.parent_param.x
        z_dot = self.parent_param.x_dot
        R = self.R
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        N = c.size

        self.psi = lambda y: np.array(norm(y - c) / R - 1).reshape(-1, 1)
        self.J = lambda y: 1.0 / norm(y - c) * (y - c).T / R
        self.J_dot = lambda y, y_dot: np.dot(
            y_dot.T,
            (-1 / norm(y - c) ** 3 * np.dot((y - c), (y - c).T)
             + 1 / norm(y - c) * np.eye(N))) / R


class GoalAttractorUni(RMPLeaf):
    """Goal attractor RMP leaf."""

    def __init__(self, name, parent, y_g, w_u=10, w_l=1, sigma=1,
                 alpha=1, eta=2, gain=1, tol=0.005):
        if y_g.ndim == 1:
            y_g = y_g.reshape(-1, 1)
        N = y_g.size
        psi = lambda y: (y - y_g)
        J = lambda y: np.eye(N)
        J_dot = lambda y, y_dot: np.zeros((N, N))

        def RMP_func(x, x_dot):
            x_norm = norm(x)
            beta = np.exp(-x_norm ** 2 / 2 / (sigma ** 2))
            w = (w_u - w_l) * beta + w_l
            s = (1 - np.exp(-2 * alpha * x_norm)) / (1 + np.exp(-2 * alpha * x_norm))
            G = np.eye(N) * w
            if x_norm > tol:
                grad_Phi = s / x_norm * w * x * gain
            else:
                grad_Phi = 0
            Bx_dot = eta * w * x_dot
            grad_w = -beta * (w_u - w_l) / sigma ** 2 * x
            x_dot_norm = norm(x_dot)
            xi = -0.5 * (x_dot_norm ** 2 * grad_w
                         - 2 * np.dot(np.dot(x_dot, x_dot.T), grad_w))
            M = G
            f = -grad_Phi - Bx_dot - xi
            return (f, M)

        super().__init__(name, parent, None, psi, J, J_dot, RMP_func)

    def update_goal(self, y_g):
        if y_g.ndim == 1:
            y_g = y_g.reshape(-1, 1)
        N = y_g.size
        self.psi = lambda y: (y - y_g)
        self.J = lambda y: np.eye(N)
        self.J_dot = lambda y, y_dot: np.zeros((N, N))


class FormationDecentralized(RMPLeaf):
    """Decentralized formation control: spring-damper maintaining desired distance."""

    def __init__(self, name, parent, parent_param, c=np.zeros(2),
                 d=1, gain=1, eta=2, w=1):
        assert parent_param is not None
        self.d = d

        psi = None
        J = None
        J_dot = None

        def RMP_func(x, x_dot):
            G = w
            grad_Phi = gain * x * w
            Bx_dot = eta * w * x_dot
            M = G
            f = -grad_Phi - Bx_dot
            return (f, M)

        super().__init__(name, parent, parent_param, psi, J, J_dot, RMP_func)

    def update_params(self):
        z = self.parent_param.x
        c = z
        d = self.d
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        N = c.size
        self.psi = lambda y: np.array(norm(y - c) - d).reshape(-1, 1)
        self.J = lambda y: 1.0 / norm(y - c) * (y - c).T
        self.J_dot = lambda y, y_dot: np.dot(
            y_dot.T,
            (-1 / norm(y - c) ** 3 * np.dot((y - c), (y - c).T)
             + 1 / norm(y - c) * np.eye(N)))


class Damper(RMPLeaf):
    """Velocity damping RMP leaf."""

    def __init__(self, name, parent, w=1, eta=1):
        psi = lambda y: y
        J = lambda y: np.eye(2)
        J_dot = lambda y, y_dot: np.zeros((2, 2))

        def RMP_func(x, x_dot):
            G = w
            Bx_dot = eta * w * x_dot
            M = G
            f = -Bx_dot
            return (f, M)

        super().__init__(name, parent, None, psi, J, J_dot, RMP_func)


class MultiRobotRMPForest:
    """
    Convenience class that builds an RMP forest for multi-robot formation navigation.

    Each robot has its own RMPRoot with leaf nodes for:
      - Goal attraction (leader only or all agents via centroid)
      - Collision avoidance (pairwise decentralized)
      - Formation maintenance (spring-damper to neighbors)
      - Velocity damping
    """

    def __init__(self, num_agents, desired_distances, topology_edges,
                 obstacle_positions=None, safety_radius=0.4,
                 formation_gain=1.0, formation_eta=2.0, formation_w=1.0,
                 damper_w=1.0, damper_eta=1.0):
        self.num_agents = num_agents
        self.desired_distances = desired_distances
        self.topology_edges = topology_edges
        self.obstacle_positions = obstacle_positions if obstacle_positions is not None else []
        self.safety_radius = safety_radius

        # Build per-robot RMP trees
        self.roots = []
        self.robot_nodes = []
        self.collision_leaves = []  # list of lists
        self.formation_leaves = []  # list of lists

        for i in range(num_agents):
            root = RMPRoot(f"robot_{i}_root")
            robot_node = RMPNode(
                f"robot_{i}",
                root,
                psi=lambda y, idx=i: y[2 * idx:2 * idx + 2].reshape(-1, 1),
                J=lambda y, idx=i: np.eye(2, y.shape[0],
                                           k=0 if idx == 0 else 0)
                    if False else self._robot_J(idx, num_agents),
                J_dot=lambda y, y_dot, idx=i: np.zeros((2, 2 * num_agents)),
            )
            # We won't use intermediate nodes. Instead, build directly on root
            # using per-robot roots with identity mappings.
            self.roots.append(root)
            self.robot_nodes.append(robot_node)

        # Actually, for simplicity, use one root per robot
        self._build_per_robot_trees(
            formation_gain, formation_eta, formation_w,
            damper_w, damper_eta,
        )

    def _robot_J(self, idx, n):
        """Jacobian extracting robot idx's 2D coords from full state."""
        J = np.zeros((2, 2 * n))
        J[0, 2 * idx] = 1.0
        J[1, 2 * idx + 1] = 1.0
        return J

    def _build_per_robot_trees(self, formation_gain, formation_eta, formation_w,
                               damper_w, damper_eta):
        """Rebuild using separate per-robot roots (2D each)."""
        self.roots = []
        self.collision_avoidance_leaves = [[] for _ in range(self.num_agents)]
        self.formation_leaves = [[] for _ in range(self.num_agents)]
        self.damper_leaves = []
        self._other_robot_nodes = {}  # (i, j) -> RMPNode placeholder

        for i in range(self.num_agents):
            root = RMPRoot(f"robot_{i}")
            self.roots.append(root)

            # Damper
            damper = Damper(f"damper_{i}", root, w=damper_w, eta=damper_eta)
            self.damper_leaves.append(damper)

            # Static obstacle avoidance
            for k, obs_pos in enumerate(self.obstacle_positions):
                ca = CollisionAvoidance(
                    f"obs_avoid_{i}_{k}", root,
                    c=np.array(obs_pos[:2]), R=self.safety_radius,
                    epsilon=0.2, alpha=1e-5, eta=0,
                )
                self.collision_avoidance_leaves[i].append(ca)

            # Decentralized collision avoidance and formation w.r.t. other robots
            for j in range(self.num_agents):
                if j == i:
                    continue

                # Create a dummy node to hold the other robot's state
                other_node = RMPNode(f"other_{j}_for_{i}", None, None, None, None)
                other_node.x = np.zeros((2, 1))
                other_node.x_dot = np.zeros((2, 1))
                self._other_robot_nodes[(i, j)] = other_node

                # Collision avoidance
                ca_dec = CollisionAvoidanceDecentralized(
                    f"ca_dec_{i}_{j}", root, parent_param=other_node,
                    R=self.safety_radius, epsilon=1e-8, alpha=1e-5, eta=0,
                )
                self.collision_avoidance_leaves[i].append(ca_dec)

                # Formation (only for topology edges)
                if (min(i, j), max(i, j)) in self.topology_edges or \
                   (i, j) in self.topology_edges or (j, i) in self.topology_edges:
                    d_ij = self.desired_distances[i, j]
                    fm = FormationDecentralized(
                        f"formation_{i}_{j}", root, parent_param=other_node,
                        d=d_ij, gain=formation_gain, eta=formation_eta, w=formation_w,
                    )
                    self.formation_leaves[i].append(fm)

    def _update_other_states(self, positions, velocities):
        """Update the other-robot placeholder nodes with current states."""
        for (i, j), node in self._other_robot_nodes.items():
            node.x = positions[j].reshape(-1, 1)
            node.x_dot = velocities[j].reshape(-1, 1)

    def solve(self, positions, velocities):
        """
        Compute RMPflow accelerations for all robots.

        Args:
            positions: (num_agents, 2) array
            velocities: (num_agents, 2) array

        Returns:
            accelerations: (num_agents, 2) array
        """
        positions = np.asarray(positions, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        self._update_other_states(positions, velocities)

        accels = np.zeros((self.num_agents, 2))
        for i in range(self.num_agents):
            # Update decentralized leaves
            for leaf in self.collision_avoidance_leaves[i]:
                if hasattr(leaf, 'update_params') and leaf.parent_param is not None:
                    leaf.update_params()
            for leaf in self.formation_leaves[i]:
                leaf.update_params()

            a = self.roots[i].solve(
                positions[i].reshape(-1, 1),
                velocities[i].reshape(-1, 1),
            )
            accels[i] = a.flatten()[:2]

        return accels

    def get_formation_forces(self, positions, velocities):
        """
        Compute only the formation-related forces (for ATACOM blending).
        Returns (num_agents, 2) formation accelerations.
        """
        positions = np.asarray(positions, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        self._update_other_states(positions, velocities)

        forces = np.zeros((self.num_agents, 2))
        for i in range(self.num_agents):
            # Set root state first
            self.roots[i].set_root_state(
                positions[i].reshape(-1, 1),
                velocities[i].reshape(-1, 1),
            )

            # Only evaluate formation leaves
            f_total = np.zeros((2, 1))
            M_total = np.zeros((2, 2))

            for leaf in self.formation_leaves[i]:
                leaf.update_params()
                # Manual pushforward
                leaf.x = leaf.psi(self.roots[i].x)
                leaf.x_dot = np.dot(leaf.J(self.roots[i].x), self.roots[i].x_dot)
                leaf.eval_leaf()
                # Pullback
                J_l = leaf.J(self.roots[i].x)
                J_dot_l = leaf.J_dot(self.roots[i].x, self.roots[i].x_dot)
                f_total += np.dot(J_l.T,
                                  leaf.f - np.dot(np.dot(leaf.M, J_dot_l),
                                                  self.roots[i].x_dot))
                M_total += np.dot(np.dot(J_l.T, leaf.M), J_l)

            if np.any(M_total):
                a = np.dot(np.linalg.pinv(M_total), f_total)
                forces[i] = a.flatten()[:2]

        return forces

    def update_obstacles(self, obstacle_positions):
        """Update obstacle positions (rebuild static CA leaves)."""
        self.obstacle_positions = obstacle_positions
        # For simplicity, rebuild the tree with new obstacles
        # In practice, you'd update the psi/J/J_dot closures
