# Leaf node RMP classes
# @author Anqi Li
# @date April 8, 2019

from rmp import RMPNode, RMPRoot, RMPLeaf
import numpy as np
from numpy.linalg import norm


def _pairwise_distance_floor(delta, eps=1e-8):
    """||delta|| lower-bounded by eps so Jacobians 1/r stay finite when agents coincide."""
    return max(float(norm(delta)), float(eps))


class CollisionAvoidance(RMPLeaf):
    """
    Obstacle avoidance RMP leaf
    """

    def __init__(self, name, parent, parent_param, c=np.zeros(2), R=1, epsilon=0.2,
        alpha=1e-5, eta=0):

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

            psi = lambda y: np.array(norm(y - c) / R - 1).reshape(-1,1)
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
            M = np.minimum(np.maximum(M, - 1e5), 1e5)

            Bx_dot = eta * g * x_dot

            f = - grad_Phi - xi - Bx_dot
            f = np.minimum(np.maximum(f, - 1e10), 1e10)

            return (f, M)

        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)



class CollisionAvoidanceDecentralized(RMPLeaf):
    """
    Decentralized collision avoidance RMP leaf for the RMPForest
    """

    def __init__(self, name, parent, parent_param, c=np.zeros(2), R=1, epsilon=1e-8,
        alpha=1e-5, eta=0):

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
            M = np.minimum(np.maximum(M, - 1e5), 1e5)

            Bx_dot = eta * g * x_dot

            f = - grad_Phi - xi - Bx_dot
            f = np.minimum(np.maximum(f, - 1e10), 1e10)

            return (f, M)


        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)


    def pushforward(self):
        """
        override pushforward() to update the curvature term
        """
        if self.verbose:
            print('%s: pushforward' % self.name)

        if self.psi is not None and self.J is not None:
            self.x = self.psi(self.parent.x)
            self.x_dot = np.dot(self.J(self.parent.x), self.parent.x_dot)
            self.x_dot_real = np.dot(
                self.J(self.parent.x),
                self.parent.x_dot - self.parent_param.x_dot)


    def eval_leaf(self):
        """
        override eval_leaf() to update the curvature term
        """
        self.f, self.M = self.RMP_func(self.x, self.x_dot, self.x_dot_real)

    def update_params(self):
        """
        update the position of the other robot
        """
        c = self.parent_param.x
        z_dot = self.parent_param.x_dot
        R = self.R
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        N = c.size
        r_floor = max(float(self.epsilon), 1e-8)

        def _r(y):
            return _pairwise_distance_floor(y - c, r_floor)

        self.psi = lambda y: np.array(_r(y) / R - 1).reshape(-1, 1)
        self.J = lambda y: (1.0 / _r(y)) * (y - c).T / R
        self.J_dot = lambda y, y_dot: np.dot(
            y_dot.T,
            (-1.0 / _r(y) ** 3 * np.dot((y - c), (y - c).T) + 1.0 / _r(y) * np.eye(N)),
        ) / R


class CollisionAvoidanceCentralized(RMPLeaf):
    """
    Centralized collision avoidance RMP leaf for a pair of robots
    """

    def __init__(self, name, parent, R=1, epsilon=1e-8,
        alpha=1e-5, eta=0):

        self.R = R

        def psi(y):
            N = int(y.size / 2)
            y1 = y[: N]
            y2 = y[N:]
            return np.array(norm(y1 - y2) / R - 1).reshape(-1,1)
        def J(y):
            N = int(y.size / 2)
            y1 = y[: N]
            y2 = y[N:]
            return np.concatenate((
                    1.0 / norm(y1 - y2) * (y1 - y2).T / R,
                    -1.0 / norm(y1 - y2) * (y1 - y2).T / R),
                axis=1)
        def J_dot(y, y_dot):
            N = int(y.size / 2)
            y1 = y[: N]
            y2 = y[N:]
            y1_dot = y_dot[: N]
            y2_dot = y_dot[N:]
            return np.concatenate((
                    np.dot(
                        y1_dot.T,
                        (-1 / norm(y1 - y2) ** 3 * np.dot((y1 - y2), (y1 - y2).T)
                            + 1 / norm(y1 - y2) * np.eye(N))) / R,
                    np.dot(
                        y2_dot.T,
                        (-1 / norm(y1 - y2) ** 3 * np.dot((y1 - y2), (y1 - y2).T)
                            + 1 / norm(y1 - y2) * np.eye(N))) / R),
                axis=1)


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
            M = np.minimum(np.maximum(M, - 1e5), 1e5)

            Bx_dot = eta * g * x_dot

            f = - grad_Phi - xi - Bx_dot
            f = np.minimum(np.maximum(f, - 1e10), 1e10)

            return (f, M)


        RMPLeaf.__init__(self, name, parent, None, psi, J, J_dot, RMP_func)



class GoalAttractorUni(RMPLeaf):
    """
    Goal Attractor RMP leaf
    """

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

            beta = np.exp(- x_norm ** 2 / 2 / (sigma ** 2))
            w = (w_u - w_l) * beta + w_l
            s = (1 - np.exp(-2 * alpha * x_norm)) / (1 + np.exp(
                    -2 * alpha * x_norm))

            G = np.eye(N) * w
            if x_norm > tol:
                grad_Phi = s / x_norm * w * x * gain
            else:
                grad_Phi = 0
            Bx_dot = eta * w * x_dot
            grad_w = - beta * (w_u - w_l) / sigma ** 2 * x

            x_dot_norm = norm(x_dot)
            xi = -0.5 * (x_dot_norm ** 2 * grad_w - 2 *
                np.dot(np.dot(x_dot, x_dot.T), grad_w))

            M = G
            f = - grad_Phi - Bx_dot - xi

            return (f, M)

        RMPLeaf.__init__(self, name, parent, None, psi, J, J_dot, RMP_func)


    def update_goal(self, y_g):
        """
        update the position of the goal
        """
        
        if y_g.ndim == 1:
            y_g = y_g.reshape(-1, 1)
        N = y_g.size
        self.psi = lambda y: (y - y_g)
        self.J = lambda y: np.eye(N)
        self.J_dot = lambda y, y_dot: np.zeros((N, N))



class FormationDecentralized(RMPLeaf):
    """
    Decentralized formation control RMP leaf for the RMPForest
    """

    def __init__(self, name, parent, parent_param, c=np.zeros(2), d=1, gain=1, eta=2, w=1):

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
            f = - grad_Phi - Bx_dot

            return (f, M)

        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)


    def update_params(self):
        """
        update the position of the other robot
        """

        z = self.parent_param.x
        z_dot = self.parent_param.x_dot

        c = z
        d = self.d
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        N = c.size
        r_floor = 1e-8

        def _r(y):
            return _pairwise_distance_floor(y - c, r_floor)

        self.psi = lambda y: np.array(_r(y) - d).reshape(-1, 1)
        self.J = lambda y: (1.0 / _r(y)) * (y - c).T
        self.J_dot = lambda y, y_dot: np.dot(
            y_dot.T,
            (
                -1.0 / _r(y) ** 3 * np.dot((y - c), (y - c).T)
                + 1.0 / _r(y) * np.eye(N)
            ),
        )


class FormationOrientationDecentralized(RMPLeaf):
    """
    Decentralized formation orientation RMP: keep edge direction i→j aligned with
    a desired unit vector r_hat^*. Task z = 1 - hat(r_ij)^T r_hat^*, with
    r_ij = x_j - x_i (unit vector from robot i toward j).
    """

    def __init__(
        self,
        name,
        parent,
        parent_param,
        r_star,
        alpha_theta=1.0,
        eta_theta=2.0,
        c_theta=10.0,
        eps=1e-8,
    ):

        assert parent_param is not None
        rs = np.asarray(r_star, dtype=np.float64).reshape(-1, 1)
        nr0 = norm(rs)
        assert nr0 >= eps, "formation_desired_direction must be non-zero"
        self.r_star = rs / nr0
        self.alpha_theta = alpha_theta
        self.eta_theta = eta_theta
        self.c_theta = c_theta
        self.eps = eps

        psi = None
        J = None
        J_dot = None

        a = alpha_theta
        eta = eta_theta
        c_met = c_theta

        def RMP_func(x, x_dot):
            grad_Phi = a * x
            Bx_dot = eta * x_dot
            M = np.array([[float(c_met)]], dtype=np.float64)
            f = -grad_Phi - Bx_dot
            return (f, M)

        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)

    def update_params(self):
        c = self.parent_param.x
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        r_star = self.r_star
        eps = self.eps

        def psi(y):
            y = np.asarray(y, dtype=np.float64)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            r_vec = c - y
            nr = float(norm(r_vec))
            if nr < eps:
                nr = eps
            hat = r_vec / nr
            z = 1.0 - float(hat.T @ r_star)
            return np.array([[z]], dtype=np.float64)

        def J_task(y):
            y = np.asarray(y, dtype=np.float64)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            r_vec = c - y
            nr = float(norm(r_vec))
            if nr < eps:
                nr = eps
            hat = r_vec / nr
            p_perp = np.eye(2) - hat @ hat.T
            # z = 1 - hat^T r_star  ⇒  ∂z/∂x_i = r_star^T (I - hat hat^T) / ||r||
            # (chain rule with r_ij = x_j - x_i)
            ji = (r_star.T @ p_perp) / nr
            return ji.reshape(1, 2)

        self.psi = psi
        self.J = J_task
        self.J_dot = lambda y, y_dot: np.zeros((1, 2), dtype=np.float64)


class FormationCentralized(RMPLeaf):
    """
    Centralized formation control RMP leaf for a pair of robots
    """

    def __init__(self, name, parent, d=1, gain=1, eta=2, w=1):

        def psi(y):
            N = int(y.size / 2)
            y1 = y[: N]
            y2 = y[N:]
            return np.array(norm(y1 - y2) - d).reshape(-1,1)
        def J(y):
            N = int(y.size / 2)
            y1 = y[: N]
            y2 = y[N:]
            return np.concatenate((
                    1.0 / norm(y1 - y2) * (y1 - y2).T,
                    -1.0 / norm(y1 - y2) * (y1 - y2).T),
                axis=1)
        def J_dot(y, y_dot):
            N = int(y.size / 2)
            y1 = y[: N]
            y2 = y[N:]
            y1_dot = y_dot[: N]
            y2_dot = y_dot[N:]
            return np.concatenate((
                    np.dot(
                        y1_dot.T,
                        (-1 / norm(y1 - y2) ** 3 * np.dot((y1 - y2), (y1 - y2).T)
                            + 1 / norm(y1 - y2) * np.eye(N))),
                    np.dot(
                        y2_dot.T,
                        (-1 / norm(y1 - y2) ** 3 * np.dot((y1 - y2), (y1 - y2).T)
                            + 1 / norm(y1 - y2) * np.eye(N)))),
                axis=1)


        def RMP_func(x, x_dot):
            G = w
            grad_Phi = gain * x * w
            Bx_dot = eta * w * x_dot
            M = G
            f = - grad_Phi - Bx_dot

            return (f, M)

        RMPLeaf.__init__(self, name, parent, None, psi, J, J_dot, RMP_func)



class Damper(RMPLeaf):
    """
    Damper RMP leaf
    """

    def __init__(self, name, parent, w=1, eta=1):

        psi = lambda y: y
        J = lambda y: np.eye(2)
        J_dot = lambda y, y_dot: np.zeros((2, 2))


        def RMP_func(x, x_dot):
            G = w
            Bx_dot = eta * w * x_dot
            M = G
            f = - Bx_dot

            return (f, M)

        RMPLeaf.__init__(self, name, parent, None, psi, J, J_dot, RMP_func)


# [ADD 2026-04-18] RL-as-leaf node for chap4 4.3.1 (真正实现 RL 叶节点参与 pullback)
class RLLeaf(RMPLeaf):
    """
    Reinforcement Learning policy leaf for RMPflow.

    Injects an externally-provided RL action (expected acceleration in the
    parent's task space) into the RMP computation graph as a proper leaf node,
    so the RL intent participates in the standard pushforward / pullback /
    resolve flow alongside geometric leaves (collision, formation, ...).

    Task mapping (identity):
        psi(y) = y,  J(y) = I,  J_dot = 0

    Natural-form RMP:
        M_RL = w * I
        f_RL = M_RL @ a_RL        (where a_RL is the current RL action)

    After `set_action(a_rl)`, the leaf's (f, M) reflects the new target at the
    next pushforward/pullback cycle. Combined with geometric leaves at the
    root, the resolved acceleration becomes

        a = (w*I + sum M_geo)^{-1} (w*a_RL + sum f_geo),

    recovering pure RL when geometric forces vanish and geometric guidance
    when they dominate.

    Args:
        name:   Unique node name for debugging/logging.
        parent: RMPNode this leaf attaches to (typically a robot node whose
                x is the 2D world position of that robot).
        w:      Scalar metric weight w_RL on identity metric. Larger = more
                faithful to RL; smaller = geometric leaves dominate.
        dim:    Task-space dimensionality (=2 for planar robots).
    """

    def __init__(self, name, parent, w=1.0, dim=2):
        self._w = float(w)
        self._dim = int(dim)
        self._a_rl = np.zeros((self._dim, 1), dtype=np.float64)

        psi = lambda y: y
        J = lambda y, d=self._dim: np.eye(d, dtype=np.float64)
        J_dot = lambda y, y_dot, d=self._dim: np.zeros((d, d), dtype=np.float64)

        # Closure over self so RMP_func re-reads self._a_rl each call.
        def RMP_func(x, x_dot):
            M = self._w * np.eye(self._dim, dtype=np.float64)
            f = M @ self._a_rl
            return (f, M)

        RMPLeaf.__init__(self, name, parent, None, psi, J, J_dot, RMP_func)

    def set_action(self, a_rl):
        """Set the RL action (expected acceleration) for the next cycle.

        Args:
            a_rl: array-like of shape (dim,) or (dim, 1).
        """
        a = np.asarray(a_rl, dtype=np.float64).reshape(-1)
        assert a.shape[0] == self._dim, (
            f"RLLeaf '{self.name}' expected dim={self._dim}, got {a.shape[0]}"
        )
        self._a_rl = a.reshape(self._dim, 1)

    def set_weight(self, w):
        """Optionally tune w_RL online (e.g., curriculum scheduling)."""
        self._w = float(w)

    def update_params(self):
        """RL leaf parameters (a_rl) are set externally via set_action; nothing here."""
        pass