"""
StateConstraint and ConstraintsSet with slack variable management.
Ported from safeRL_manifold/safepo/common/constrained_manifold/constraints.py,
simplified for 2D (no dim_x).
"""

import numpy as np


class StateConstraint:
    """
    Single or multiple inequality constraints with slack variables.

    For inequality c(q) <= 0, slack variables convert to equality:
        c(q) + penalty(s) = 0

    Slack types:
        - softcorner: penalty(s) = -log(-expm1(beta*s)) / beta  (default, sharp)
        - softplus:   penalty(s) = log(1 + exp(beta*s)) / beta
        - square:     penalty(s) = beta * s^2
        - None:       equality constraint (no slack)
    """

    def __init__(self, dim_q, dim_out, fun, jac_q,
                 slack_type='softcorner', slack_beta=30.0, threshold=1e-3):
        self.dim_q = dim_q
        self.dim_out = dim_out
        self.fun_origin = fun
        self.jac_q_fun = jac_q

        self.slack_type = slack_type

        if np.isscalar(slack_beta):
            slack_beta = np.ones(self.dim_out) * slack_beta
        self.beta = slack_beta
        assert self.beta.shape[0] == dim_out

        if np.isscalar(threshold):
            threshold = np.ones(self.dim_out) * threshold
        assert threshold.shape[0] == dim_out
        self.threshold = threshold

        self.dim_slack = self.dim_out if slack_type is not None else 0
        self.s = np.zeros(self.dim_slack) if self.dim_slack > 0 else np.array([])

        # Cache
        self.c_last = np.nan
        self.q_last = np.nan

    def fun(self, q, origin_constr=False):
        """Evaluate constraint (with or without slack relaxation)."""
        if not (np.array_equal(self.q_last, q)):
            self.c_last = self.fun_origin(q)
            self.q_last = q.copy()

        if origin_constr or self.slack_type is None:
            return np.atleast_1d(self.c_last)

        if self.slack_type == 'square':
            ret = self.c_last + self.beta * self.s ** 2
        elif self.slack_type == 'softplus':
            ret = self.c_last + np.log1p(np.exp(self.beta * self.s)) / self.beta
        elif self.slack_type == 'softcorner':
            ret = self.c_last - np.log(-np.expm1(self.beta * self.s)) / self.beta
        else:
            raise NotImplementedError(f"Unknown slack type: {self.slack_type}")

        return np.atleast_1d(ret)

    def jac_q(self, q):
        """Jacobian of constraint w.r.t. q. Returns (dim_out, dim_q)."""
        if not (np.array_equal(self.q_last, q)):
            self.c_last = self.fun_origin(q)
            self.q_last = q.copy()
        return np.atleast_2d(self.jac_q_fun(q))

    def jac_slack(self):
        """Diagonal of slack Jacobian. Returns (dim_out,) or scalar 0."""
        if self.slack_type is None:
            return np.atleast_1d(0.0)

        if self.slack_type == 'square':
            return 2 * self.beta * self.s
        elif self.slack_type == 'softplus':
            exp_s = np.exp(self.beta * self.s)
            return exp_s / (1 + exp_s)
        elif self.slack_type == 'softcorner':
            exp_s = np.exp(self.beta * self.s)
            return exp_s / -np.expm1(self.beta * self.s)
        else:
            raise NotImplementedError

    def reset_slack(self, q):
        """Initialize slack so that c(q) + penalty(s) ≈ 0."""
        c_init = np.maximum(-self.fun(q, origin_constr=True), self.threshold)

        if self.slack_type is None:
            self.s = np.array([])
        elif self.slack_type == 'square':
            self.s = np.sqrt(c_init / self.beta)
        elif self.slack_type == 'softplus':
            self.s = np.log(np.expm1(self.beta * c_init)) / self.beta
        elif self.slack_type == 'softcorner':
            self.s = np.log1p(-np.exp(-self.beta * c_init)) / self.beta
        else:
            raise NotImplementedError


class ConstraintsSet:
    """Aggregates multiple StateConstraints. Computes joint Jacobians."""

    def __init__(self, dim_q):
        self.dim_q = dim_q
        self.constraints_list = []
        self.dim_out = 0
        self.dim_slack = 0
        self.dim_null = 0

    def add_constraint(self, c: StateConstraint):
        assert self.dim_q == c.dim_q
        self.dim_out += c.dim_out
        self.dim_slack += c.dim_slack
        self.dim_null = self.dim_q + self.dim_slack - self.dim_out
        self.constraints_list.append(c)

    def c(self, q, origin_constr=False):
        """Evaluate all constraints, concatenated."""
        ret = [c_i.fun(q, origin_constr) for c_i in self.constraints_list]
        return np.concatenate(ret)

    def get_jacobians(self, q):
        """
        Returns:
            J_q: (dim_out, dim_q)
            J_s: (dim_out, dim_slack) block-diagonal
        """
        ret_jac_q = []
        ret_slack = np.zeros((self.dim_out, self.dim_slack))
        row = 0
        col = 0
        for c_i in self.constraints_list:
            ret_jac_q.append(c_i.jac_q(q))
            ret_slack[row:row + c_i.dim_out, col:col + c_i.dim_slack] = np.diag(c_i.jac_slack())
            row += c_i.dim_out
            col += c_i.dim_slack
        return np.vstack(ret_jac_q), ret_slack

    @property
    def s(self):
        """Concatenated slack variables."""
        return np.concatenate([c_i.s for c_i in self.constraints_list])

    def reset_slack(self, q):
        for c_i in self.constraints_list:
            c_i.reset_slack(q)
