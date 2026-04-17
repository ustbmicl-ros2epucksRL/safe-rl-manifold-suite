import numpy as np


class StateConstraint:
    def __init__(self, dim_q, dim_out, fun, jac_q, dim_x=0, jac_x=None, slack_type='softcorner', slack_beta=30., threshold=1e-3):
        """
        Define single/multiple equality/inequality constraints
        :param dim_q: [int] Dimension of the directly controllable state
        :param dim_out: [int] Dimension of the constraints
        :param fun: [object] The callback function of the constraint, output 1d array: c(q, x)=0
        :param jac_q: [object] The Jacobian function w.r.t q, i.e., J_q(q, x)
        :param dim_x: [int] Dimension of the directly uncontrollable state
        :param jac_x: [object] The Jacobian function w.r.t x, i.e., J_x(q, x). if J_x does not exist, remain None
        :param slack_type [string, None]: The type of slack variable. For equality constraint, slack_type = None
        :param slack_beta [float, ndarray]: The hyperparameter of the slack variable
        :param threshold [float, ndarray]: The threshold of the constraint, c(q, x) + epsilon < 0, epsilon >= 0
        """
        self.dim_q = dim_q
        self.dim_x = dim_x
        self.dim_out = dim_out
        self.fun_origin = fun
        self.jac_q_fun = jac_q
        self.jac_x_fun = jac_x

        self.slack_type = slack_type

        # if np.isscalar(slack_beta):
        #     slack_beta = np.ones(self.dim_out) * slack_beta
        self.beta = slack_beta
        # assert self.beta.shape[0] == dim_out, "The dimension of threshold is not matching"

        # if np.isscalar(threshold):
        #     threshold = np.ones(self.dim_out) * threshold
        # assert threshold.shape[0] == dim_out, "The dimension of threshold is not matching"

        self.threshold = threshold

        self.dim_slack = self.dim_out
        if slack_type is None:
            self.s = np.array([])
            self.dim_slack = 0

        self.c_last = np.NaN
        self.q_last = np.NaN
        self.x_last = np.NaN

    def fun(self, q, x=None, origin_constr=False):
        if not (np.equal(self.q_last, q).all() and np.equal(self.x_last, x).all()):
            self.c_last = self.fun_origin(q, x)
            self.q_last = q
            self.x_last = x

        if origin_constr:
            ret = self.c_last
        else:
            slack_beta = np.ones(self.dim_out) * self.beta
            if self.slack_type is None:
                ret = self.c_last
            elif self.slack_type == 'square':
                ret = self.c_last + slack_beta * self.s ** 2
            elif self.slack_type == 'exp':
                ret = self.c_last + np.exp(slack_beta * self.s)
            elif self.slack_type == 'softplus':
                ''' self.s is exp(beta * s) '''
                ret = self.c_last + np.log1p(np.exp(slack_beta * self.s)) / slack_beta
            elif self.slack_type == 'softcorner':
                ''' self.s is exp(beta * s) '''
                ret = self.c_last - np.log(-np.expm1(slack_beta * self.s)) / slack_beta
            else:
                raise NotImplementedError
        return np.atleast_1d(ret)

    def jac_q(self, q, x=None):
        if not (np.equal(self.q_last, q).all() and np.equal(self.x_last, x).all()):
            self.c_last = self.fun_origin(q, x)
            self.q_last = q
        return np.atleast_2d(self.jac_q_fun(q, x))

    def jac_x(self, q, x=None):
        if not (np.equal(self.q_last, q).all() and np.equal(self.x_last, x).all()):
            self.c_last = self.fun_origin(q, x)
            self.q_last = q

        if self.dim_x == 0:
            return np.zeros((self.dim_out, 1))
        elif self.jac_x_fun is None:
            return np.zeros((self.dim_out, self.dim_x))
        else:
            return np.atleast_2d(self.jac_x_fun(q, x))

    def jac_slack(self):
        slack_beta = np.ones(self.dim_out) * self.beta
        if self.slack_type is None:
            return np.atleast_2d(0.)
        elif self.slack_type == 'square':
            return 2 * slack_beta * self.s
        elif self.slack_type == 'exp':
            return slack_beta * np.exp(slack_beta * self.s)
        elif self.slack_type == 'softplus':
            exp_s = np.exp(slack_beta * self.s)
            return exp_s / (1 + exp_s)
        elif self.slack_type == 'softcorner':
            exp_s = np.exp(slack_beta * self.s)
            return exp_s / -np.expm1(slack_beta * self.s)
        else:
            raise NotImplementedError

    def reset_slack(self, q, x=None):
        if self.dim_out > 0:
            threshold = np.ones(self.dim_out) * self.threshold
        else:
            threshold = self.threshold
        fun = -self.fun(q, x, origin_constr=True)
        if fun.size == 0:
            fun = np.empty(self.dim_out,)
        c_init = np.maximum(fun, threshold)
        if self.dim_out > 0:
            slack_beta = np.ones(self.dim_out) * self.beta
        else:
            slack_beta = self.beta
        if self.slack_type is None:
            self.s = np.array([])
        elif self.slack_type == 'square':
            self.s = np.sqrt(c_init / slack_beta)
        elif self.slack_type == 'exp':
            self.s = np.log(c_init) / slack_beta
        elif self.slack_type == 'softplus':
            ''' self.s is exp(beta * s) '''
            self.s = np.log(np.expm1(slack_beta * c_init)) / slack_beta
        elif self.slack_type == 'softcorner':
            ''' self.s is exp(beta * s) '''
            self.s = np.log1p(-np.exp(-slack_beta * c_init)) / slack_beta
        else:
            raise NotImplementedError


class ConstraintsSet:
    """
    The class to gather multiple constraints
    """
    def __init__(self, dim_q, dim_x):
        self.dim_q = dim_q
        self.dim_x = dim_x
        self.constraints_list = list()
        self.dim_out = 0
        self.dim_slack = 0
        self.dim_null = 0

    def add_constraint(self, c: StateConstraint):
        assert self.dim_q == c.dim_q, "The dimension of q is not equal the constraint."
        assert self.dim_x == c.dim_x, "The dimension of x is not equal the constraint."
        self.dim_out += c.dim_out
        self.dim_slack += c.dim_slack
        self.dim_null = self.dim_q + self.dim_slack - self.dim_out
        self.constraints_list.append(c)

    def c(self, q, x=None, origin_constr=False):
        ret = list()
        for c_i in self.constraints_list:
            ret.append(c_i.fun(q, x, origin_constr))
        return np.concatenate(ret)

    def get_jacobians(self, q, x=None):
        ret_jac_q = list()
        ret_jac_x = list()
        ret_slack = np.zeros((self.dim_out, self.dim_slack))
        row = 0
        col = 0
        for c_i in self.constraints_list:
            ret_jac_q.append(c_i.jac_q(q, x))
            ret_jac_x.append(c_i.jac_x(q, x))

            ret_slack[row:row+c_i.dim_out, col:col+c_i.dim_slack] = np.diag(c_i.jac_slack())
            row += c_i.dim_out
            col += c_i.dim_slack
        return np.vstack(ret_jac_q), np.vstack(ret_jac_x), ret_slack

    @property
    def s(self):
        ret = list()
        for c_i in self.constraints_list:
            ret.append(c_i.s)
        return np.concatenate(ret)

    def reset_slack(self, q, x=None):
        for c_i in self.constraints_list:
            c_i.reset_slack(q, x=x)
