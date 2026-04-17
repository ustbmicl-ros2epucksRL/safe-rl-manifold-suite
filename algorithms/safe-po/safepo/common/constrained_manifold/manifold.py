import numpy as np
import gymnasium
from safepo.common.constrained_manifold.constraints import ConstraintsSet


class AtacomEnvWrapper:
    """
    Environment wrapper of ATACOM
    """

    def __init__(self, base_env, dq_max, constraints, step_size=0.01, update_with_agent_freq=True):
        """
        Constructor
        Args:
            base_env (mushroom_rl.Core.Environment): The base environment inherited from
            dq_max (array, float): the maximum velocity of the directly controllable variable
            constraints (ConstraintsSet): the constraints set c(q, mu) = 0
            step_size (float): the step size of the environment
            update_with_agent_freq (bool):
        """
        # self.env = base_env
        self.constraints = constraints
        self.step_size = step_size
        self._logger = None

        self.K_c = 100 # / self.step_size

        # self.q = np.zeros(self.constraints.dim_q)
        self.x = np.zeros(self.constraints.dim_x)
        self.dx = np.zeros(self.constraints.dim_x)

        # self._mdp_info = self.env.info.copy()
        self.action_space = gymnasium.spaces.Box(low=-np.ones(self.constraints.dim_null),
                                          high=np.ones(self.constraints.dim_null))

        if np.isscalar(dq_max):
            self.dq_max = np.ones(self.constraints.dim_q) * dq_max
        else:
            self.dq_max = dq_max
            assert np.shape(self.dq_max)[0] == self.constraints.dim_q

        # self.state = self.env.reset()

        self.constr_logs = list()

        if not update_with_agent_freq:
        #     # Update at agent's frequency
        #     self.env._preprocess_action = self._preprocess_action
        # else:
            # Update at simulation's frequency
            self.env.step_action_function = self.step_action_function

    def _get_q(self, state):
        raise NotImplementedError

    def _get_x(self):
        """
        The function to get the indirectly controllable variable. If it doesn't exist, return None
        """
        return None

    def _get_dx(self):
        """
        The function to get the indirectly controllable variable. If it doesn't exist, return 0.
        """
        return np.atleast_1d(0.)

    def compute_ctrl_action(self, dq):
        """
        Convert the dq to designed control action a
        """
        raise NotImplementedError

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self, state=None):
        self.q = self._get_q()
        self.x = self._get_x()
        self.constraints.reset_slack(self.q[0], self.x)
        self.constraints.reset_slack(self.q[1], self.x)
        self.h_pos = np.empty((0, 3))
        self.h_num = 0

    def render(self):
        self.env.render()

    def stop(self):
        self.env.stop()

    def step(self, action):
        alpha = np.clip(action, self.info.action_space.low, self.info.action_space.high)
        action = self._preprocess_action(alpha)
        self.state, reward, absorb, info = self.env.step(action)
        self.q = self._get_q(self.state)
        self.x = self._get_x(self.state)
        self.dx = self._get_dx(self.state)

        if not hasattr(self.env, "get_constraints_logs"):
            self._update_constraint_stats(self.q, self.x)
        return self.state.copy(), reward, absorb, info

    def action_truncation_scale(self, dq):
        scale = np.maximum((np.abs(dq) / self.env.action_space.high).max(), 1)
        return 1 / scale

    def _preprocess_action(self, alpha):
        return self.step_action_function(self.env._state, alpha)

    def step_action_function(self, sim_state, alpha):
        self.state = self.env._create_observation(sim_state)
        self.q = self._get_q(self.state)
        self.x = self._get_x(self.state)
        self.dx = self._get_dx(self.state)
        self.constraints.reset_slack(self.q, self.x)

        Jc, J_x = self._construct_Jc(self.q, self.x)
        Jc_inv = Jc.T @ np.linalg.inv(Jc @ Jc.T)

        bases = np.diag(self.dq_max[:self.constraints.dim_null])
        # Projected Null Space
        Nc = (np.eye(Jc.shape[1]) - Jc_inv @ Jc)[:, :self.constraints.dim_null] @ bases

        dq_null = Nc @ alpha
        dq_uncontrol = -Jc_inv @ J_x @ self.dx
        dq_err = -self.K_c * (Jc_inv @ self.constraints.c(self.q, self.x, origin_constr=False))
        dq = dq_null + dq_err + dq_uncontrol

        scale = self.action_truncation_scale(dq[:self.constraints.dim_q])
        dq = scale * dq
        ctrl_action = self.compute_ctrl_action(dq[:self.constraints.dim_q])
        return ctrl_action

    @property
    def info(self):
        return self._mdp_info

    def _construct_Jc(self, q, x=None):
        J_q, J_x, J_s = self.constraints.get_jacobians(q, x)
        return np.hstack([J_q, J_s]), J_x

    def set_logger(self, logger):
        self._logger = logger

    def _update_constraint_stats(self, q, x=None):
        c_i = self.constraints.c(q, x, origin_constr=True)
        self.constr_logs.append(c_i)
        if len(self.constr_logs) > 10000000:
            raise BufferError("The constraint buffer size is more than 1000000, please clear it")

    def get_constraints_logs(self):
        if not hasattr(self.env, "get_constraints_logs"):
            constr_logs = np.array(self.constr_logs)
            c_avg = np.mean(constr_logs)
            c_max = np.max(constr_logs)
            self.constr_logs.clear()
            return c_avg, c_max
        else:
            return self.env.get_constraints_logs()
