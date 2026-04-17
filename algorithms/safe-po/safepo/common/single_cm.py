import numpy as np
from safepo.common.constrained_manifold.manifold import AtacomEnvWrapper
from safepo.common.constrained_manifold.constraints import StateConstraint, ConstraintsSet

class SingleNavAtacom(AtacomEnvWrapper):
    def __init__(
        self,
        base_env,
        gamma=0.99,
        horizon=500,
        step_action_function=None,
        timestep=1 / 30.0,
        n_intermediate_steps=1,
        debug_gui=False,
        init_state=None,
        terminate_on_collision=True,
        save_key_frame=False,
        slack_type="softcorner",
        update_with_agent_freq=True,
        is_opponent_moving=True,
        slack_beta_hazard_constraint=1,
        slack_beta_sigwall_constraint=0.5,
        slack_threshold_hazard_constraint=0.01,
        slack_threshold_sigwall_constraint=0.1,
        noise_std: float = 0.0,
    ):

        dim_q = 2  # base_env.action_space.shape[0]
        dim_x = 0
        self.base_env = base_env
        self.env = base_env  # 父类 AtacomEnvWrapper 未设置，_get_q 等依赖 self.env
        # 高斯噪声标准差（位置观测），在 eval 时希望被记录
        self.noise_std = noise_std
        constraints = ConstraintsSet(dim_q, dim_x=dim_x)
        # print("obs:",base_env.task._obstacles)
        for obs in base_env.task._obstacles:
            if obs.name == "hazards":
                num = obs.num
                c_hazard = StateConstraint(dim_q=dim_q, dim_out=num, fun=self.hazard_f, jac_q=self.hazard_J,
                                    dim_x=dim_x,
                                    slack_type=slack_type, slack_beta=slack_beta_hazard_constraint,
                                    threshold=slack_threshold_hazard_constraint)
                constraints.add_constraint(c_hazard)

            if obs.name == "sigwalls":
                num = obs.num
                c_sigwall = StateConstraint(dim_q=dim_q, dim_out=num, fun=self.sigwall_f, jac_q=self.sigwall_J,
                                    dim_x=dim_x,
                                    slack_type=slack_type, slack_beta=slack_beta_sigwall_constraint,
                                    threshold=slack_threshold_sigwall_constraint)
                constraints.add_constraint(c_sigwall)
         
        atacom_step_size = timestep
        if update_with_agent_freq:
            atacom_step_size = timestep * n_intermediate_steps
        super().__init__(
            base_env,
            dq_max=1,
            constraints=constraints,
            step_size=atacom_step_size,
            update_with_agent_freq=update_with_agent_freq,
        )

    def _get_q(self):
        pos = self.env.task.agent.pos.copy()
        # 为位姿观测添加高斯噪声（仅在 noise_std > 0 时启用）
        if getattr(self, "noise_std", 0.0) > 0.0:
            noise = np.random.normal(
                loc=0.0,
                scale=self.noise_std,
                size=pos.shape,
            )
            pos = pos + noise
        theta = np.arctan2(
            self.env.task.agent.mat[1][0], self.env.task.agent.mat[0][0]
        )
        return np.array([pos[0], pos[1], theta])

    def hazard_f(self, q, x=None):
        h_num = 0
        h_pos = []
        for obs in self.env.task._obstacles:
            if obs.name == "hazards":
                h_num = obs.num
                h_pos = obs.pos
        h_f = np.zeros(h_num)
        for i in range(h_num):
            h_f[i] = -np.linalg.norm((q - h_pos[i])[:2]) + 0.2
        return h_f

    def hazard_J(self, q, x=None):
        J = []
        h_num = 0
        h_pos = []
        for obs in self.env.task._obstacles:
            if obs.name == "hazards":
                h_num = obs.num
                h_pos = obs.pos
        for i in range(h_num):
            dx = q[0] - h_pos[i][0]
            dy = q[1] - h_pos[i][1]
            denom = np.sqrt(dx**2 + dy**2)
            J.append([-dx / denom, -dy / denom, 0])
        return J @ self.J_drive(q)
    
    def sigwall_f(self, q, x=None):
        w_num = 0
        w_pos = 0
        for obs in self.env.task._obstacles:
            if obs.name == "sigwalls":
                w_num = obs.num
                w_pos = obs.locate_factor

        w_f = np.zeros(w_num)
        if w_num==2:
            w_f[0] = -self.q[0] - w_pos
            w_f[1] = self.q[0] - w_pos
        if w_num == 4:
            w_f[2] = -q[1] - w_pos
            w_f[3] = -q[0] - w_pos
        return w_f

    def sigwall_J(self, q, x=None):
        J = []
        w_num = 0
        for obs in self.env.task._obstacles:
            if obs.name == "sigwalls":
                w_num = obs.num
        if w_num==2:
            J.append([-1, 0, 0])
            J.append([1, 0, 0])
        if w_num==4:
            J.append([0, -1, 0])
            J.append([0, 1, 0])
        return J @ self.J_drive(q)


    def J_drive(self, q):
        theta = q[2]
        return np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0., 1.]])

    def compute_ctrl_action(self, action):
        return action

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.q = self._get_q()
        self.x = self._get_x()
        self.constraints.reset_slack(self.q[0], self.x)
        self.constraints.reset_slack(self.q[1], self.x)
        self.h_pos = np.empty((0, 3))
        self.h_num = 0
        return obs, info

    def step(self, action):
        alpha = np.clip(action, self.action_space.low, self.action_space.high)
        action_f = self.step_action_function(alpha)
        next_obs, reward, cost, terminated, truncated, info  = self.env.step(action_f) 
        self.q = self._get_q()
        self.x = self._get_x()
        self.dx = self._get_dx()

        # 修reward
        a = 0.01
        p = (action[0]-action_f[0])**2+(action[1]-action_f[1])**2
        reward = reward - a*p

        if not hasattr(self.env, "get_constraints_logs"):
            self._update_constraint_stats(self.q, self.x)
        return p, next_obs, reward, cost, terminated, truncated, info 
    
    def step_action_function(self, alpha):
        self.q = self._get_q()
        self.x = self._get_x()
        self.dx = self._get_dx()
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

    def reset_log_info(self):
        self.env.reset_log_info()

    def get_log_info(self):
        return self.env.get_log_info()


if __name__ == "__main__":
    mdp = TiagoNavAtacom(debug_gui=True, slack_type='softcorner')
    while True:
        mdp.reset()
        for i in range(mdp.info.horizon):
            # action = np.random.uniform(mdp.info.action_space.low, mdp.info.action_space.high)
            action = np.zeros_like(mdp.info.action_space.low)
            # action[0] = 1.
            state, reward, absorbing, _ = mdp.step(action)
            if absorbing:
                print("############ Collisions! ####################")
                break
        print(mdp.get_constraints_logs())
