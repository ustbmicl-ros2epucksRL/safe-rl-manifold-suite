import numpy as np
from safepo.common.constrained_manifold.manifold import AtacomEnvWrapper
from safepo.common.constrained_manifold.constraints import StateConstraint, ConstraintsSet

class SingleNavAtacomV2(AtacomEnvWrapper):
    def __init__(self, base_env, gamma=0.99, horizon=500, step_action_function=None, timestep=1 / 30., n_intermediate_steps=1,
                 debug_gui=False, init_state=None, terminate_on_collision=True, save_key_frame=False,
                 slack_type='softcorner', update_with_agent_freq=True, is_opponent_moving=True,
                 slack_beta_hazard_constraint=1, slack_beta_sigwall_constraint=0.5,
                 slack_threshold_hazard_constraint=0.01, slack_threshold_sigwall_constraint=0.1,
                 ):

        self.dim_q = 2 #base_env.action_space.shape[0]
        self.dim_x = 0
        self.h_pos = np.empty((0, 3))
        self.h_num = 0
        self.env = base_env

        constraints = ConstraintsSet(self.dim_q, dim_x=self.dim_x)
        q = self._get_q()
        
        # 如果有距离小于0.8的障碍物，则添加障碍物约束
        for obs in base_env.task._obstacles:
            if obs.name == "hazards" :
                # print("hazard pos:" + str(obs.pos))
                for pos in obs.pos:
                    if self.dist(pos,q) < 0.8:
                        self.h_num += 1
                        self.h_pos = np.vstack([self.h_pos, pos])
        # 添加约束（即使h_num为0也要添加，以确保Jacobian矩阵维度一致）
        c_hazard = StateConstraint(dim_q=self.dim_q, dim_out=self.h_num, fun=self.h_f, jac_q=self.h_J,
                                dim_x=self.dim_x,
                                slack_type=slack_type, slack_beta=slack_beta_hazard_constraint,
                                threshold=slack_threshold_hazard_constraint)
        constraints.add_constraint(c_hazard)
         
        atacom_step_size = timestep
        if update_with_agent_freq:
            atacom_step_size = timestep * n_intermediate_steps
        super().__init__(base_env, dq_max=1, constraints=constraints,
                         step_size=atacom_step_size, update_with_agent_freq=update_with_agent_freq)

    def _get_q(self):
        pos = self.env.task.agent.pos
        # add Gauss Noise to pose observation
        # noise_std = 7
        # noise = np.random.normal(loc=0.0, scale=np.sqrt(noise_std), size=pos.shape)
        # pos = pos + noise
        theta = np.arctan2(self.env.task.agent.mat[1][0],self.env.task.agent.mat[0][0])
        return np.array([pos[0], pos[1], theta])

    def h_f(self, q, x=None):
        h_f = np.zeros(self.h_num)
        for i in range(self.h_num):
            h_f[i] = -np.linalg.norm((q - self.h_pos[i])[:2]) + 0.2
        return h_f

    def h_J(self, q, x=None):
        J = []
        for i in range(self.h_num):
            dx = q[0] - self.h_pos[i][0]
            dy = q[1] - self.h_pos[i][1]
            denom = np.sqrt(dx**2 + dy**2)
            J.append([-dx / denom, -dy / denom, 0])
        if not len(J):
            return np.zeros((0, 3))  # Return empty matrix with 3 columns when no constraints
        return J @ self.J_drive(q)
    
    # def update_f_J(self, q, x=None):
    #     h_num = 0
    #     h_f = []
    #     J = []
    #     for obs in self.env.task._obstacles:
    #         if obs.name == "hazards" and self.dist(obs.pos, q) < 0.8 and (obs.pos not in self.h_pos):
    #             # update_pos
    #             self.h_pos.append(obs.pos)

    #             # update_num
    #             h_num += 1
                
    #             # update_f
    #             h_f.append(-np.linalg.norm((q - obs.pos)[:2]) + 0.2)
                
    #             # update_J
    #             dx = q[0] - obs.pos[0]
    #             dy = q[1] - obs.pos[1]
    #             denom = np.sqrt(dx**2 + dy**2)
    #             J.append([-dx / denom, -dy / denom, 0])
    #     return h_num, h_f, J @ self.J_drive(q)

    def J_drive(self, q):
        theta = q[2]
        return np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0., 1.]])

    def compute_ctrl_action(self, action):
        return action
    
    def step(self, action):
        alpha = np.clip(action, self.action_space.low, self.action_space.high)
        if self.h_num > 0:
            action_f = self.step_action_function(alpha)
        else:
            action_f = alpha
        next_obs, reward, cost, terminated, truncated, info  = self.env.step(action_f) 
        self.q = self._get_q()
        self.x = self._get_x()
        self.dx = self._get_dx()

        # 如果有距离小于0.8的障碍物且不在已有约束中，则添加障碍物约束
        for obs in self.env.task._obstacles:
            if obs.name == "hazards":
                # print("hazard pos:" + str(obs.pos))
                for pos in obs.pos:
                    # if self.dist(pos,self.q) < 0.8 and not np.any(np.all(self.h_pos == pos, axis=1)):
                    if self.dist(pos, self.q) < 0.8 and (
                            self.h_pos.shape[0] == 0 or not np.any(np.all(self.h_pos == pos, axis=1))
                        ):

                        self.h_num +=1
                        self.h_pos = np.vstack([self.h_pos, pos])
                        self.constraints.dim_out = self.h_num
                        self.constraints.constraints_list[0].dim_out = self.h_num
                        self.constraints.dim_slack = self.h_num
                        self.constraints.constraints_list[0].dim_slack = self.h_num
                        # print("new hazard added! now:"+str(self.h_num) + " pos" + str(pos))

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
    
    def dist(self, pos, q):
        return float(np.linalg.norm((q - pos)[:2]))
