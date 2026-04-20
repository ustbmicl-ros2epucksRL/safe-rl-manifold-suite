import numpy as np
import torch
from safepo.common.constrained_manifold.manifold import AtacomEnvWrapper
from safepo.common.constrained_manifold.constraints import StateConstraint, ConstraintsSet

class MultiNavAtacom(AtacomEnvWrapper):
    def __init__(self, base_env, gamma=0.99, horizon=500, step_action_function=None, timestep=1 / 30., n_intermediate_steps=1,
                 debug_gui=False, init_state=None, terminate_on_collision=True, save_key_frame=False,
                 slack_type='softcorner', update_with_agent_freq=True, is_opponent_moving=True,
                 slack_beta_hazard_constraint=0.1, 
                 slack_threshold_hazard_constraint=0.01, 
                 ):

        dim_q = 2 #base_env.action_space.shape[0]
        dim_x = 0
        constraints = ConstraintsSet(dim_q, dim_x=dim_x)
        self.num_agents = base_env.num_agents
        self.observation_spaces = base_env.observation_spaces
        self.share_observation_spaces = base_env.share_observation_spaces
        self.action_spaces = base_env.action_spaces
        for obs in base_env.task._obstacles:
            if obs.name == "hazards":
                num = obs.num
                c_hazard = StateConstraint(dim_q=dim_q, dim_out=num, fun=self.hazard_f, jac_q=self.hazard_J,
                                    dim_x=dim_x, 
                                    slack_type=slack_type, slack_beta=slack_beta_hazard_constraint,
                                    threshold=slack_threshold_hazard_constraint)
                constraints.add_constraint(c_hazard)
        
        atacom_step_size = timestep
        if update_with_agent_freq:
            atacom_step_size = timestep * n_intermediate_steps
        super().__init__(base_env, dq_max=1, constraints=constraints,
                         step_size=atacom_step_size, update_with_agent_freq=update_with_agent_freq)

    def _get_q(self):
        pos0 = self.env.task.agent.pos_0
        pos1 = self.env.task.agent.pos_1
        theta0 = np.arctan2(self.env.task.agent.mat_0[1][0],self.env.task.agent.mat_0[0][0])
        theta1 = np.arctan2(self.env.task.agent.mat_1[1][0],self.env.task.agent.mat_1[0][0])
        q = np.zeros((2,3))
        q[0] = (pos0[0],pos0[1],theta0)
        q[1] = (pos1[0],pos1[1],theta1)
        return q

    def hazard_f(self, q, x=None):
        for obs in self.env.task._obstacles:
            if obs.name == "hazards":
                h_num = obs.num
                h_pos = obs.pos
        h_f = np.zeros(h_num)
        for i in range(h_num):
            h_f = -np.linalg.norm((q - h_pos[i])[:2]) + 0.2
        return h_f

    def hazard_J(self, q, x=None):
        J = []
        for obs in self.env.task._obstacles:
            if obs.name == "hazards":
                h_num = obs.num
                h_pos = obs.pos
        for i in range(h_num):
            dx = q[0] - h_pos[i][0]
            dy = q[1] - h_pos[i][1]
            denom = np.sqrt(dx**2 + dy**2)
            J.append([dx / denom, dy / denom, 0])
        return J @ self.J_drive(q)


    def J_drive(self, q):
        theta = q[2]
        return np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0., 1.]])

    def compute_ctrl_action(self, action):
        return action
    
    def step(self, action):
        self.q = self._get_q()
        self.x = self._get_x()
        self.dx = self._get_dx()
        q0 = self.q[0]
        q1 = self.q[1]
        action0 = np.clip(action[0], self.action_space.low, self.action_space.high)
        action1 = np.clip(action[1], self.action_space.low, self.action_space.high)
        action[0] = torch.tensor(self.step_action_function(action0,q0))
        action[1] = torch.tensor(self.step_action_function(action1,q1))
        obs, share_obs, rewards, costs, dones, infos, avail_actions = self.env.step(action)
        self.q = self._get_q()
        self.x = self._get_x()
        self.dx = self._get_dx()

        if not hasattr(self.env, "get_constraints_logs"):
            self._update_constraint_stats(self.q, self.x)
        return obs, share_obs, rewards, costs, dones, infos, avail_actions 
    
    def step_action_function(self, alpha,q):
        self.x = self._get_x()
        self.dx = self._get_dx()
        self.constraints.reset_slack(q, self.x)

        Jc, J_x = self._construct_Jc(q, self.x)
        Jc_inv = Jc.T @ np.linalg.inv(Jc @ Jc.T)

        bases = np.diag(self.dq_max[:self.constraints.dim_null])
        # Projected Null Space
        Nc = (np.eye(Jc.shape[1]) - Jc_inv @ Jc)[:, :self.constraints.dim_null] @ bases

        alpha = np.array(alpha).reshape(-1)
        dq_null = Nc @ alpha
        dq_uncontrol = -Jc_inv @ J_x @ self.dx
        dq_err = -self.K_c * (Jc_inv @ self.constraints.c(q, self.x, origin_constr=False))
        dq = dq_null + dq_err + dq_uncontrol

        scale = self.action_truncation_scale(dq[:self.constraints.dim_q])
        dq = scale * dq
        ctrl_action = self.compute_ctrl_action(dq[:self.constraints.dim_q])
        return ctrl_action
    
    def action_truncation_scale(self, dq):
        scale = np.maximum((np.abs(dq) / self.env.single_action_space.high).max(), 1)
        return 1 / scale

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
