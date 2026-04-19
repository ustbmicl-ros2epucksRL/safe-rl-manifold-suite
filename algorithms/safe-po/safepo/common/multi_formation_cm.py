"""MultiFormationAtacom — ATACOM (约束流形零空间投影) 硬安全层包装器

针对 SafetyPointMultiFormationGoal0-v0 / MultiFormationNavEnv 任务的硬安全层。

设计动机:
    - 现有 safepo/common/mult_cm.py 的 MultiNavAtacom 硬编码 N=2 智能体，
      且只处理 hazards 类型障碍物。
    - MultiFormationGoalLevel0 任务使用 Sigwalls（两面墙）作为障碍，且
      num_agents 是运行时参数（N >= 2）。
    - 直接在 MultiNavAtacom 上打补丁会污染 chap3 单机实验代码，因此本类
      作为独立实现，与 MultiNavAtacom 职责划分清晰。

控制栈位置 (对应论文第 5 章 5.2.1 分层控制栈):
    MAPPO 策略 --a_RL--> RMPCorrector.apply_correction --a_soft-->
    env.step(a_soft) [被 MultiFormationAtacom 包裹]: step_action_function
    将 a_soft 投影到约束流形切空间 --a_safe--> 物理推进

约束定义:
    c_i(q) = d_safe - ||q_xy - wall_center||_xy  <= 0
    对每个机器人 i 与每面 sigwall 生成一个约束。

使用方法:
    base_env = MultiFormationNavEnv(task="SafetyPointMultiFormationGoal0-v0",
                                    seed=..., num_agents=4, ...)
    env = MultiFormationAtacom(base_env, d_safe=0.3)

状态: STUB (未完成) - 以下标注 TODO 的部分需在实现阶段补完
"""

from __future__ import annotations

import numpy as np
import torch

from safepo.common.constrained_manifold.manifold import AtacomEnvWrapper
from safepo.common.constrained_manifold.constraints import (
    StateConstraint,
    ConstraintsSet,
)


class MultiFormationAtacom(AtacomEnvWrapper):
    """N 智能体 + Sigwalls 版 ATACOM 包装器 (硬安全层).

    与 MultiNavAtacom 的差异:
        - 支持 num_agents >= 2 (N 动态)
        - 障碍源: Sigwalls (墙) 而非 Hazards
        - 约束生成: 每机器人 x 每墙 一个约束 => 总 N * M 个约束
    """

    def __init__(
        self,
        base_env,
        d_safe: float = 0.3,
        step_size: float = 1.0 / 30.0,
        slack_type: str = "softcorner",
        slack_beta: float = 0.1,
        slack_threshold: float = 0.01,
        update_with_agent_freq: bool = True,
        n_intermediate_steps: int = 1,
    ):
        """
        Args:
            base_env: MultiFormationNavEnv 实例 (未包裹)
            d_safe: 机器人到墙的最小安全距离
            step_size: ATACOM 积分步长 (通常等于控制周期)
            slack_type / slack_beta / slack_threshold: 松弛变量参数 (见 chap3.2.1)
            update_with_agent_freq: 是否按 agent 频率更新 (而非仿真频率)
            n_intermediate_steps: 每个 agent step 对应的仿真 substep 数
        """
        self.num_agents = base_env.num_agents
        self.observation_spaces = base_env.observation_spaces
        self.share_observation_spaces = base_env.share_observation_spaces
        self.action_spaces = base_env.action_spaces
        self.d_safe = float(d_safe)

        # 每个机器人的配置空间维度: (x, y, theta) = 3
        # ATACOM 内部 dim_q 指可控自由度维度: 差速驱动 = 2 (v, omega)
        dim_q = 2
        dim_x = 0

        # --- 提取 sigwall 位置 ---
        # TODO(stub): 确认 MultiFormationGoalLevel0 暴露的 sigwalls 接口
        #   方案 A: base_env.env.task.sigwalls.locations -> [(x0,y0), (x1,y1), ...]
        #   方案 B: 遍历 task._obstacles 找 name == "sigwalls"
        # 下面假设方案 A (与 multi_formation_level0.py:38-41 一致)
        self._sigwall_centers = self._extract_sigwall_centers(base_env)
        num_walls = len(self._sigwall_centers)

        # --- 构造约束集 ---
        # 单个 StateConstraint 输出维度 = N x M (每机器人 x 每墙)
        # 为避免维度展平复杂性, 这里每个机器人分别建一组约束
        constraints = ConstraintsSet(dim_q, dim_x=dim_x)
        for agent_idx in range(self.num_agents):
            c_wall = StateConstraint(
                dim_q=dim_q,
                dim_out=num_walls,
                fun=lambda q, x=None, i=agent_idx: self._wall_f(q, i),
                jac_q=lambda q, x=None, i=agent_idx: self._wall_J(q, i),
                dim_x=dim_x,
                slack_type=slack_type,
                slack_beta=slack_beta,
                threshold=slack_threshold,
            )
            constraints.add_constraint(c_wall)

        atacom_step_size = step_size
        if update_with_agent_freq:
            atacom_step_size = step_size * n_intermediate_steps

        super().__init__(
            base_env,
            dq_max=1,
            constraints=constraints,
            step_size=atacom_step_size,
            update_with_agent_freq=update_with_agent_freq,
        )

    # ---------------------- 约束函数族 ----------------------

    def _extract_sigwall_centers(self, base_env) -> list[np.ndarray]:
        """提取任务中 sigwall 的中心 XY 坐标."""
        # TODO(stub): 当前只支持 MultiFormationGoalLevel0; 未来可扩展:
        #   - 从 base_env.env.task 通过属性名 'sigwalls' 读取
        #   - 或从 _obstacles 列表按 name 匹配
        task = getattr(base_env, "env", base_env).task
        if hasattr(task, "sigwalls"):
            return [np.asarray(loc, dtype=np.float64) for loc in task.sigwalls.locations]
        # 退化: 使用任务默认值 (multi_formation_level0.py:38-41)
        return [np.array([-1.0, 0.0]), np.array([1.0, 0.0])]

    def _wall_f(self, q_all: np.ndarray, agent_idx: int) -> np.ndarray:
        """对第 agent_idx 个机器人, 返回与所有墙的约束向量.

        c_i(q) = d_safe - ||q_xy - wall_center||

        Returns:
            长度为 num_walls 的 ndarray; <= 0 表示安全
        """
        # TODO(stub): 验证 q_all 的形状 — 来自 AtacomEnvWrapper.step_action_function
        #   可能是 (dim_q,) 单机器人切片, 也可能是 (N, 3) 多机器人矩阵
        q_i = q_all if q_all.ndim == 1 else q_all[agent_idx]
        p_xy = np.asarray(q_i[:2], dtype=np.float64)
        out = np.empty(len(self._sigwall_centers), dtype=np.float64)
        for k, c in enumerate(self._sigwall_centers):
            out[k] = self.d_safe - float(np.linalg.norm(p_xy - c))
        return out

    def _wall_J(self, q_all: np.ndarray, agent_idx: int) -> np.ndarray:
        """约束对 q 的雅可比.

        d c_k / d (v, omega) = d c_k / d (x,y,theta) @ J_drive(theta)

        Returns:
            shape (num_walls, dim_q) 矩阵
        """
        q_i = q_all if q_all.ndim == 1 else q_all[agent_idx]
        p_xy = np.asarray(q_i[:2], dtype=np.float64)
        theta = float(q_i[2]) if q_i.shape[0] >= 3 else 0.0

        rows = []
        for c in self._sigwall_centers:
            dx, dy = p_xy[0] - c[0], p_xy[1] - c[1]
            denom = float(np.sqrt(dx * dx + dy * dy) + 1e-9)
            # d c_k / d (x, y, theta) = [-dx/|.|, -dy/|.|, 0]  (因为 c = d_safe - ||...||)
            rows.append([-dx / denom, -dy / denom, 0.0])
        grad_q = np.asarray(rows)  # (num_walls, 3)
        return grad_q @ self._J_drive(theta)

    @staticmethod
    def _J_drive(theta: float) -> np.ndarray:
        """差速驱动: d(x,y,theta)/d(v,omega) = [[cos,0],[sin,0],[0,1]]."""
        return np.array(
            [[np.cos(theta), 0.0],
             [np.sin(theta), 0.0],
             [0.0, 1.0]],
            dtype=np.float64,
        )

    # ---------------------- 状态获取 ----------------------

    def _get_q(self) -> np.ndarray:
        """获取所有 N 个机器人的 (x, y, theta).

        Returns:
            shape (num_agents, 3)
        """
        # TODO(stub): 验证 agent 接口 — MultiFormationGoalLevel0.calculate_reward
        # 使用 self.agent.pos_at(i), 所以下面的 pos_i / mat_i 可能需换成 pos_at(i)
        agent = self.env.task.agent
        q = np.zeros((self.num_agents, 3))
        for i in range(self.num_agents):
            pos = getattr(agent, f"pos_{i}", None)
            mat = getattr(agent, f"mat_{i}", None)
            if pos is None and hasattr(agent, "pos_at"):
                pos = agent.pos_at(i)
            if mat is None:
                theta = 0.0
            else:
                theta = float(np.arctan2(mat[1][0], mat[0][0]))
            q[i] = (float(pos[0]), float(pos[1]), theta)
        return q

    # ---------------------- step / action ----------------------

    def compute_ctrl_action(self, action):
        return action

    def step(self, action):
        """
        Args:
            action: List[torch.Tensor] 或 list of np.ndarray, 长度 = num_agents
        """
        self.q = self._get_q()          # (N, 3)
        self.x = self._get_x()
        self.dx = self._get_dx()

        # TODO(stub): 确认 step_action_function 的调用约定 —
        #   基类 AtacomEnvWrapper 可能期望对 (q, x, alpha) 整体处理;
        #   MultiNavAtacom 是逐机器人调用. 这里先逐机器人复用该范式.
        processed = [None] * self.num_agents
        for i in range(self.num_agents):
            q_i = self.q[i]
            a_i = action[i]
            if isinstance(a_i, torch.Tensor):
                a_i = a_i.cpu().numpy()
            a_i = np.clip(a_i, self.action_space.low, self.action_space.high)
            projected = self.step_action_function(a_i, q_i)
            processed[i] = torch.tensor(projected)

        # 替换原 action list
        for i in range(self.num_agents):
            action[i] = processed[i]

        obs, share_obs, rewards, costs, dones, infos, avail_actions = self.env.step(action)
        self.q = self._get_q()
        self.x = self._get_x()
        self.dx = self._get_dx()

        if not hasattr(self.env, "get_constraints_logs"):
            self._update_constraint_stats(self.q, self.x)
        return obs, share_obs, rewards, costs, dones, infos, avail_actions

    def step_action_function(self, alpha, q):
        """单个机器人的零空间投影 (沿用 chap3 公式).

        与 MultiNavAtacom.step_action_function 一致, 但约束函数不同.
        """
        self.x = self._get_x()
        self.dx = self._get_dx()
        self.constraints.reset_slack(q, self.x)

        Jc, J_x = self._construct_Jc(q, self.x)
        Jc_inv = Jc.T @ np.linalg.inv(Jc @ Jc.T)

        bases = np.diag(self.dq_max[: self.constraints.dim_null])
        Nc = (np.eye(Jc.shape[1]) - Jc_inv @ Jc)[:, : self.constraints.dim_null] @ bases

        alpha = np.asarray(alpha).reshape(-1)
        dq_null = Nc @ alpha
        dq_uncontrol = -Jc_inv @ J_x @ self.dx
        dq_err = -self.K_c * (Jc_inv @ self.constraints.c(q, self.x, origin_constr=False))
        dq = dq_null + dq_err + dq_uncontrol

        scale = self.action_truncation_scale(dq[: self.constraints.dim_q])
        dq = scale * dq
        return self.compute_ctrl_action(dq[: self.constraints.dim_q])

    def action_truncation_scale(self, dq):
        scale = np.maximum((np.abs(dq) / self.env.single_action_space.high).max(), 1)
        return 1 / scale

    def reset_log_info(self):
        self.env.reset_log_info()

    def get_log_info(self):
        return self.env.get_log_info()


# =============================================================
# 待实现 / 验证 清单 (TODO)
# =============================================================
# [1] _extract_sigwall_centers: 确认 task.sigwalls.locations 的访问路径
# [2] _get_q: 确认 agent.pos_i / mat_i 在 MultiFormationGoalLevel0 是否存在,
#     或改用 agent.pos_at(i) 与 agent.mat_at(i)
# [3] step: 验证 action list 的元素形状 (dim_q=2 还是 dim_action=2 一致)
# [4] _wall_f / _wall_J: q_all 的输入形状 (来自基类的约定)
# [5] 小规模 sanity check: 跑 2k steps,
#     - 期望: cost >> 0 (约束激活) 时动作修正幅度非零
#     - 期望: 约束 c(q) <= 0 恒成立 (硬保证)
# [6] 与 rmp_corrector 的 sigwall_centers 对齐 (本文件 d_safe 应该与
#     rmp_corrector 的 collision_safety_radius 保持一致性策略)
