"""Multi Formation level 0."""

from __future__ import annotations

import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import Goal, Sigwalls
from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask
from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_formation.formation_reward_edges import (
    build_formation_distance_constraints,
)

_GEOM_EPS = 1e-9


class MultiFormationGoalLevel0(BaseTask):
    """An agent must navigate to a formation."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        # self.placements_conf.extents = [-1, -1, 1, 1]
        # 调整初始位置范围，确保机器人在安全区域内生成（避免与墙碰撞）
        # 墙在 x = -0.8 和 x = 0.8，安全区域大约是 x = -0.7 到 0.7
        self.agent.placements = [(-0.7, -0.8, 0.7, 0.8)]
        self.agent.keepout = 0

        # 创建 Sigwalls 和 Goal 实例
        sigwalls = Sigwalls(num=2, locate_factor=1.125, is_constrained=True)
        goal = Goal(keepout=0.305, locations=[(0, 2.0)])  # 目标固定在通道尾部

        self._add_geoms(
            goal,
            sigwalls,
        )

        # 设置两个墙的固定位置，形成狭长通道
        sigwalls.locations = [
            (-1.0, 0),  # 左侧墙
            (1.0, 0),  # 右侧墙
        ]

        # 目标相关奖励权重（提高权重，增加目标导航占 reward 的比例）
        self.goal.reward_distance = 2.0  # 靠近目标的稠密奖励系数，强化目标导航
        self.goal.reward_goal = 2.5  # 到达目标的稀疏奖励

        self._reset_dist_cache()

        # 编队：与 RMP 拓扑一致的约束边 + 形状相关的辅助成形项
        # （字段由 Underlying._parse 从 config 写入 self）
        if self.formation_target_distance is not None:
            self.formation_target_distance = float(self.formation_target_distance)
        else:
            self.formation_target_distance = 0.4

        self.formation_tolerance = float(
            self.formation_tolerance if self.formation_tolerance is not None else 0.1,
        )
        self.formation_reward_scale = float(
            self.formation_reward_scale if self.formation_reward_scale is not None else 0.04,
        )
        self.mesh_alignment_reward_scale = float(
            self.mesh_alignment_reward_scale
            if self.mesh_alignment_reward_scale is not None
            else 0.025,
        )
        self.line_width_reward_scale = float(
            self.line_width_reward_scale if self.line_width_reward_scale is not None else 0.02,
        )
        self.boundary_penalty_scale = 0.2
        self._reward_clip = 10.0

        self._formation_shape = str(self.formation_shape).strip().lower()
        line_axis = str(self.formation_line_axis).strip().lower()
        wedge_deg = float(self.formation_wedge_half_angle_deg)
        gd = np.asarray(self.formation_desired_direction, dtype=np.float64).reshape(2)

        self._formation_line_axis = line_axis
        self._formation_edges, self._mesh_r_star = build_formation_distance_constraints(
            self._formation_shape,
            self.agent.nums,
            self.formation_target_distance,
            gd,
            line_axis=line_axis,
            wedge_half_angle_rad=np.deg2rad(wedge_deg),
        )

    def _agent_keys(self):
        return [f'agent_{i}' for i in range(self.agent.nums)]

    def _reset_dist_cache(self):
        self.last_dist_goal = {k: None for k in self._agent_keys()}

    def dist_goal(self, agent_id: int = None) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal'), 'Please make sure you have added goal into env.'
        if agent_id is None:
            agent_id = 0
        return self.agent.dist_xy(agent_id, self.goal.pos)  # pylint: disable=no-member

    def _shape_auxiliary_reward(self, positions: list[np.ndarray]) -> float:
        """Shape-dependent reward shaping beyond pairwise distance (see formation_reward_edges)."""
        na = len(positions)
        if na < 2:
            return 0.0

        if self._formation_shape in ('mesh', 'full', 'default', 'complete'):
            if self._mesh_r_star is None or not self._formation_edges:
                return 0.0
            dots = []
            for i, j, _ in self._formation_edges:
                delta = np.asarray(positions[j], dtype=np.float64) - np.asarray(
                    positions[i],
                    dtype=np.float64,
                )
                n = float(np.linalg.norm(delta))
                if n < _GEOM_EPS:
                    continue
                hats = delta / n
                dots.append(float(np.dot(hats, self._mesh_r_star)))
            if not dots:
                return 0.0
            return self.mesh_alignment_reward_scale * float(np.mean(dots))

        if self._formation_shape in ('line', 'line_horizontal', 'row'):
            axis = self._formation_line_axis
            if axis in ('x', 'horizontal', 'h'):
                spread = float(np.var([float(p[1]) for p in positions]))
            else:
                spread = float(np.var([float(p[0]) for p in positions]))
            return -self.line_width_reward_scale * spread

        # wedge / circle: distance constraints carry the geometry (no y-variance, etc.)
        return 0.0

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = {k: 0.0 for k in self._agent_keys()}
        cost = self.calculate_cost()

        na = self.agent.nums
        positions = [self.agent.pos_at(i)[:2] for i in range(na)]

        if self._formation_edges:
            dist_errors = []
            for i, j, d_t in self._formation_edges:
                pi = np.asarray(positions[i], dtype=np.float64)
                pj = np.asarray(positions[j], dtype=np.float64)
                dist_errors.append(abs(float(np.linalg.norm(pi - pj)) - float(d_t)))
            distance_error = float(np.mean(dist_errors))
        else:
            distance_error = 0.0

        formation_reward = self.formation_reward_scale * np.exp(
            -distance_error / self.formation_tolerance,
        )
        shape_aux = self._shape_auxiliary_reward(positions)

        for agent_id, agent_name in enumerate(self._agent_keys()):
            dist_goal = self.dist_goal(agent_id)

            if self.last_dist_goal[agent_name] is not None:
                dist_delta = self.last_dist_goal[agent_name] - dist_goal
                dist_delta = np.clip(dist_delta, -0.8, 0.8)
                reward[agent_name] += dist_delta * self.goal.reward_distance
            self.last_dist_goal[agent_name] = dist_goal

            if self.goal_achieved[agent_id]:
                reward[agent_name] += self.goal.reward_goal

            reward[agent_name] += formation_reward
            reward[agent_name] += shape_aux

            if cost[agent_name].get('cost_out_of_boundary', 0) > 0:
                reward[agent_name] -= self.boundary_penalty_scale

            reward[agent_name] = float(
                np.clip(reward[agent_name], -self._reward_clip, self._reward_clip),
            )

        return reward

    def specific_reset(self):
        """Reset task-specific parameters."""
        self._reset_dist_cache()

    def specific_step(self):
        pass

    def build_goal_position(self) -> None:
        """Build goal position at fixed location (channel end)."""
        import mujoco

        fixed_goal_pos = np.array([0, 2.0])

        if 'goal' in self.world_info.layout:
            del self.world_info.layout['goal']
        self.world_info.layout['goal'] = fixed_goal_pos

        if 'goal' in self.world_info.world_config_dict['geoms']:
            self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = fixed_goal_pos
            self._set_goal('goal', fixed_goal_pos)

        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        for agent_id, agent_name in enumerate(self._agent_keys()):
            self.last_dist_goal[agent_name] = self.dist_goal(agent_id)

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved for each agent."""
        return tuple(self.dist_goal(i) <= self.goal.size for i in range(self.agent.nums))
