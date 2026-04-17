# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hazard."""

from dataclasses import dataclass

import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.assets.color import COLOR
from safety_gymnasium.tasks.safe_multi_agent.assets.group import GROUP
from safety_gymnasium.tasks.safe_multi_agent.bases.base_object import Geom


@dataclass
class Sigwalls(Geom):  # pylint: disable=too-many-instance-attributes
    """Non collision object."""

    name: str = 'sigwalls'
    num: int = 2
    locate_factor: float = 1.125
    size: float = 3.5
    placements: list = None
    keepout: float = 0.0

    color: np.array = COLOR['sigwall']
    group: np.array = GROUP['sigwall']
    is_lidar_observed: bool = False
    is_constrained: bool = False

    def __post_init__(self) -> None:
        assert self.num in (2, 4), 'Sigwalls are specific for Circle and Run tasks.'
        assert (
            self.locate_factor >= 0
        ), 'For cost calculation, the locate_factor must be greater than or equal to zero.'
        self.locations: list = [
            (self.locate_factor, 0),
            (-self.locate_factor, 0),
            (0, self.locate_factor),
            (0, -self.locate_factor),
        ]

        self.index: int = 0

    def index_tick(self):
        """Count index."""
        self.index += 1
        self.index %= self.num

    def get_config(self, xy_pos, rot):  # pylint: disable=unused-argument
        """To facilitate get specific config for this object."""
        geom = {
            'name': self.name,
            'size': np.array([0.05, self.size, 0.3]),
            'pos': np.r_[xy_pos, 0.25],
            'rot': 0,
            'type': 'box',
            'contype': 0,
            'conaffinity': 0,
            'group': self.group,
            'rgba': self.color * [1, 1, 1, 0.1],
        }
        if self.index >= 2:
            geom.update({'rot': np.pi / 2})
        self.index_tick()
        return geom

    def cal_cost(self):
        """Contacts Processing."""
        na = self.agent.nums
        cost = {f'agent_{i}': {} for i in range(na)}
        if not self.is_constrained:
            return cost
        
        # 对于多智能体环境，需要检查所有智能体
        # 根据实际墙的位置检查边界（x 坐标应该在 -0.8 到 0.8 之间）
        # 如果 locations 被手动设置，使用实际墙的位置
        if hasattr(self, 'locations') and len(self.locations) >= 2:
            # 使用实际墙的位置来确定边界
            left_wall_center_x = self.locations[0][0] if len(self.locations) > 0 else -self.locate_factor
            right_wall_center_x = self.locations[1][0] if len(self.locations) > 1 else self.locate_factor
            
            # 墙的厚度是 0.05（从 get_config 中的 size[0]）
            wall_thickness = 0.05
            # 墙的中心到边缘的距离是厚度的一半
            wall_half_thickness = wall_thickness / 2.0
            
            # 计算有效区域的边界（墙的内侧边缘）
            # 左墙的右边缘（机器人可以到达的最左侧）
            left_boundary = left_wall_center_x + wall_half_thickness
            # 右墙的左边缘（机器人可以到达的最右侧）
            right_boundary = right_wall_center_x - wall_half_thickness
            
            # 机器人半径约 0.05，所以机器人中心应该在有效区域内
            # 考虑到浮点误差和初始位置生成，使用更宽松的边界
            agent_radius = 0.05
            tolerance = 0.05  # 增加浮点误差容忍度，避免初始位置误报
            
            # 有效区域：机器人中心应该在 [left_boundary + agent_radius, right_boundary - agent_radius] 内
            # 超出这个范围就认为越界
            safe_left = left_boundary + agent_radius - tolerance
            safe_right = right_boundary - agent_radius + tolerance
            
            for i in range(na):
                ax = self.agent.pos_at(i)[0]
                if (ax < safe_left) or (ax > safe_right):
                    cost[f'agent_{i}']['cost_out_of_boundary'] = 1.0
        else:
            for i in range(na):
                p = self.agent.pos_at(i)
                out = np.abs(p[0]) > self.locate_factor
                if self.num == 4:
                    out = out or np.abs(p[1]) > self.locate_factor
                if out:
                    cost[f'agent_{i}']['cost_out_of_boundary'] = 1.0

        return cost

    @property
    def pos(self):
        """Helper to get list of Sigwalls positions."""
