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
"""Base task."""

from __future__ import annotations

import abc
import os
from collections import OrderedDict
from dataclasses import dataclass

import gymnasium
import mujoco
import numpy as np
import yaml

import safety_gymnasium
from safety_gymnasium.tasks.safe_multi_agent.bases.underlying import Underlying
from safety_gymnasium.tasks.safe_multi_agent.utils.common_utils import ResamplingError
from safety_gymnasium.tasks.safe_multi_agent.utils.task_utils import theta2vec


@dataclass
class LidarConf:
    r"""Lidar observation parameters.

    Attributes:
        num_bins (int): Bins (around a full circle) for lidar sensing.
        max_dist (float): Maximum distance for lidar sensitivity (if None, exponential distance).
        exp_gain (float): Scaling factor for distance in exponential distance lidar.
        type (str): 'pseudo', 'natural', see self._obs_lidar().
        alias (bool): Lidar bins alias into each other.
    """

    num_bins: int = 16
    max_dist: float = 3
    exp_gain: float = 1.0
    type: str = 'pseudo'
    alias: bool = True


@dataclass
class CompassConf:
    r"""Compass observation parameters.

    Attributes:
        shape (int): 2 for XY unit vector, 3 for XYZ unit vector.
    """

    shape: int = 2


@dataclass
class RewardConf:
    r"""Reward options.

    Attributes:
        reward_orientation (bool): Reward for being upright.
        reward_orientation_scale (float): Scale for uprightness reward.
        reward_orientation_body (str): What body to get orientation from.
        reward_exception (float): Reward when encountering a mujoco exception.
        reward_clip (float): Clip reward, last resort against physics errors causing magnitude spikes.
    """

    reward_orientation: bool = False
    reward_orientation_scale: float = 0.002
    reward_orientation_body: str = 'agent'
    reward_exception: float = -10.0
    reward_clip: float = 10


@dataclass
class CostConf:
    r"""Cost options.

    Attributes:
        constrain_indicator (bool): If true, all costs are either 1 or 0 for a given step.
    """

    constrain_indicator: bool = True


@dataclass
class MechanismConf:
    r"""Mechanism options.

    Starting position distribution.

    Attributes:
        randomize_layout (bool): If false, set the random seed before layout to constant.
        continue_goal (bool): If true, draw a new goal after achievement.
        terminate_resample_failure (bool): If true, end episode when resampling fails,
        otherwise, raise a python exception.
    """

    randomize_layout: bool = True
    continue_goal: bool = True
    terminate_resample_failure: bool = True


@dataclass
class ObservationInfo:
    r"""Observation information generated in running.

    Attributes:
        obs_space_dict (:class:`gymnasium.spaces.Dict`): Observation space dictionary.
    """

    obs_space_dict: gymnasium.spaces.Dict = None


class BaseTask(Underlying):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    r"""Base task class for defining some common characteristic and mechanism.

    Methods:

    - :meth:`dist_goal`: Return the distance from the agent to the goal XY position.
    - :meth:`calculate_cost`: Determine costs depending on the agent and obstacles, actually all
      cost calculation is done in different :meth:`safety_gymnasium.bases.base_obstacle.BaseObject.cal_cost`
      which implemented in different types of object, We just combine all results of them here.
    - :meth:`build_observation_space`: Build observation space, combine agent specific observation space
      and task specific observation space together.
    - :meth:`_build_placements_dict`: Build placement dictionary for all types of object.
    - :meth:`toggle_observation_space`: Toggle observation space.
    - :meth:`_build_world_config`: Create a world_config from all separate configs of different types of object.
    - :meth:`_build_static_geoms_config`: Build static geoms config from yaml files.
    - :meth:`build_goal_position`: Build goal position, it will be called when the task is initialized or
      when the goal is achieved.
    - :meth:`_placements_dict_from_object`: Build placement dictionary for a specific type of object.
    - :meth:`obs`: Combine and return all separate observations of different types of object.
    - :meth:`_obs_lidar`: Return lidar observation, unify natural lidar and pseudo lidar in API.
    - :meth:`_obs_lidar_natural`: Return natural lidar observation.
    - :meth:`_obs_lidar_pseudo`: Return pseudo lidar observation.
    - :meth:`_obs_compass`: Return compass observation.
    - :meth:`_obs_vision`: Return vision observation, that is RGB image captured by camera
      fixed in front of agent.
    - :meth:`_ego_xy`: Return the egocentric XY vector to a position from the agent.
    - :meth:`calculate_reward`: Calculate reward, it will be called in every timestep, and it is
      implemented in different task.
    - :meth:`specific_reset`: Reset task specific parameters, it will be called in every reset.
    - :meth:`specific_step`: Step task specific parameters, it will be called in every timestep.
    - :meth:`update_world`: Update world, it will be called when ``env.reset()`` or :meth:`goal_achieved` == True.

    Attributes:

    - :attr:`num_steps` (int): Maximum number of environment steps in an episode.
    - :attr:`lidar_conf` (:class:`LidarConf`): Lidar observation parameters.
    - :attr:`reward_conf` (:class:`RewardConf`): Reward options.
    - :attr:`cost_conf` (:class:`CostConf`): Cost options.
    - :attr:`mechanism_conf` (:class:`MechanismConf`): Mechanism options.
    - :attr:`action_space` (gymnasium.spaces.Box): Action space.
    - :attr:`observation_space` (gymnasium.spaces.Dict): Observation space.
    - :attr:`obs_info` (:class:`ObservationInfo`): Observation information generated in running.
    - :attr:`_is_load_static_geoms` (bool): Whether to load static geoms in current task which is mean
      some geoms that has no randomness.
    - :attr:`goal_achieved` (bool): Determine whether the goal is achieved, it will be called in every timestep
      and it is implemented in different task.
    """

    def __init__(self, config: dict) -> None:  # pylint: disable-next=too-many-statements
        """Initialize the task.

        Args:
            config (dict): Configuration dictionary, used to pre-config some attributes
              according to tasks via :meth:`safety_gymnasium.register`.
        """
        super().__init__(config=config)

        self.num_steps = 1000  # Maximum number of environment steps in an episode

        self.lidar_conf = LidarConf()
        self.compass_conf = CompassConf()
        self.reward_conf = RewardConf()
        self.cost_conf = CostConf()
        self.mechanism_conf = MechanismConf()

        self.action_space = self.agent.action_space
        self.observation_space = None
        self.obs_info = ObservationInfo()

        self._is_load_static_geoms = False  # Whether to load static geoms in current task.
        self.static_geoms_names: dict
        self.static_geoms_contact_cost: float = None
        self.contact_other_cost: float = None

    def dist_goal(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(self.goal.pos)  # pylint: disable=no-member

    def calculate_cost(self) -> dict:
        """Determine costs depending on the agent and obstacles."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        agent_keys = [f'agent_{i}' for i in range(self.agent.nums)]
        cost = {k: {} for k in agent_keys}

        # Calculate constraint violations
        for obstacle in self._obstacles:
            obj_cost = obstacle.cal_cost()
            for k in agent_keys:
                if k in obj_cost:
                    cost[k].update(obj_cost[k])

        if self.contact_other_cost:
            bis = self.agent.body_info
            for i in range(self.agent.nums):
                for j in range(i + 1, self.agent.nums):
                    g0, g1 = set(bis[i].geom_names), set(bis[j].geom_names)
                    for contact in self.data.contact[: self.data.ncon]:
                        geom_ids = [contact.geom1, contact.geom2]
                        geom_names = {self.model.geom(g).name for g in geom_ids}
                        if geom_names & g0 and geom_names & g1:
                            cost[f'agent_{i}']['cost_contact_other'] = self.contact_other_cost
                            cost[f'agent_{j}']['cost_contact_other'] = self.contact_other_cost
                            break

        if self._is_load_static_geoms and self.static_geoms_contact_cost:
            cost['cost_static_geoms_contact'] = 0.0
            all_agent_geoms = {n for bi in self.agent.body_info for n in bi.geom_names}
            for contact in self.data.contact[: self.data.ncon]:
                geom_ids = [contact.geom1, contact.geom2]
                geom_names = sorted([self.model.geom(g).name for g in geom_ids])
                if any(n in self.static_geoms_names for n in geom_names) and any(
                    n in all_agent_geoms for n in geom_names
                ):
                    # pylint: disable-next=no-member
                    cost['cost_static_geoms_contact'] += self.static_geoms_contact_cost

        # Sum all costs into single total cost
        for k, agent_cost in cost.items():
            if not k.startswith('agent_'):
                continue
            agent_cost['cost_sum'] = sum(v for k2, v in agent_cost.items() if k2.startswith('cost_'))
        return cost

    @staticmethod
    def _sensor_agent_index(sensor_name: str, num_agents: int) -> int:
        """Map MuJoCo sensor name suffix (``...11``) to agent index (supports up to 9 agents)."""
        for i in range(num_agents - 1, 0, -1):
            suf = str(i)
            if sensor_name.endswith(suf):
                return i
        return 0

    # pylint: disable-next=too-many-branches
    def build_observation_space(self) -> gymnasium.spaces.Dict:
        """Construct observation space.  Happens only once during __init__ in Builder."""
        obs_space_dict = OrderedDict()  # See self.obs()
        na = self.agent.nums

        sensor_dict = self.agent.build_sensor_observation_space()
        buckets = [OrderedDict() for _ in range(na)]
        for name, space in sensor_dict.items():
            idx = self._sensor_agent_index(name, na)
            buckets[idx][name] = space

        lid_box = lambda: gymnasium.spaces.Box(0.0, 1.0, (self.lidar_conf.num_bins,), dtype=np.float64)
        comp_box = lambda: gymnasium.spaces.Box(
            -1.0, 1.0, (self.compass_conf.shape,), dtype=np.float64
        )

        for i in range(na):
            obs_space_dict.update(buckets[i])
            lid_suffix = '' if i == 0 else str(i)
            for obstacle in self._obstacles:
                if obstacle.is_lidar_observed:
                    obs_space_dict[obstacle.name + '_lidar' + lid_suffix] = lid_box()
                if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                    obs_space_dict[obstacle.name + '_comp' + lid_suffix] = comp_box()
            if self.observe_vision:
                width, height = self.vision_env_conf.vision_size
                rows, cols = height, width
                self.vision_env_conf.vision_size = (rows, cols)
                obs_space_dict[f'vision_{i}'] = gymnasium.spaces.Box(
                    0,
                    255,
                    (*self.vision_env_conf.vision_size, 3),
                    dtype=np.uint8,
                )

        self.obs_info.obs_space_dict = gymnasium.spaces.Dict(obs_space_dict)

        if self.observation_flatten:
            self.observation_space = gymnasium.spaces.utils.flatten_space(
                self.obs_info.obs_space_dict,
            )
        else:
            self.observation_space = self.obs_info.obs_space_dict

    def _build_placements_dict(self) -> None:
        """Build a dict of placements.

        Happens only once.
        """
        placements = {}

        placements.update(self._placements_dict_from_object('agent'))
        for obstacle in self._obstacles:
            placements.update(self._placements_dict_from_object(obstacle.name))

        self.placements_conf.placements = placements

    def toggle_observation_space(self) -> None:
        """Toggle observation space."""
        self.observation_flatten = not self.observation_flatten
        self.build_observation_space()

    def _build_world_config(self, layout: dict) -> dict:  # pylint: disable=too-many-branches
        """Create a world_config from our own config."""
        na = self.agent.nums
        layout_agent = layout['agent']
        if isinstance(layout_agent, np.ndarray):
            layout_agent = layout_agent.reshape(-1).tolist()
        world_config = {
            'floor_type': self.floor_conf.type,
            'floor_size': self.floor_conf.size,
            'agent_base': self.agent.base,
            'agent_xy': layout_agent,
        }
        if self.agent.rot is None:
            world_config['agent_rot'] = self.random_generator.generate_rots(na)
        else:
            world_config['agent_rot'] = [float(self.agent.rot)] * na

        # process world config via different objects.
        world_config.update(
            {
                'geoms': {},
                'free_geoms': {},
                'mocaps': {},
            },
        )
        for obstacle in self._obstacles:
            num = obstacle.num if hasattr(obstacle, 'num') else 1
            if obstacle.name == 'agent':
                num = na
            obstacle.process_config(world_config, layout, self.random_generator.generate_rots(num))
        if self._is_load_static_geoms:
            self._build_static_geoms_config(world_config['geoms'])

        return world_config

    def _build_static_geoms_config(self, geoms_config: dict) -> None:
        """Load static geoms from .yaml file.

        Static geoms are geoms which won't be considered when calculate reward and cost in general.
        And have no randomness.
        Some tasks may generate cost when contacting static geoms.
        """
        env_info = self.__class__.__name__.split('Level')
        config_name = env_info[0].lower()
        level = int(env_info[1])

        # load all config of meshes in specific environment from .yaml file
        base_dir = os.path.dirname(safety_gymnasium.__file__)
        with open(os.path.join(base_dir, f'configs/{config_name}.yaml'), encoding='utf-8') as file:
            meshes_config = yaml.load(file, Loader=yaml.FullLoader)  # noqa: S506

        self.static_geoms_names = set()
        for idx in range(level + 1):
            for group in meshes_config[idx].values():
                geoms_config.update(group)
                for item in group.values():
                    self.static_geoms_names.add(item['name'])

    def build_goal_position(self) -> None:
        """Build a new goal position, maybe with resampling due to hazards."""
        # Resample until goal is compatible with layout
        if 'goal' in self.world_info.layout:
            del self.world_info.layout['goal']
        for _ in range(10000):  # Retries
            if self.random_generator.sample_goal_position():
                break
        else:
            raise ResamplingError('Failed to generate goal')
        # Move goal geom to new layout position
        if self.goal_achieved[0]:
            self.world_info.world_config_dict['geoms']['goal_red']['pos'][
                :2
            ] = self.world_info.layout['goal_red']
            self._set_goal('goal_red', self.world_info.layout['goal_red'])
        if self.goal_achieved[1]:
            self.world_info.world_config_dict['geoms']['goal_blue']['pos'][
                :2
            ] = self.world_info.layout['goal_blue']
            self._set_goal('goal_blue', self.world_info.layout['goal_blue'])
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def _placements_dict_from_object(self, object_name: dict) -> dict:
        """Get the placements dict subset just for a given object name."""
        placements_dict = {}

        assert hasattr(self, object_name), f'object{object_name} does not exist, but you use it!'
        data_obj = getattr(self, object_name)

        if hasattr(data_obj, 'num'):  # Objects with multiplicity
            object_fmt = object_name[:-1] + '{i}'
            object_num = getattr(data_obj, 'num', None)
            object_locations = getattr(data_obj, 'locations', [])
            object_placements = getattr(data_obj, 'placements', None)
            object_keepout = data_obj.keepout
        else:  # Unique objects
            object_fmt = object_name
            object_num = 1
            object_locations = getattr(data_obj, 'locations', [])
            object_placements = getattr(data_obj, 'placements', None)
            object_keepout = data_obj.keepout
        for i in range(object_num):
            if i < len(object_locations):
                x, y = object_locations[i]  # pylint: disable=invalid-name
                k = object_keepout + 1e-9  # Epsilon to account for numerical issues
                placements = [(x - k, y - k, x + k, y + k)]
            else:
                placements = object_placements
            placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
        return placements_dict

    def obs(self) -> dict | np.ndarray:
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor's data correct
        obs = {}

        obs.update(self.agent.obs_sensor())

        na = self.agent.nums
        for i in range(na):
            suf = '' if i == 0 else str(i)
            for obstacle in self._obstacles:
                if obstacle.is_lidar_observed:
                    if self.lidar_conf.type == 'pseudo':
                        obs[obstacle.name + '_lidar' + suf] = self._obs_lidar_pseudo_at(
                            obstacle.pos, i
                        )
                    elif self.lidar_conf.type == 'natural':
                        obs[obstacle.name + '_lidar' + suf] = self._obs_lidar_natural_at(
                            obstacle.group, i
                        )
                    else:
                        raise ValueError(f'Invalid lidar_type {self.lidar_conf.type}')
                if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                    obs[obstacle.name + '_comp' + suf] = self._obs_compass_at(obstacle.pos, i)

        if self.observe_vision:
            for i in range(na):
                cam = 'vision' if i == 0 else f'vision{i}'
                obs[f'vision_{i}'] = self._obs_vision(camera_name=cam)

        assert self.obs_info.obs_space_dict.contains(
            obs,
        ), f'Bad obs {obs} {self.obs_info.obs_space_dict}'

        if self.observation_flatten:
            obs = gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict, obs)
        return obs

    def _obs_lidar(self, positions: np.ndarray | list, group: int) -> np.ndarray:
        """Calculate and return a lidar observation.

        See sub methods for implementation.
        """
        if self.lidar_conf.type == 'pseudo':
            return self._obs_lidar_pseudo(positions)

        if self.lidar_conf.type == 'natural':
            return self._obs_lidar_natural(group)

        raise ValueError(f'Invalid lidar_type {self.lidar_conf.type}')

    def _obs_lidar1(self, positions: np.ndarray | list, group: int) -> np.ndarray:
        """Calculate and return a lidar observation.

        See sub methods for implementation.
        """
        if self.lidar_conf.type == 'pseudo':
            return self._obs_lidar_pseudo1(positions)

        if self.lidar_conf.type == 'natural':
            return self._obs_lidar_natural(group)

        raise ValueError(f'Invalid lidar_type {self.lidar_conf.type}')

    def _obs_lidar_natural(self, group: int) -> np.ndarray:
        """Natural lidar casts rays based on the ego-frame of agent 0."""
        return self._obs_lidar_natural_at(group, 0)

    def _obs_lidar_natural_at(self, group: int, agent_index: int) -> np.ndarray:
        """Natural lidar from agent ``agent_index``'s body frame."""
        bname = self.agent.body_name_for_index(agent_index)
        body = self.model.body(bname).id
        # pylint: disable-next=no-member
        grp = np.asarray([i == group for i in range(int(mujoco.mjNGROUP))], dtype='uint8')
        pos = np.asarray(self.agent.pos_at(agent_index), dtype='float64')
        mat_t = self.agent.mat_at(agent_index)
        obs = np.zeros(self.lidar_conf.num_bins)
        for i in range(self.lidar_conf.num_bins):
            theta = (i / self.lidar_conf.num_bins) * np.pi * 2
            vec = np.matmul(mat_t, theta2vec(theta))
            vec = np.asarray(vec, dtype='float64')
            geom_id = np.array([0], dtype='int32')
            dist = mujoco.mj_ray(  # pylint: disable=no-member
                self.model,
                self.data,
                pos,
                vec,
                grp,
                1,
                body,
                geom_id,
            )
            if dist >= 0:
                obs[i] = np.exp(-dist)
        return obs

    def _obs_lidar_pseudo(self, positions: np.ndarray) -> np.ndarray:
        """Pseudo lidar from agent 0's frame."""
        return self._obs_lidar_pseudo_at(positions, 0)

    def _obs_lidar_pseudo1(self, positions: np.ndarray) -> np.ndarray:
        """Pseudo lidar from agent 1's frame."""
        return self._obs_lidar_pseudo_at(positions, 1)

    def _obs_lidar_pseudo_at(self, positions: np.ndarray, agent_index: int) -> np.ndarray:
        """Pseudo lidar in the egocentric frame of agent ``agent_index``."""
        positions = np.array(positions, ndmin=2)
        obs = np.zeros(self.lidar_conf.num_bins)
        for pos in positions:
            pos = np.asarray(pos)
            if pos.shape == (3,):
                pos = pos[:2]
            z = complex(*self._ego_xy_at(pos, agent_index))
            dist = np.abs(z)
            angle = np.angle(z) % (np.pi * 2)
            bin_size = (np.pi * 2) / self.lidar_conf.num_bins
            bin = int(angle / bin_size)  # pylint: disable=redefined-builtin
            bin_angle = bin_size * bin
            if self.lidar_conf.max_dist is None:
                sensor = np.exp(-self.lidar_conf.exp_gain * dist)
            else:
                sensor = max(0, self.lidar_conf.max_dist - dist) / self.lidar_conf.max_dist
            obs[bin] = max(obs[bin], sensor)
            if self.lidar_conf.alias:
                alias = (angle - bin_angle) / bin_size
                assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin}'
                bin_plus = (bin + 1) % self.lidar_conf.num_bins
                bin_minus = (bin - 1) % self.lidar_conf.num_bins
                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
        return obs

    def _obs_compass(self, pos: np.ndarray) -> np.ndarray:
        """Compass from agent 0 to ``pos``."""
        return self._obs_compass_at(pos, 0)

    def _obs_compass_at(self, pos: np.ndarray, agent_index: int) -> np.ndarray:
        """Compass from agent ``agent_index`` to ``pos`` (egocentric unit XY vector)."""
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.concatenate([pos, [0]])
        vec = pos - self.agent.pos_at(agent_index)
        vec = np.matmul(vec, self.agent.mat_at(agent_index))
        vec_2 = vec[: self.compass_conf.shape]
        vec_2 /= np.sqrt(np.sum(np.square(vec_2))) + 0.001
        assert vec_2.shape == (self.compass_conf.shape,), f'Bad vec {vec_2}'
        return vec_2

    def _obs_vision(self, camera_name='vision') -> np.ndarray:
        """Return pixels from the agent camera.

        Note:
            This is a 3D array of shape (rows, cols, channels).
            The channels are RGB, in that order.
            If you are on a headless machine, you may need to checkout this:
            URL: `issue <https://github.com/PKU-Alignment/safety-gymnasium/issues/27>`_
        """
        rows, cols = self.vision_env_conf.vision_size
        width, height = cols, rows
        return self.render(
            width,
            height,
            mode='rgb_array',
            camera_name=camera_name,
            cost={f'agent_{i}': {} for i in range(self.agent.nums)},
        )

    def _ego_xy(self, pos: np.ndarray) -> np.ndarray:
        """Return the egocentric XY vector to a position from agent 0."""
        return self._ego_xy_at(pos, 0)

    def _ego_xy1(self, pos: np.ndarray) -> np.ndarray:
        """Return the egocentric XY vector to a position from agent 1."""
        return self._ego_xy_at(pos, 1)

    def _ego_xy_at(self, pos: np.ndarray, agent_index: int) -> np.ndarray:
        """Egocentric XY vector to ``pos`` from agent ``agent_index``."""
        assert pos.shape == (2,), f'Bad pos {pos}'
        agent_3vec = self.agent.pos_at(agent_index)
        agent_mat = self.agent.mat_at(agent_index)
        pos_3vec = np.concatenate([pos, [0]])
        world_3vec = pos_3vec - agent_3vec
        return np.matmul(world_3vec, agent_mat)[:2]

    @abc.abstractmethod
    def calculate_reward(self) -> float:
        """Determine reward depending on the agent and tasks."""

    @abc.abstractmethod
    def specific_reset(self) -> None:
        """Set positions and orientations of agent and obstacles."""

    @abc.abstractmethod
    def specific_step(self) -> None:
        """Each task can define a specific step function.

        It will be called when :meth:`safety_gymnasium.builder.Builder.step()` is called using env.step().
        For example, you can do specific data modification.
        """

    @abc.abstractmethod
    def update_world(self) -> None:
        """Update one task specific goal."""

    @property
    @abc.abstractmethod
    def goal_achieved(self) -> bool:
        """Check if task specific goal is achieved."""
