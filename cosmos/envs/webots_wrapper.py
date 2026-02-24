"""
Webots E-puck Multi-Robot Environment Wrapper

Integration with Webots simulator for realistic robot simulation.

Installation:
    1. Install Webots: https://cyberbotics.com/
    2. pip install controller  # Webots Python API
    3. Set WEBOTS_HOME environment variable

E-puck Robot:
    - Differential drive mobile robot
    - 8 infrared proximity sensors
    - Camera, LEDs, speaker
    - Common platform for swarm robotics research

Reference:
    https://cyberbotics.com/doc/guide/epuck
    https://www.gctronic.com/doc/index.php/e-puck2
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import os
import subprocess
import time

from cosmos.registry import ENV_REGISTRY
from cosmos.envs.base import BaseMultiAgentEnv

# Check Webots availability
WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "")
WEBOTS_AVAILABLE = os.path.exists(WEBOTS_HOME) if WEBOTS_HOME else False

try:
    # Add Webots Python path
    if WEBOTS_HOME:
        import sys
        sys.path.append(os.path.join(WEBOTS_HOME, "lib", "controller", "python"))
    from controller import Supervisor, Robot
    WEBOTS_CONTROLLER_AVAILABLE = True
except ImportError:
    WEBOTS_CONTROLLER_AVAILABLE = False


class EpuckController:
    """
    Controller for a single E-puck robot in Webots.

    E-puck specifications:
    - Wheel radius: 0.0205 m
    - Axle length: 0.052 m
    - Max wheel speed: 6.28 rad/s (~1000 steps/s)
    - 8 proximity sensors (IR)
    - Ground sensors optional
    """

    WHEEL_RADIUS = 0.0205  # meters
    AXLE_LENGTH = 0.052    # meters
    MAX_SPEED = 6.28       # rad/s

    def __init__(self, robot: 'Robot', timestep: int = 64):
        self.robot = robot
        self.timestep = timestep

        # Motors
        self.left_motor = robot.getDevice("left wheel motor")
        self.right_motor = robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        # Proximity sensors
        self.proximity_sensors = []
        for i in range(8):
            sensor = robot.getDevice(f"ps{i}")
            sensor.enable(timestep)
            self.proximity_sensors.append(sensor)

        # Position sensor (encoders)
        self.left_encoder = robot.getDevice("left wheel sensor")
        self.right_encoder = robot.getDevice("right wheel sensor")
        if self.left_encoder:
            self.left_encoder.enable(timestep)
        if self.right_encoder:
            self.right_encoder.enable(timestep)

    def get_proximity_readings(self) -> np.ndarray:
        """Get normalized proximity sensor readings [0, 1]."""
        readings = []
        for sensor in self.proximity_sensors:
            value = sensor.getValue()
            # Normalize: 0 = far, 1 = close
            normalized = min(value / 4096.0, 1.0)
            readings.append(normalized)
        return np.array(readings)

    def set_velocity(self, left_speed: float, right_speed: float):
        """Set wheel velocities in rad/s."""
        left_speed = np.clip(left_speed, -self.MAX_SPEED, self.MAX_SPEED)
        right_speed = np.clip(right_speed, -self.MAX_SPEED, self.MAX_SPEED)
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def set_velocity_from_action(self, action: np.ndarray):
        """
        Convert action to wheel velocities.

        Action format: [v, omega] where
        - v: linear velocity (-1 to 1)
        - omega: angular velocity (-1 to 1)
        """
        v = action[0] * self.MAX_SPEED * self.WHEEL_RADIUS  # m/s
        omega = action[1] * self.MAX_SPEED / (self.AXLE_LENGTH / 2)  # rad/s

        # Differential drive kinematics
        left_speed = (v - omega * self.AXLE_LENGTH / 2) / self.WHEEL_RADIUS
        right_speed = (v + omega * self.AXLE_LENGTH / 2) / self.WHEEL_RADIUS

        self.set_velocity(left_speed, right_speed)


@ENV_REGISTRY.register("webots_epuck", aliases=["webots", "epuck"])
class WebotsEpuckEnv(BaseMultiAgentEnv):
    """
    Webots E-puck Multi-Robot Environment.

    Simulates multiple E-puck robots in Webots for:
    - Formation control
    - Swarm robotics
    - Collective navigation
    - Sim-to-real transfer

    Config options:
        world_file: Path to Webots world file
        num_agents: Number of E-puck robots (default: 4)
        timestep: Simulation timestep in ms (default: 64)
        max_steps: Maximum steps per episode (default: 1000)
        arena_size: Arena size in meters (default: 1.0)
        headless: Run without GUI (default: False)

    Observation space (per agent):
        - 8 proximity sensor readings [0, 1]
        - Relative position to goal [x, y]
        - Relative positions to other agents [x, y] * (n-1)
        Total: 8 + 2 + 2*(n-1) dimensions

    Action space (per agent):
        - Linear velocity [-1, 1]
        - Angular velocity [-1, 1]
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        world_file: Optional[str] = None,
        num_agents: int = 4,
        timestep: int = 64,
        max_steps: int = 1000,
        arena_size: float = 1.0,
        headless: bool = False,
        **kwargs
    ):
        if not WEBOTS_AVAILABLE:
            raise ImportError(
                "Webots not found. Install Webots and set WEBOTS_HOME:\n"
                "  1. Download from https://cyberbotics.com/\n"
                "  2. export WEBOTS_HOME=/path/to/webots"
            )

        if cfg:
            world_file = cfg.get("world_file", world_file)
            num_agents = cfg.get("num_agents", num_agents)
            timestep = cfg.get("timestep", timestep)
            max_steps = cfg.get("max_steps", max_steps)
            arena_size = cfg.get("arena_size", arena_size)
            headless = cfg.get("headless", headless)

        self._num_agents = num_agents
        self._timestep = timestep
        self._max_steps = max_steps
        self._arena_size = arena_size
        self._headless = headless

        # Observation: 8 proximity + 2 goal + 2*(n-1) neighbors
        self._obs_dim = 8 + 2 + 2 * (num_agents - 1)
        self._act_dim = 2  # [v, omega]
        self._share_obs_dim = self._obs_dim * num_agents

        # World file
        if world_file is None:
            world_file = self._create_default_world()
        self._world_file = world_file

        # Will be initialized when Webots starts
        self._supervisor = None
        self._controllers: List[EpuckController] = []
        self._robot_nodes = []

        # State
        self._step_count = 0
        self._goal_positions = None
        self._webots_process = None

    def _create_default_world(self) -> str:
        """Create a default world file with E-puck robots."""
        world_content = self._generate_world_file()
        world_path = "/tmp/cosmos_epuck_world.wbt"
        with open(world_path, 'w') as f:
            f.write(world_content)
        return world_path

    def _generate_world_file(self) -> str:
        """Generate Webots world file content."""
        robots = ""
        for i in range(self._num_agents):
            angle = 2 * np.pi * i / self._num_agents
            x = 0.3 * np.cos(angle)
            y = 0.3 * np.sin(angle)
            robots += f'''
  E-puck {{
    translation {x} {y} 0
    rotation 0 0 1 {angle}
    name "e-puck_{i}"
    controller "extern"
  }}
'''

        world = f'''#VRML_SIM R2023b utf8
WorldInfo {{
  info ["COSMOS E-puck Multi-Robot Environment"]
  basicTimeStep {self._timestep}
}}
Viewpoint {{
  orientation -0.5 0.5 0.7 1.5
  position 0 -2 2
}}
TexturedBackground {{
}}
TexturedBackgroundLight {{
}}
RectangleArena {{
  floorSize {self._arena_size} {self._arena_size}
}}
{robots}
'''
        return world

    def _start_webots(self):
        """Start Webots simulator."""
        webots_exe = os.path.join(WEBOTS_HOME, "webots")

        args = [webots_exe, self._world_file]
        if self._headless:
            args.extend(["--mode=fast", "--no-rendering"])

        self._webots_process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for Webots to start
        time.sleep(3)

    def _init_supervisor(self):
        """Initialize supervisor and robot controllers."""
        if not WEBOTS_CONTROLLER_AVAILABLE:
            raise ImportError("Webots controller module not available")

        self._supervisor = Supervisor()

        # Get robot nodes
        self._robot_nodes = []
        self._controllers = []

        for i in range(self._num_agents):
            node = self._supervisor.getFromDef(f"E-PUCK_{i}")
            if node is None:
                node = self._supervisor.getFromDef(f"e-puck_{i}")
            self._robot_nodes.append(node)

            # Create controller
            controller = EpuckController(self._supervisor, self._timestep)
            self._controllers.append(controller)

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32
        )

    @property
    def action_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._act_dim,),
            dtype=np.float32
        )

    @property
    def share_observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._share_obs_dim,),
            dtype=np.float32
        )

    def get_obs_dim(self) -> int:
        return self._obs_dim

    def get_act_dim(self) -> int:
        return self._act_dim

    def get_share_obs_dim(self) -> int:
        return self._share_obs_dim

    def _get_robot_positions(self) -> np.ndarray:
        """Get positions of all robots."""
        positions = []
        for node in self._robot_nodes:
            if node:
                pos = node.getPosition()
                positions.append([pos[0], pos[1], pos[2]])
            else:
                positions.append([0, 0, 0])
        return np.array(positions)

    def _get_robot_velocities(self) -> np.ndarray:
        """Get velocities of all robots."""
        velocities = []
        for node in self._robot_nodes:
            if node:
                vel = node.getVelocity()
                velocities.append([vel[0], vel[1], vel[2]])
            else:
                velocities.append([0, 0, 0])
        return np.array(velocities)

    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents."""
        positions = self._get_robot_positions()
        obs_list = []

        for i in range(self._num_agents):
            # Proximity sensors
            if i < len(self._controllers):
                proximity = self._controllers[i].get_proximity_readings()
            else:
                proximity = np.zeros(8)

            # Relative position to goal
            if self._goal_positions is not None:
                rel_goal = self._goal_positions[i, :2] - positions[i, :2]
            else:
                rel_goal = np.zeros(2)

            # Relative positions to other agents
            rel_others = []
            for j in range(self._num_agents):
                if i != j:
                    rel_pos = positions[j, :2] - positions[i, :2]
                    rel_others.extend(rel_pos)

            obs = np.concatenate([proximity, rel_goal, rel_others])
            obs_list.append(obs)

        return np.array(obs_list)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        self._step_count = 0

        # Generate random goal positions
        self._goal_positions = np.random.uniform(
            -self._arena_size / 2 * 0.8,
            self._arena_size / 2 * 0.8,
            (self._num_agents, 3)
        )
        self._goal_positions[:, 2] = 0

        # Reset robot positions in simulation
        if self._supervisor:
            for i, node in enumerate(self._robot_nodes):
                if node:
                    angle = 2 * np.pi * i / self._num_agents
                    x = 0.3 * np.cos(angle)
                    y = 0.3 * np.sin(angle)
                    node.getField("translation").setSFVec3f([x, y, 0])
                    node.getField("rotation").setSFRotation([0, 0, 1, angle])

            self._supervisor.step(self._timestep)

        obs = self._get_observations()
        share_obs = np.tile(obs.flatten(), (self._num_agents, 1))

        return obs, share_obs, {}

    def step(self, actions: np.ndarray):
        # Apply actions to robots
        for i, controller in enumerate(self._controllers):
            if i < len(actions):
                controller.set_velocity_from_action(actions[i])

        # Step simulation
        if self._supervisor:
            self._supervisor.step(self._timestep)

        self._step_count += 1

        # Get new state
        obs = self._get_observations()
        positions = self._get_robot_positions()

        # Compute rewards (negative distance to goal)
        rewards = []
        for i in range(self._num_agents):
            dist = np.linalg.norm(positions[i, :2] - self._goal_positions[i, :2])
            reward = -dist
            rewards.append(reward)

        rewards = np.array(rewards).reshape(-1, 1)

        # Compute costs (collisions)
        costs = np.zeros((self._num_agents, 1))
        collision_dist = 0.07  # E-puck radius ~3.5cm
        for i in range(self._num_agents):
            for j in range(i + 1, self._num_agents):
                dist = np.linalg.norm(positions[i, :2] - positions[j, :2])
                if dist < collision_dist * 2:
                    costs[i] += 1
                    costs[j] += 1

        # Check termination
        done = self._step_count >= self._max_steps
        dones = np.full(self._num_agents, done)

        share_obs = np.tile(obs.flatten(), (self._num_agents, 1))
        info_list = [{} for _ in range(self._num_agents)]

        return obs, share_obs, rewards, costs, dones, info_list, done

    def get_constraint_info(self) -> Dict[str, Any]:
        positions = self._get_robot_positions()
        velocities = self._get_robot_velocities()

        return {
            "positions": positions,
            "velocities": velocities,
            "obstacles": np.zeros((0, 3)),
            "goals": self._goal_positions,
        }

    def render(self, mode: str = "human"):
        # Webots handles rendering
        pass

    def close(self):
        if self._supervisor:
            self._supervisor = None

        if self._webots_process:
            self._webots_process.terminate()
            self._webots_process = None


# Simulated E-puck environment (no Webots required)
@ENV_REGISTRY.register("epuck_sim", aliases=["epuck_simple"])
class EpuckSimEnv(BaseMultiAgentEnv):
    """
    Simplified E-puck simulation without Webots.

    Uses simple kinematics for fast training before
    transferring to full Webots simulation.

    Same interface as WebotsEpuckEnv but runs standalone.
    """

    WHEEL_RADIUS = 0.0205
    AXLE_LENGTH = 0.052
    MAX_SPEED = 6.28
    ROBOT_RADIUS = 0.035

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        num_agents: int = 4,
        dt: float = 0.064,
        max_steps: int = 500,
        arena_size: float = 1.0,
        **kwargs
    ):
        if cfg:
            num_agents = cfg.get("num_agents", num_agents)
            dt = cfg.get("dt", dt)
            max_steps = cfg.get("max_steps", max_steps)
            arena_size = cfg.get("arena_size", arena_size)

        self._num_agents = num_agents
        self._dt = dt
        self._max_steps = max_steps
        self._arena_size = arena_size

        self._obs_dim = 8 + 2 + 2 * (num_agents - 1)  # proximity + goal + neighbors
        self._act_dim = 2
        self._share_obs_dim = self._obs_dim * num_agents

        # State - initialize with default positions
        self._positions = np.zeros((num_agents, 3))
        self._orientations = np.zeros(num_agents)
        self._velocities = np.zeros((num_agents, 2))
        self._goal_positions = np.zeros((num_agents, 2))
        self._step_count = 0

        # Initialize to default positions
        self._init_default_positions()

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32
        )

    @property
    def action_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._act_dim,),
            dtype=np.float32
        )

    @property
    def share_observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._share_obs_dim,),
            dtype=np.float32
        )

    def get_obs_dim(self) -> int:
        return self._obs_dim

    def get_act_dim(self) -> int:
        return self._act_dim

    def get_share_obs_dim(self) -> int:
        return self._share_obs_dim

    def _init_default_positions(self):
        """Initialize robots to default grid positions."""
        n = self._num_agents
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = min(0.2, self._arena_size / (grid_size + 1))

        for i in range(n):
            row = i // grid_size
            col = i % grid_size
            x = (col - (grid_size - 1) / 2) * spacing
            y = (row - (grid_size - 1) / 2) * spacing
            self._positions[i] = [x, y, 0]
            self._orientations[i] = np.random.uniform(-np.pi, np.pi)

        # Set random goals
        for i in range(n):
            self._goal_positions[i] = np.random.uniform(
                -self._arena_size / 3, self._arena_size / 3, size=2
            )

    def _simulate_proximity_sensors(self, robot_idx: int) -> np.ndarray:
        """Simulate 8 IR proximity sensors."""
        readings = np.zeros(8)
        pos = self._positions[robot_idx, :2]
        theta = self._orientations[robot_idx]

        # Sensor angles relative to robot heading
        sensor_angles = np.array([
            -150, -90, -45, -15, 15, 45, 90, 150
        ]) * np.pi / 180

        for i, angle in enumerate(sensor_angles):
            sensor_dir = np.array([
                np.cos(theta + angle),
                np.sin(theta + angle)
            ])

            # Check other robots
            for j in range(self._num_agents):
                if i != j:
                    rel_pos = self._positions[j, :2] - pos
                    dist = np.linalg.norm(rel_pos)

                    # Project onto sensor direction
                    proj = np.dot(rel_pos, sensor_dir)
                    if proj > 0 and proj < 0.1:  # In front, within 10cm
                        lateral = np.abs(np.cross(sensor_dir, rel_pos / dist))
                        if lateral < 0.02:  # Within sensor cone
                            readings[i] = max(readings[i], 1 - dist / 0.1)

            # Check walls
            for wall_check in [
                (pos[0] + self._arena_size / 2, sensor_dir[0]),
                (pos[0] - self._arena_size / 2, -sensor_dir[0]),
                (pos[1] + self._arena_size / 2, sensor_dir[1]),
                (pos[1] - self._arena_size / 2, -sensor_dir[1]),
            ]:
                wall_dist, alignment = wall_check
                if alignment > 0.5:
                    dist = abs(wall_dist)
                    if dist < 0.1:
                        readings[i] = max(readings[i], 1 - dist / 0.1)

        return readings

    def _get_observations(self) -> np.ndarray:
        obs_list = []
        for i in range(self._num_agents):
            # Proximity sensors
            proximity = self._simulate_proximity_sensors(i)

            # Relative goal position (in robot frame)
            rel_goal = self._goal_positions[i, :2] - self._positions[i, :2]
            theta = self._orientations[i]
            rot = np.array([[np.cos(-theta), -np.sin(-theta)],
                           [np.sin(-theta), np.cos(-theta)]])
            rel_goal = rot @ rel_goal

            # Relative neighbor positions
            rel_others = []
            for j in range(self._num_agents):
                if i != j:
                    rel_pos = self._positions[j, :2] - self._positions[i, :2]
                    rel_pos = rot @ rel_pos
                    rel_others.extend(rel_pos)

            obs = np.concatenate([proximity, rel_goal, rel_others])
            obs_list.append(obs)

        return np.array(obs_list, dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        self._step_count = 0

        # Initialize positions in circle
        self._positions = np.zeros((self._num_agents, 3))
        self._orientations = np.zeros(self._num_agents)
        self._velocities = np.zeros((self._num_agents, 3))

        for i in range(self._num_agents):
            angle = 2 * np.pi * i / self._num_agents
            self._positions[i, 0] = 0.2 * np.cos(angle)
            self._positions[i, 1] = 0.2 * np.sin(angle)
            self._orientations[i] = angle + np.pi

        # Random goals
        self._goal_positions = np.random.uniform(
            -self._arena_size / 2 * 0.7,
            self._arena_size / 2 * 0.7,
            (self._num_agents, 3)
        )
        self._goal_positions[:, 2] = 0

        obs = self._get_observations()
        share_obs = np.tile(obs.flatten(), (self._num_agents, 1))

        return obs, share_obs, {}

    def step(self, actions: np.ndarray):
        # Apply differential drive kinematics
        for i in range(self._num_agents):
            v = actions[i, 0] * self.MAX_SPEED * self.WHEEL_RADIUS
            omega = actions[i, 1] * self.MAX_SPEED / (self.AXLE_LENGTH / 2)

            # Update pose
            theta = self._orientations[i]
            self._positions[i, 0] += v * np.cos(theta) * self._dt
            self._positions[i, 1] += v * np.sin(theta) * self._dt
            self._orientations[i] += omega * self._dt

            # Clip to arena
            self._positions[i, :2] = np.clip(
                self._positions[i, :2],
                -self._arena_size / 2 + self.ROBOT_RADIUS,
                self._arena_size / 2 - self.ROBOT_RADIUS
            )

            # Update velocity
            self._velocities[i, 0] = v * np.cos(theta)
            self._velocities[i, 1] = v * np.sin(theta)

        self._step_count += 1

        # Observations
        obs = self._get_observations()

        # Rewards
        rewards = []
        for i in range(self._num_agents):
            dist = np.linalg.norm(self._positions[i, :2] - self._goal_positions[i, :2])
            rewards.append(-dist)
        rewards = np.array(rewards).reshape(-1, 1)

        # Costs (collisions)
        costs = np.zeros((self._num_agents, 1))
        for i in range(self._num_agents):
            for j in range(i + 1, self._num_agents):
                dist = np.linalg.norm(self._positions[i, :2] - self._positions[j, :2])
                if dist < self.ROBOT_RADIUS * 2:
                    costs[i] += 1
                    costs[j] += 1

        done = self._step_count >= self._max_steps
        dones = np.full(self._num_agents, done)
        share_obs = np.tile(obs.flatten(), (self._num_agents, 1))
        info_list = [{"collisions": int(costs[i, 0])} for i in range(self._num_agents)]

        return obs, share_obs, rewards, costs, dones, info_list, done

    def get_constraint_info(self) -> Dict[str, Any]:
        return {
            "positions": self._positions.copy(),
            "velocities": self._velocities.copy(),
            "orientations": self._orientations.copy(),
            "obstacles": np.zeros((0, 3)),
            "goals": self._goal_positions.copy(),
        }

    def render(self, mode: str = "human"):
        pass

    def close(self):
        pass
