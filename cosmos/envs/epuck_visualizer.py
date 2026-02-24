"""
E-puck Visualization for Colab/Jupyter

Provides matplotlib-based visualization for E-puck simulation.
Works in Colab without requiring display server.

Usage:
    from cosmos.envs.epuck_visualizer import EpuckVisualizer

    env = EpuckSimEnv(num_agents=4)
    vis = EpuckVisualizer(env)

    obs, _, _ = env.reset()
    vis.render(env)  # Show frame

    # Or create animation
    vis.create_animation(frames, "output.gif")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from typing import List, Optional, Tuple
import io

# For Colab compatibility
try:
    from IPython.display import HTML, display, clear_output
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False


class EpuckVisualizer:
    """
    Matplotlib-based visualizer for E-puck simulation.

    Features:
    - Real-time plotting in Jupyter/Colab
    - Animation generation (GIF/MP4)
    - Proximity sensor visualization
    - Goal and trajectory display
    """

    # E-puck physical parameters
    ROBOT_RADIUS = 0.035  # meters
    SENSOR_ANGLES = np.array([-150, -90, -45, -15, 15, 45, 90, 150]) * np.pi / 180

    def __init__(
        self,
        env,
        figsize: Tuple[int, int] = (8, 8),
        show_sensors: bool = True,
        show_goals: bool = True,
        show_trails: bool = True,
        trail_length: int = 50
    ):
        """
        Args:
            env: EpuckSimEnv instance
            figsize: Figure size in inches
            show_sensors: Show proximity sensor rays
            show_goals: Show goal positions
            show_trails: Show robot trajectories
            trail_length: Number of past positions to show
        """
        self.env = env
        self.figsize = figsize
        self.show_sensors = show_sensors
        self.show_goals = show_goals
        self.show_trails = show_trails
        self.trail_length = trail_length

        # Colors for agents
        self.colors = plt.cm.tab10.colors

        # Trail history
        self.trails = [[] for _ in range(env.num_agents)]

        # Setup figure
        self.fig = None
        self.ax = None

    def _setup_figure(self):
        """Create figure and axis."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlim(-self.env._arena_size / 2 - 0.1,
                         self.env._arena_size / 2 + 0.1)
        self.ax.set_ylim(-self.env._arena_size / 2 - 0.1,
                         self.env._arena_size / 2 + 0.1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

    def render(self, env=None, return_array: bool = False):
        """
        Render current state.

        Args:
            env: Environment (uses self.env if None)
            return_array: Return image as numpy array

        Returns:
            If return_array: numpy array of shape (H, W, 3)
            Else: None (displays in notebook)
        """
        env = env or self.env

        if self.fig is None:
            self._setup_figure()

        self.ax.clear()

        # Set limits and style
        arena = env._arena_size / 2
        self.ax.set_xlim(-arena - 0.1, arena + 0.1)
        self.ax.set_ylim(-arena - 0.1, arena + 0.1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        # Draw arena boundary
        rect = patches.Rectangle(
            (-arena, -arena), env._arena_size, env._arena_size,
            linewidth=2, edgecolor='black', facecolor='white'
        )
        self.ax.add_patch(rect)

        # Get state
        positions = env._positions
        orientations = env._orientations
        goals = env._goal_positions

        # Update trails
        for i in range(env.num_agents):
            self.trails[i].append(positions[i, :2].copy())
            if len(self.trails[i]) > self.trail_length:
                self.trails[i].pop(0)

        # Draw trails
        if self.show_trails:
            for i, trail in enumerate(self.trails):
                if len(trail) > 1:
                    trail_arr = np.array(trail)
                    alphas = np.linspace(0.1, 0.5, len(trail))
                    for j in range(len(trail) - 1):
                        self.ax.plot(
                            [trail_arr[j, 0], trail_arr[j + 1, 0]],
                            [trail_arr[j, 1], trail_arr[j + 1, 1]],
                            color=self.colors[i % len(self.colors)],
                            alpha=alphas[j],
                            linewidth=1
                        )

        # Draw goals
        if self.show_goals and goals is not None:
            for i in range(env.num_agents):
                self.ax.plot(
                    goals[i, 0], goals[i, 1],
                    marker='*', markersize=15,
                    color=self.colors[i % len(self.colors)],
                    alpha=0.5
                )
                # Line to goal
                self.ax.plot(
                    [positions[i, 0], goals[i, 0]],
                    [positions[i, 1], goals[i, 1]],
                    '--', color=self.colors[i % len(self.colors)],
                    alpha=0.3, linewidth=1
                )

        # Draw robots
        for i in range(env.num_agents):
            color = self.colors[i % len(self.colors)]
            pos = positions[i, :2]
            theta = orientations[i]

            # Robot body (circle)
            circle = patches.Circle(
                pos, self.ROBOT_RADIUS,
                facecolor=color, edgecolor='black',
                linewidth=1.5, alpha=0.8
            )
            self.ax.add_patch(circle)

            # Direction indicator
            dx = self.ROBOT_RADIUS * 1.2 * np.cos(theta)
            dy = self.ROBOT_RADIUS * 1.2 * np.sin(theta)
            self.ax.arrow(
                pos[0], pos[1], dx, dy,
                head_width=0.015, head_length=0.01,
                fc='black', ec='black'
            )

            # Agent label
            self.ax.text(
                pos[0], pos[1] + self.ROBOT_RADIUS + 0.03,
                f'R{i}', ha='center', va='bottom', fontsize=8
            )

            # Proximity sensors
            if self.show_sensors:
                readings = env._simulate_proximity_sensors(i)
                for j, angle in enumerate(self.SENSOR_ANGLES):
                    sensor_dir = np.array([
                        np.cos(theta + angle),
                        np.sin(theta + angle)
                    ])
                    # Sensor range visualization
                    sensor_len = 0.08 * (1 - readings[j])  # Shorter if closer
                    end = pos + sensor_dir * sensor_len
                    intensity = readings[j]
                    self.ax.plot(
                        [pos[0], end[0]], [pos[1], end[1]],
                        color='red' if intensity > 0.5 else 'green',
                        alpha=0.3 + 0.4 * intensity,
                        linewidth=1
                    )

        # Title with step info
        self.ax.set_title(
            f'E-puck Simulation | Step: {env._step_count} | '
            f'Agents: {env.num_agents}',
            fontsize=12
        )

        if return_array:
            # Convert to numpy array
            self.fig.canvas.draw()
            buf = self.fig.canvas.buffer_rgba()
            img = np.asarray(buf)[:, :, :3]
            return img
        else:
            if IN_NOTEBOOK:
                clear_output(wait=True)
                display(self.fig)
            else:
                plt.pause(0.01)

    def reset(self):
        """Reset trails and figure."""
        self.trails = [[] for _ in range(self.env.num_agents)]
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = None
        self.ax = None

    def create_animation(
        self,
        frames: List[dict],
        output_path: str = "epuck_animation.gif",
        fps: int = 10
    ):
        """
        Create animation from recorded frames.

        Args:
            frames: List of frame data dicts with 'positions', 'orientations', 'goals'
            output_path: Output file path (.gif or .mp4)
            fps: Frames per second
        """
        self._setup_figure()

        def update(frame_idx):
            frame = frames[frame_idx]
            self.ax.clear()

            arena = self.env._arena_size / 2
            self.ax.set_xlim(-arena - 0.1, arena + 0.1)
            self.ax.set_ylim(-arena - 0.1, arena + 0.1)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)

            # Arena
            rect = patches.Rectangle(
                (-arena, -arena), self.env._arena_size, self.env._arena_size,
                linewidth=2, edgecolor='black', facecolor='white'
            )
            self.ax.add_patch(rect)

            positions = frame['positions']
            orientations = frame['orientations']
            goals = frame.get('goals')

            # Goals
            if goals is not None:
                for i in range(len(positions)):
                    self.ax.plot(
                        goals[i, 0], goals[i, 1],
                        marker='*', markersize=12,
                        color=self.colors[i % len(self.colors)],
                        alpha=0.5
                    )

            # Robots
            for i in range(len(positions)):
                color = self.colors[i % len(self.colors)]
                pos = positions[i, :2]
                theta = orientations[i]

                circle = patches.Circle(
                    pos, self.ROBOT_RADIUS,
                    facecolor=color, edgecolor='black',
                    linewidth=1.5, alpha=0.8
                )
                self.ax.add_patch(circle)

                dx = self.ROBOT_RADIUS * 1.2 * np.cos(theta)
                dy = self.ROBOT_RADIUS * 1.2 * np.sin(theta)
                self.ax.arrow(
                    pos[0], pos[1], dx, dy,
                    head_width=0.015, head_length=0.01,
                    fc='black', ec='black'
                )

            self.ax.set_title(f'Step: {frame_idx}')
            return []

        anim = FuncAnimation(
            self.fig, update,
            frames=len(frames),
            interval=1000 // fps,
            blit=True
        )

        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        else:
            anim.save(output_path, writer='ffmpeg', fps=fps)

        plt.close(self.fig)
        print(f"Animation saved to: {output_path}")

        return output_path

    def close(self):
        """Close figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


def run_episode_with_visualization(
    env,
    policy=None,
    max_steps: int = 200,
    render_interval: int = 5,
    save_animation: bool = True,
    output_path: str = "epuck_episode.gif"
):
    """
    Run episode with visualization (Colab compatible).

    Args:
        env: EpuckSimEnv instance
        policy: Policy function (obs -> actions). If None, uses random.
        max_steps: Maximum steps
        render_interval: Steps between renders
        save_animation: Save animation to file
        output_path: Animation output path

    Returns:
        Dict with episode statistics
    """
    vis = EpuckVisualizer(env, show_sensors=True)
    frames = []

    obs, share_obs, info = env.reset()
    total_reward = 0
    total_cost = 0

    for step in range(max_steps):
        # Get action
        if policy is not None:
            actions, _ = policy(obs)
        else:
            actions = np.random.uniform(-0.5, 0.5, (env.num_agents, env.get_act_dim()))

        # Step
        next_obs, next_share, rewards, costs, dones, infos, truncated = env.step(actions)

        total_reward += rewards.sum()
        total_cost += costs.sum()

        # Record frame
        frames.append({
            'positions': env._positions.copy(),
            'orientations': env._orientations.copy(),
            'goals': env._goal_positions.copy() if env._goal_positions is not None else None,
        })

        # Render
        if step % render_interval == 0:
            vis.render(env)

        obs = next_obs

        if dones.all() or truncated:
            break

    # Save animation
    if save_animation and len(frames) > 0:
        vis.create_animation(frames, output_path)

    vis.close()

    return {
        'total_reward': total_reward,
        'total_cost': total_cost,
        'steps': step + 1,
        'animation_path': output_path if save_animation else None,
    }
