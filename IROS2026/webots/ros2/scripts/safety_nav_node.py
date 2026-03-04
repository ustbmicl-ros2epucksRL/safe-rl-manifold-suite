#!/usr/bin/env python3
"""
ROS2 Safety Navigation Node for E-puck

Demonstrates the manifold safety filter as a ROS2 component.
Subscribes to GPS/odom for position, applies DistanceFilter,
and publishes safe velocity commands.

Topics:
    Subscribed:
        /odom (nav_msgs/Odometry) - Robot odometry (from ros2_control)
        /gps (geometry_msgs/PointStamped) - GPS position (from Webots)
    Published:
        /cmd_vel (geometry_msgs/TwistStamped) - Safe velocity commands
        /safety/status (std_msgs/String) - Safety filter status
"""

import os
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import TwistStamped, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

# Add IROS2026 src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IROS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, os.path.join(IROS_DIR, 'src'))

from safety.distance_filter import DistanceFilter
from ekf import StandardEKF, EKFConfig

# E-puck physical parameters
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
ROBOT_RADIUS = 0.035
MAX_LINEAR_SPEED = WHEEL_RADIUS * 6.28  # ~0.129 m/s
OBSTACLE_RADIUS = 0.10
GOAL_THRESHOLD = 0.08

# Known obstacle positions (matching world file)
OBSTACLES = [
    np.array([-0.4, 0.3]),
    np.array([0.3, 0.5]),
    np.array([-0.3, -0.4]),
    np.array([0.5, -0.3]),
    np.array([0.0, 0.05]),
    np.array([-0.15, 0.55]),
]


class SafetyNavNode(Node):
    """ROS2 node integrating DistanceFilter safety layer with EKF."""

    def __init__(self):
        super().__init__('safety_nav_node')

        # Parameters
        self.declare_parameter('goal_x', 0.5)
        self.declare_parameter('goal_y', 0.5)
        self.declare_parameter('use_safety', True)
        self.declare_parameter('use_ekf', True)
        self.declare_parameter('gps_noise_std', 0.04)

        self.goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
        ])
        self.use_safety = self.get_parameter('use_safety').value
        self.use_ekf = self.get_parameter('use_ekf').value
        gps_noise = self.get_parameter('gps_noise_std').value

        self.get_logger().info(
            f'SafetyNavNode: goal=({self.goal[0]:.2f}, {self.goal[1]:.2f}), '
            f'safety={self.use_safety}, ekf={self.use_ekf}'
        )

        # State
        self.position = None
        self.heading = 0.0
        self.nav_pos = None
        self.nav_heading = 0.0
        self.prev_action = np.array([0.0, 0.0])
        self.step_count = 0
        self.collision_count = 0

        # Safety filter
        self.safety_filter = None
        if self.use_safety:
            self.safety_filter = DistanceFilter(
                hazard_radius=OBSTACLE_RADIUS + ROBOT_RADIUS,
                danger_radius=0.12,
                stop_radius=0.08,
                max_forward_vel=MAX_LINEAR_SPEED,
                max_angular_vel=2.0,
                lambda_calib=0.0,
            )
            self.safety_filter.reset(OBSTACLES)
            self.get_logger().info('DistanceFilter safety layer initialized')

        # EKF
        self.ekf = None
        if self.use_ekf:
            dt = 0.064  # basicTimeStep
            self.ekf = StandardEKF(config=EKFConfig(
                dt=dt, sigma_lat=gps_noise, sigma_up=0.05))
            self.get_logger().info('StandardEKF initialized')

        # Subscribers
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, sensor_qos)

        self.gps_sub = self.create_subscription(
            PointStamped, '/gps', self._gps_callback, sensor_qos)

        # Publishers
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/safety/status', 10)

        # Control loop at ~15 Hz
        self.timer = self.create_timer(0.064, self._control_loop)

        self.get_logger().info('SafetyNavNode ready')

    def _odom_callback(self, msg: Odometry):
        """Extract heading from odometry."""
        orient = msg.pose.pose.orientation
        siny_cosp = 2 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1 - 2 * (orient.y * orient.y + orient.z * orient.z)
        self.heading = np.arctan2(siny_cosp, cosy_cosp)

        # Use odom position if GPS not available
        if self.position is None:
            self.position = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
            ])

    def _gps_callback(self, msg: PointStamped):
        """Update position from GPS (with noise)."""
        self.position = np.array([msg.point.x, msg.point.y])

    def _control_loop(self):
        """Main control loop: P-controller + safety filter."""
        if self.position is None:
            return

        self.step_count += 1

        # Apply EKF if enabled
        if self.ekf is not None:
            if self.step_count == 1:
                self.ekf.reset(np.array([
                    self.position[0], self.position[1], self.heading]))
            else:
                self.ekf.predict(self.prev_action, dt=0.064)
                self.ekf.update(np.array([
                    self.position[0], self.position[1], self.heading]))

            est = self.ekf.get_position()
            self.nav_pos = est[:2]
            self.nav_heading = est[2]
        else:
            self.nav_pos = self.position.copy()
            self.nav_heading = self.heading

        # Check goal reached
        dist_to_goal = np.linalg.norm(self.nav_pos - self.goal)
        if dist_to_goal < GOAL_THRESHOLD:
            self._publish_cmd(0.0, 0.0)
            self._publish_status('GOAL_REACHED')
            return

        # P-controller with obstacle avoidance
        v, omega = self._p_controller(self.nav_pos, self.nav_heading, self.goal)

        # Apply safety filter
        if self.safety_filter is not None:
            robot_pose = np.array([
                self.nav_pos[0], self.nav_pos[1], self.nav_heading])
            result = self.safety_filter.project(np.array([v, omega]), robot_pose)
            v_safe = float(result.action_safe[0])
            omega_safe = float(result.action_safe[1])
            correction = np.linalg.norm(result.correction)
        else:
            v_safe, omega_safe = v, omega
            correction = 0.0

        self.prev_action = np.array([v_safe, omega_safe])
        self._publish_cmd(v_safe, omega_safe)

        # Status
        if self.step_count % 50 == 0:
            self._publish_status(
                f'step={self.step_count} d_goal={dist_to_goal:.3f} '
                f'correction={correction:.4f}')

    def _p_controller(self, position, heading, goal):
        """Simple P-controller with obstacle avoidance."""
        error = goal - position
        dist = np.linalg.norm(error)
        if dist < GOAL_THRESHOLD:
            return 0.0, 0.0

        desired_heading = np.arctan2(error[1], error[0])

        # Weak obstacle avoidance (safety filter handles hard constraints)
        goal_dir = error / (dist + 1e-8)
        avoidance_radius = 0.15
        closest_gap = float('inf')
        closest_obs = None

        for obs in OBSTACLES:
            to_obs = obs - position
            center_dist = np.linalg.norm(to_obs)
            gap = center_dist - OBSTACLE_RADIUS
            if gap < avoidance_radius and center_dist > 1e-4:
                obs_dir = to_obs / center_dist
                if np.dot(obs_dir, goal_dir) > -0.2 and gap < closest_gap:
                    closest_gap = gap
                    closest_obs = obs

        if closest_obs is not None and closest_gap < avoidance_radius:
            to_obs = closest_obs - position
            center_dist = np.linalg.norm(to_obs) + 1e-6
            t1 = np.array([-to_obs[1], to_obs[0]]) / center_dist
            t2 = np.array([to_obs[1], -to_obs[0]]) / center_dist
            avoid_dir = t1 if np.dot(t1, goal_dir) >= np.dot(t2, goal_dir) else t2
            blend = np.clip(1.0 - closest_gap / avoidance_radius, 0.0, 1.0) ** 0.5
            combined = (1.0 - blend) * goal_dir + blend * avoid_dir
            n = np.linalg.norm(combined)
            if n > 1e-6:
                combined /= n
            desired_heading = np.arctan2(combined[1], combined[0])

        heading_error = desired_heading - heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        v = min(0.4 * dist, MAX_LINEAR_SPEED)
        if abs(heading_error) > 0.5:
            v *= 0.2
        omega = 3.5 * heading_error

        return v, omega

    def _publish_cmd(self, v, omega):
        """Publish velocity command."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(np.clip(v, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))
        msg.twist.angular.z = float(np.clip(omega, -4.0, 4.0))
        self.cmd_pub.publish(msg)

    def _publish_status(self, status_str):
        """Publish safety status."""
        msg = String()
        msg.data = status_str
        self.status_pub.publish(msg)
        self.get_logger().info(status_str)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
