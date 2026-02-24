#!/usr/bin/env python3
"""
E-puck Formation Controller ROS2 Node

Manages multi-robot formation control with COSMOS safety guarantees.

Topics:
    Subscribed:
        /epuck{i}/proximity - IR sensor readings
        /epuck{i}/odom - Robot odometry
        /formation/goal - Formation center goal

    Published:
        /epuck{i}/cmd_vel - Velocity commands
        /formation/status - Formation status

Services:
        /formation/set_shape - Change formation shape
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from std_msgs.msg import Float32MultiArray, String

import numpy as np
import sys
import os

# Add COSMOS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cosmos.safety.cosmos_filter import CBFFilter


class FormationController(Node):
    """
    Multi-robot formation controller with safety guarantees.
    """

    # E-puck physical parameters
    WHEEL_RADIUS = 0.0205  # m
    AXLE_LENGTH = 0.052    # m
    ROBOT_RADIUS = 0.035   # m
    MAX_SPEED = 0.12       # m/s

    def __init__(self):
        super().__init__('formation_controller')

        # Parameters
        self.declare_parameter('num_robots', 4)
        self.declare_parameter('arena_size', 1.0)
        self.declare_parameter('formation_type', 'square')
        self.declare_parameter('dt', 0.064)
        self.declare_parameter('safety_margin', 0.08)

        self.num_robots = self.get_parameter('num_robots').value
        self.arena_size = self.get_parameter('arena_size').value
        self.formation_type = self.get_parameter('formation_type').value
        self.dt = self.get_parameter('dt').value
        self.safety_margin = self.get_parameter('safety_margin').value

        self.get_logger().info(f'Formation Controller: {self.num_robots} robots')

        # Robot states
        self.positions = np.zeros((self.num_robots, 3))  # x, y, theta
        self.velocities = np.zeros((self.num_robots, 2))  # v, omega
        self.proximity = np.zeros((self.num_robots, 8))   # 8 IR sensors

        # Formation targets
        self.formation_center = np.array([0.0, 0.0])
        self.formation_offsets = self._compute_formation_offsets()

        # Safety filter
        self.cbf = CBFFilter(
            env_cfg={'arena_size': self.arena_size, 'num_agents': self.num_robots},
            safety_cfg={'safety_margin': self.safety_margin}
        )

        # QoS for sensor data
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # Subscribers
        self.odom_subs = []
        self.prox_subs = []
        for i in range(self.num_robots):
            # Odometry
            odom_sub = self.create_subscription(
                Odometry,
                f'/epuck{i}/odom',
                lambda msg, idx=i: self._odom_callback(msg, idx),
                sensor_qos
            )
            self.odom_subs.append(odom_sub)

            # Proximity sensors
            prox_sub = self.create_subscription(
                Float32MultiArray,
                f'/epuck{i}/proximity',
                lambda msg, idx=i: self._proximity_callback(msg, idx),
                sensor_qos
            )
            self.prox_subs.append(prox_sub)

        # Goal subscriber
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/formation/goal',
            self._goal_callback,
            10
        )

        # Publishers
        self.cmd_pubs = []
        for i in range(self.num_robots):
            pub = self.create_publisher(Twist, f'/epuck{i}/cmd_vel', 10)
            self.cmd_pubs.append(pub)

        self.status_pub = self.create_publisher(String, '/formation/status', 10)

        # Control loop timer
        self.timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info('Formation Controller initialized')

    def _compute_formation_offsets(self) -> np.ndarray:
        """Compute formation shape offsets."""
        n = self.num_robots
        offsets = np.zeros((n, 2))

        if self.formation_type == 'square':
            # Square formation
            side = int(np.ceil(np.sqrt(n)))
            spacing = 0.15
            for i in range(n):
                row = i // side
                col = i % side
                offsets[i] = [
                    (col - (side - 1) / 2) * spacing,
                    (row - (side - 1) / 2) * spacing
                ]

        elif self.formation_type == 'line':
            spacing = 0.12
            for i in range(n):
                offsets[i] = [(i - (n - 1) / 2) * spacing, 0]

        elif self.formation_type == 'circle':
            radius = 0.15
            for i in range(n):
                angle = 2 * np.pi * i / n
                offsets[i] = [radius * np.cos(angle), radius * np.sin(angle)]

        elif self.formation_type == 'v':
            spacing = 0.12
            for i in range(n):
                if i == 0:
                    offsets[i] = [0, 0]
                else:
                    side = 1 if i % 2 == 1 else -1
                    row = (i + 1) // 2
                    offsets[i] = [side * row * spacing * 0.7, -row * spacing]

        return offsets

    def _odom_callback(self, msg: Odometry, robot_idx: int):
        """Handle odometry update."""
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation

        # Extract yaw from quaternion
        siny_cosp = 2 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1 - 2 * (orient.y * orient.y + orient.z * orient.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        self.positions[robot_idx] = [pos.x, pos.y, yaw]

        # Velocity
        self.velocities[robot_idx] = [
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        ]

    def _proximity_callback(self, msg: Float32MultiArray, robot_idx: int):
        """Handle proximity sensor update."""
        if len(msg.data) >= 8:
            self.proximity[robot_idx] = np.array(msg.data[:8])

    def _goal_callback(self, msg: PoseStamped):
        """Handle formation goal update."""
        self.formation_center = np.array([
            msg.pose.position.x,
            msg.pose.position.y
        ])
        self.get_logger().info(f'New formation goal: {self.formation_center}')

    def _control_loop(self):
        """Main control loop."""
        # Compute target positions for each robot
        targets = self.formation_center + self.formation_offsets

        # Compute nominal actions (simple proportional control)
        actions = np.zeros((self.num_robots, 2))
        for i in range(self.num_robots):
            pos = self.positions[i, :2]
            theta = self.positions[i, 2]
            target = targets[i]

            # Error in world frame
            error = target - pos
            dist = np.linalg.norm(error)

            if dist > 0.01:
                # Desired heading
                desired_heading = np.arctan2(error[1], error[0])

                # Heading error
                heading_error = desired_heading - theta
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

                # P control
                v = min(0.5 * dist, self.MAX_SPEED)
                omega = 2.0 * heading_error

                actions[i] = [v, omega]

        # Apply CBF safety filter
        constraint_info = {
            'positions': self.positions[:, :2],
            'velocities': self.velocities,
            'orientations': self.positions[:, 2],
        }

        try:
            safe_actions = self.cbf.project(actions, constraint_info)
        except Exception as e:
            self.get_logger().warn(f'CBF failed: {e}, using nominal actions')
            safe_actions = actions

        # Publish velocity commands
        for i in range(self.num_robots):
            cmd = Twist()
            cmd.linear.x = float(np.clip(safe_actions[i, 0], -self.MAX_SPEED, self.MAX_SPEED))
            cmd.angular.z = float(np.clip(safe_actions[i, 1], -2.0, 2.0))
            self.cmd_pubs[i].publish(cmd)

        # Publish status
        formation_error = np.mean([
            np.linalg.norm(self.positions[i, :2] - targets[i])
            for i in range(self.num_robots)
        ])

        status = String()
        status.data = f'formation_error: {formation_error:.3f}'
        self.status_pub.publish(status)


def main(args=None):
    rclpy.init(args=args)
    node = FormationController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
