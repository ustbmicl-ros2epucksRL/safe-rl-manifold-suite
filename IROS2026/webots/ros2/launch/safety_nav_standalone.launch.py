"""
Standalone ROS2 Launch file for E-puck Safety Navigation.

Does not require package installation. Uses absolute paths.

Usage (from IROS2026 directory):
    ros2 launch webots/ros2/launch/safety_nav_standalone.launch.py
    ros2 launch webots/ros2/launch/safety_nav_standalone.launch.py use_safety:=false
"""

import os
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.wait_for_controller_connection import WaitForControllerConnection


def generate_launch_description():
    # Resolve absolute paths
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    ros2_dir = os.path.dirname(launch_dir)
    webots_dir = os.path.dirname(ros2_dir)

    world_file = os.path.join(webots_dir, 'worlds', 'epuck_navigation_ros2.wbt')
    urdf_file = os.path.join(ros2_dir, 'resource', 'epuck_safety.urdf')
    ros2_control_params = os.path.join(
        get_package_share_directory('webots_ros2_epuck'),
        'resource', 'ros2_control.yml')

    use_sim_time = LaunchConfiguration('use_sim_time', default=True)

    # Webots simulator
    webots = WebotsLauncher(world=world_file, ros2_supervisor=True)

    # Robot state publisher (minimal)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': '<robot name=""><link name=""/></robot>'
        }],
    )

    # TF
    footprint_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
    )

    # Webots controller (extern) with custom URDF
    mappings = [
        ('/diffdrive_controller/cmd_vel', '/cmd_vel'),
        ('/diffdrive_controller/odom', '/odom'),
    ]
    epuck_driver = WebotsController(
        robot_name='e-puck',
        parameters=[
            {'robot_description': urdf_file,
             'use_sim_time': use_sim_time,
             'set_robot_state_publisher': True},
            ros2_control_params
        ],
        remappings=mappings,
        respawn=True
    )

    # ROS control spawners
    controller_manager_timeout = ['--controller-manager-timeout', '50']
    diffdrive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['diffdrive_controller'] + controller_manager_timeout,
        parameters=[{'use_sim_time': use_sim_time}],
    )
    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['joint_state_broadcaster'] + controller_manager_timeout,
        parameters=[{'use_sim_time': use_sim_time}],
    )

    waiting_nodes = WaitForControllerConnection(
        target_driver=epuck_driver,
        nodes_to_start=[diffdrive_spawner, joint_state_spawner]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_safety', default_value='true'),
        DeclareLaunchArgument('use_ekf', default_value='true'),
        DeclareLaunchArgument('goal_x', default_value='0.5'),
        DeclareLaunchArgument('goal_y', default_value='0.5'),

        webots,
        webots._supervisor,
        robot_state_publisher,
        footprint_publisher,
        epuck_driver,
        waiting_nodes,

        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(
                    event=launch.events.Shutdown()
                )],
            )
        ),
    ])
