"""
ROS2 Launch file for E-puck Safety Navigation experiment.

Launches Webots with the obstacle arena and the safety navigation node.

Usage:
    ros2 launch safety_nav.launch.py
    ros2 launch safety_nav.launch.py use_safety:=false
    ros2 launch safety_nav.launch.py goal_x:=0.3 goal_y:=-0.5
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
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ros2_dir = os.path.dirname(script_dir)
    webots_dir = os.path.dirname(ros2_dir)
    world_file = os.path.join(webots_dir, 'worlds', 'epuck_navigation_ros2.wbt')
    urdf_file = os.path.join(ros2_dir, 'resource', 'epuck_safety.urdf')
    ros2_control_params = os.path.join(
        get_package_share_directory('webots_ros2_epuck'),
        'resource', 'ros2_control.yml')

    use_sim_time = LaunchConfiguration('use_sim_time', default=True)

    # Webots launcher
    webots = WebotsLauncher(world=world_file, ros2_supervisor=True)

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': '<robot name=""><link name=""/></robot>'
        }],
    )

    # Webots controller (extern)
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

    # Safety navigation node
    safety_nav = Node(
        package='webots_ros2_epuck',  # We run standalone, but need a valid package
        executable='safety_nav_node.py',
        name='safety_nav_node',
        output='screen',
        parameters=[{
            'goal_x': LaunchConfiguration('goal_x', default=0.5),
            'goal_y': LaunchConfiguration('goal_y', default=0.5),
            'use_safety': LaunchConfiguration('use_safety', default=True),
            'use_ekf': LaunchConfiguration('use_ekf', default=True),
            'gps_noise_std': 0.04,
            'use_sim_time': use_sim_time,
        }],
    )

    # Wait for controller connection before spawning controllers
    waiting_nodes = WaitForControllerConnection(
        target_driver=epuck_driver,
        nodes_to_start=[diffdrive_spawner, joint_state_spawner]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_safety', default_value='true',
                              description='Enable safety filter'),
        DeclareLaunchArgument('use_ekf', default_value='true',
                              description='Enable EKF state estimation'),
        DeclareLaunchArgument('goal_x', default_value='0.5',
                              description='Goal X coordinate'),
        DeclareLaunchArgument('goal_y', default_value='0.5',
                              description='Goal Y coordinate'),

        webots,
        webots._supervisor,
        robot_state_publisher,
        epuck_driver,
        waiting_nodes,

        # Shutdown when Webots exits
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(
                    event=launch.events.Shutdown()
                )],
            )
        ),
    ])
