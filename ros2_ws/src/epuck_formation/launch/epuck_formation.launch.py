"""
E-puck Formation Control Launch File

Launches Webots simulation with multiple E-puck robots and COSMOS safety controller.

Usage:
    ros2 launch epuck_formation epuck_formation.launch.py
    ros2 launch epuck_formation epuck_formation.launch.py num_robots:=6
"""

import os
import pathlib
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_dir = get_package_share_directory('epuck_formation')

    # Launch arguments
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='4',
        description='Number of E-puck robots'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([package_dir, 'worlds', 'epuck_arena.wbt']),
        description='Webots world file'
    )

    use_safety_arg = DeclareLaunchArgument(
        'use_safety',
        default_value='true',
        description='Enable COSMOS safety filter'
    )

    policy_model_arg = DeclareLaunchArgument(
        'policy_model',
        default_value='',
        description='Path to trained MAPPO policy model'
    )

    # Webots launcher (from webots_ros2)
    webots = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('webots_ros2_driver'),
            '/launch/robot_launch.py'
        ]),
        launch_arguments={
            'world': LaunchConfiguration('world'),
        }.items()
    )

    # Formation controller node
    formation_controller = Node(
        package='epuck_formation',
        executable='formation_controller.py',
        name='formation_controller',
        output='screen',
        parameters=[{
            'num_robots': LaunchConfiguration('num_robots'),
            'arena_size': 1.0,
            'formation_type': 'square',
            'dt': 0.064,
        }]
    )

    # Safety filter node (COSMOS/CBF)
    safety_filter = Node(
        package='epuck_formation',
        executable='safety_filter_node.py',
        name='safety_filter',
        output='screen',
        parameters=[{
            'num_robots': LaunchConfiguration('num_robots'),
            'safety_margin': 0.08,
            'cbf_gamma': 1.0,
        }],
        condition=LaunchConfiguration('use_safety')
    )

    # MAPPO policy node (if model provided)
    mappo_policy = Node(
        package='epuck_formation',
        executable='mappo_policy_node.py',
        name='mappo_policy',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('policy_model'),
            'num_robots': LaunchConfiguration('num_robots'),
        }]
    )

    return LaunchDescription([
        num_robots_arg,
        world_arg,
        use_safety_arg,
        policy_model_arg,
        webots,
        formation_controller,
        safety_filter,
        mappo_policy,
    ])
