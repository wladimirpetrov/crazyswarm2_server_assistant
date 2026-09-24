#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_rviz = LaunchConfiguration("use_rviz")
    output = LaunchConfiguration("output")

    bernstein_trajectory_player_online = Node(
        package="crazyflie",
        executable="bernstein_trajectory_player_online_mpp.py",
        name="bernstein_trajectory_player_online",
        output=output,
        emulate_tty=True,
    )

    experiment_sos_mpp_over_obs = Node(
        package="crazyflie",
        executable="experiment_sos_mpp_over_obs.py",
        name="experiment_sos_mpp_over_obs",
        output=output,
        emulate_tty=True,
    )

    lab_floor_iowa_dynamic = Node(
        package="crazyflie",
        executable="lab_floor_iowa_dynamic.py",
        name="lab_floor_iowa_dynamic",
        output=output,
        emulate_tty=True,
    )

    plotting_drones = Node(
        package="crazyflie",
        executable="plotting_drones.py",
        name="plotting_drones",
        output=output,
        emulate_tty=True,
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output=output,
        condition=IfCondition(use_rviz),
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Start RViz2 together with the Crazyflie visualization/planning nodes.",
            ),
            DeclareLaunchArgument(
                "output",
                default_value="screen",
                description="Node output mode.",
            ),

            bernstein_trajectory_player_online,
            experiment_sos_mpp_over_obs,
            lab_floor_iowa_dynamic,
            plotting_drones,

            TimerAction(
                period=1.0,
                actions=[rviz2],
            ),
        ]
    )
