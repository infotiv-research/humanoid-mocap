import os
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("human_support", package_name="humanoid_support_moveit_config").to_moveit_configs()
    return generate_spawn_controllers_launch(moveit_config)


def generate_spawn_controllers_launch(moveit_config):
    controller_names = moveit_config.trajectory_execution.get(
        "moveit_simple_controller_manager", {}
    ).get("controller_names", [])
    ld = LaunchDescription()
    for controller in controller_names + ["joint_state_broadcaster"]:
        ld.add_action(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller],
                output="screen",
                parameters=[{"use_sim_time": True}],
                remappings=[
                ("/controller_manager/robot_description", "/robot_description"),
            ],
            )
        )
    return ld