   
import os
import random

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_humanoid_robot = get_package_share_directory('humanoid_robot')
    world_file_name = 'empty.sdf'
    world = os.path.join(pkg_humanoid_robot, 'worlds', world_file_name)

    ld = LaunchDescription()

    # Start Gazebo sim
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory('ros_gz_sim'),
                    "launch",
                    "gz_sim.launch.py",
                )
            ]
        ),
        launch_arguments={"gz_args": [" -r -v 4 ", world]}.items(),
    )

    # Spawn robot from /robot_description topic
    gz_spawn = Node(
        package="ros_gz_sim",
        executable="create",
        output="log",
        arguments=[
            "-topic",
            "robot_description",
            '-x', str(random.uniform(-1.0, 1.0)),
            '-y', str(random.uniform(-2.0,2.0)),
            '-z', '0.0',
            "--ros-args",
            # "--log-level",
            # log_level,
            # "-x", "0",
            # "-y", "0",
            # "-z", "1.4",
        ],
        parameters=[{"use_sim_time": True}],
    )
    
    # Start ros_gz_bridge (clock -> ROS 2)
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="log",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "--ros-args",
            # "--log-level",
            # log_level,
        ],
        parameters=[{"use_sim_time": True}],
    )

    ld.add_action(gz_sim)
    ld.add_action(gz_spawn)
    ld.add_action(gz_bridge)

    return ld