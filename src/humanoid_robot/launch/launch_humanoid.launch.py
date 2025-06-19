import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    launch_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('humanoid_support_moveit_config'), 'launch'),
            '/launch_moveit.launch.py'])
        )

    launch_gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('humanoid_support_moveit_config'), 'launch'),
            '/launch_gz.launch.py'])
        )
    
    launch_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('humanoid_support_moveit_config'), 'launch'),
            '/launch_controllers.launch.py'])
        )
    launch_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('camera_system'), 'launch'),
            '/spawn_camera.launch.py'])
        )
    return LaunchDescription([
        launch_rviz,
        launch_gz,
        launch_control,
        launch_camera
    ])