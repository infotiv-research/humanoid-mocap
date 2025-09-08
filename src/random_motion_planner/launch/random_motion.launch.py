from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'motion_filename',
            default_value="random",
            description='Path to motion file'
        ),
        DeclareLaunchArgument(
            'output_dir',
            default_value='DATASET/motion_data',
            description='Directory to save output motions'
        ),
        Node(
            package='random_motion_planner',
            executable='motion_planner',
            name='random_motion_planner',
            output='screen',
            emulate_tty=True,
            parameters=[{'motion_filename': LaunchConfiguration('motion_filename')},
                        {'output_dir': LaunchConfiguration('output_dir')}
                        ]

        )
    ])
