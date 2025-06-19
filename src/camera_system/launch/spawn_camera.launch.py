from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():
   
    camera_description_path = os.path.join(
        get_package_share_directory('camera_system'),
        'urdf',
        'indoor_camera.xacro'
    )
    
    doc = xacro.process_file(camera_description_path)
    robot_description = doc.toxml()

    camera_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='camera_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
        remappings=[
            ('/robot_description', '/camera_description'),
        ]
    )

    spawn_camera = Node(
        package='ros_gz_sim',  
        executable='create',
        output='screen',
        arguments=[
            '-topic', '/camera_description',
            '-name', 'camera',
            '-x', '3.5',
            '-y', '0.0',
            '-z', '2'
        ]
    )


    # Bridge between ROS 2 and Ignition
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='camera_bridge',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '/world/empty/model/camera/link/hhcamera/sensor/camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image',
        ],
        remappings=[
            ('/world/empty/model/camera/link/hhcamera/sensor/camera_sensor/image', '/camera/image_raw')
        ]
    )



    return LaunchDescription([
        camera_state_publisher,
        spawn_camera,
        bridge  
    ])
