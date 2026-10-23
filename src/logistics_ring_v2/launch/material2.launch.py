"""Compatibility launcher for the split material nodes and unified gate."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    unified = os.path.join(get_package_share_directory('logistics_ring_v2'),
                           'launch', 'vision_all.launch.py')
    def include(context):
        material_mode = LaunchConfiguration('start_mode').perform(context)
        return [IncludeLaunchDescription(PythonLaunchDescriptionSource(unified), launch_arguments={
            'display': LaunchConfiguration('display'),
            'camera': LaunchConfiguration('camera'),
            'camera_device': LaunchConfiguration('camera_device'),
            'start_mode': 'raw',
            'start_material_mode': material_mode,
        }.items())]

    return LaunchDescription([
        DeclareLaunchArgument('display', default_value='true'),
        DeclareLaunchArgument('camera', default_value='true'),
        DeclareLaunchArgument('camera_device', default_value='/dev/video0'),
        DeclareLaunchArgument('start_mode', default_value='all',
                              description='all / idle / track <color>; locate removed'),
        OpaqueFunction(function=include),
    ])
