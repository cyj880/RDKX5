"""视觉系统一键启动: 相机采集 → 去畸变 → 物料识别 + 圆环识别 → 数字识别。

用法:
    ros2 launch logistics_vision_capture vision.launch.py
    ros2 launch logistics_vision_capture vision.launch.py display:=false camera_device:=/dev/video1

物料/圆环/数字节点启动后均为 idle, 等命令(终端5 或主控):
    物料: locate <色1><色2><色3> / track <色> / idle   → /vision/material_command
    数字: digits / dstop                               → /vision/region_command
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    display = LaunchConfiguration('display')
    camera_device = LaunchConfiguration('camera_device')

    return LaunchDescription([
        DeclareLaunchArgument('display', default_value='true',
                              description='调试窗口开关(物料/圆环/数字三个识别节点)'),
        DeclareLaunchArgument('camera_device', default_value='/dev/video0',
                              description='USB 相机设备号(复位后可能变化)'),
        Node(
            package='logistics_vision_capture',
            executable='camera_capture',
            name='camera_capture',
            output='screen',
            parameters=[{
                'camera_device': camera_device,
                'width': 640,
                'height': 480,
                'display': False,
            }],
        ),
        Node(
            package='logistics_vision_capture',
            executable='camera_undistort',
            name='camera_undistort',
            output='screen',
            parameters=[{'display': False}],
        ),
        Node(
            package='logistics_vision_capture',
            executable='material_detect',
            name='material_detect',
            output='screen',
            parameters=[{
                'display': display,
                'camera_to_tray_mm': 280.0,
            }],
        ),
        Node(
            package='logistics_vision_capture',
            executable='ring_detect',
            name='ring_detect',
            output='screen',
            parameters=[{'display': display}],
        ),
        Node(
            package='logistics_vision_capture',
            executable='number_detect',
            name='number_detect',
            output='screen',
            parameters=[{'display': display}],
        ),
    ])
