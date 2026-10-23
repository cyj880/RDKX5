"""统一启动：相机、去畸变、物料、圆环、转盘和唯一调试窗口。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera = LaunchConfiguration('camera')
    camera_device = LaunchConfiguration('camera_device')
    display = LaunchConfiguration('display')
    serial_port = LaunchConfiguration('serial_port')
    serial_baudrate = LaunchConfiguration('serial_baudrate')
    material_mode = LaunchConfiguration('start_material_mode')
    ring_mode = LaunchConfiguration('start_ring_mode')
    turntable_mode = LaunchConfiguration('start_turntable_mode')
    stacked_mode = LaunchConfiguration('start_stacked_mode')
    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value='true'),
        DeclareLaunchArgument('camera_device', default_value='/dev/video0'),
        DeclareLaunchArgument('display', default_value='true'),
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyS1'),
        DeclareLaunchArgument('serial_baudrate', default_value='115200'),
        DeclareLaunchArgument('start_mode', default_value='raw'),
        DeclareLaunchArgument('ring_roi_enabled', default_value='true'),
        DeclareLaunchArgument('ring_fast_enabled', default_value='true'),
        DeclareLaunchArgument('ring_opencv_threads', default_value='2'),
        DeclareLaunchArgument('start_material_mode', default_value='idle'),
        DeclareLaunchArgument('start_ring_mode', default_value='stop'),
        DeclareLaunchArgument('start_turntable_mode', default_value='stop'),
        DeclareLaunchArgument('start_stacked_mode', default_value='stop'),
        Node(package='logistics_vision_capture', executable='camera_capture',
             name='camera_capture', output='screen', condition=IfCondition(camera),
             parameters=[{'camera_device': camera_device, 'width': 640,
                          'height': 480, 'display': False}]),
        Node(package='logistics_vision_capture', executable='camera_undistort',
             name='camera_undistort', output='screen', condition=IfCondition(camera),
             parameters=[{'display': False}]),
        Node(package='logistics_ring_v2', executable='material_all_detect',
             name='material_all_detect', output='screen',
             parameters=[{'display': False, 'display_debug': False,
                          'start_mode': 'stop',
                          'command_topic': '/vision/gated/material_all/command'}]),
        Node(package='logistics_ring_v2', executable='material_track_detect',
             name='material_track_detect', output='screen',
             parameters=[{'display': False, 'display_debug': False,
                          'start_mode': 'stop',
                          'command_topic': '/vision/gated/material_track/command'}]),
        Node(package='logistics_ring_v2', executable='ring_pipeline',
             name='ring2_pipeline', output='screen',
             parameters=[{'display': False, 'start_mode': 'stop',
                          'roi_enabled': LaunchConfiguration('ring_roi_enabled'),
                          'fast_enabled': LaunchConfiguration('ring_fast_enabled'),
                          'opencv_threads': LaunchConfiguration('ring_opencv_threads'),
                          'command_topic': '/vision/gated/ring/command'}]),
        Node(package='logistics_ring_v2', executable='turntable_detect',
             name='turntable_detect', output='screen',
             parameters=[{'display': False, 'start_mode': 'stop',
                          'command_topic': '/vision/gated/turntable/command'}]),
        Node(package='logistics_ring_v2', executable='stacked_ring_detect',
             name='stacked_ring_detect', output='screen',
             parameters=[{'display': False, 'start_mode': 'stop',
                          'command_topic': '/vision/gated/stacked/command'}]),
        Node(package='logistics_ring_v2', executable='ring_serial_bridge',
             name='ring_serial_bridge', output='screen',
             parameters=[{'serial_port': serial_port,
                          'baudrate': serial_baudrate,
                          'mode_topic': '/vision/mode_command',
                          'ring_result_topic': '/vision/ring2/sites',
                          'turntable_result_topic': '/vision/turntable/results'}]),
        Node(package='logistics_ring_v2', executable='mode_gate',
             name='mode_gate', output='screen',
             parameters=[{'start_mode': LaunchConfiguration('start_mode'),
                          'start_material_mode': material_mode,
                          'start_ring_mode': ring_mode,
                          'start_turntable_mode': turntable_mode,
                          'start_stacked_mode': stacked_mode}]),
        Node(package='logistics_ring_v2', executable='debug_display',
             name='debug_display', output='screen',
             parameters=[{'display': display}]),
    ])
