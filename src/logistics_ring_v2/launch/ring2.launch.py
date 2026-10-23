"""Ring V2 圆环链一键启动: 旧包采集 → 旧包去畸变 → 圆环同帧处理节点。

用法:
    ros2 launch logistics_ring_v2 ring2.launch.py
    ros2 launch logistics_ring_v2 ring2.launch.py display:=false
    ros2 launch logistics_ring_v2 ring2.launch.py camera_device:=/dev/video1
    # 双链联跑(与 material2.launch.py 同时运行): 相机层只起一份
    ros2 launch logistics_ring_v2 ring2.launch.py start:=stop            # 先起的链带相机
    ros2 launch logistics_ring_v2 ring2.launch.py camera:=false start:=stop  # 后起的链不带相机

处理门控: 节点常驻, /vision/ring2/command 发 run 开始处理 / stop 跳过(省 CPU);
start 参数决定启动时的门控初值。主控到工位板发 run, 离开发 stop。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    display = LaunchConfiguration('display')
    display_debug = LaunchConfiguration('display_debug')
    camera_device = LaunchConfiguration('camera_device')
    camera = LaunchConfiguration('camera')
    start = LaunchConfiguration('start')

    return LaunchDescription([
        DeclareLaunchArgument('ring_fast_enabled', default_value='true'),
        DeclareLaunchArgument('ring_opencv_threads', default_value='2'),
        DeclareLaunchArgument('ring_roi_enabled', default_value='true',
                              description='确认三环后局部检圆，false用于全图效果对比'),
        DeclareLaunchArgument('display', default_value='true',
                              description='最终结果窗口(Ring V2 Final)开关'),
        DeclareLaunchArgument('display_debug', default_value='false',
                              description='同帧CLAHE调试窗口开关'),
        DeclareLaunchArgument('camera_device', default_value='/dev/video0',
                              description='USB 相机设备号(复位后可能变化)'),
        DeclareLaunchArgument('camera', default_value='true',
                              description='是否启动相机采集/去畸变层(双链联跑时只起一份)'),
        DeclareLaunchArgument('start', default_value='run',
                              description='圆环节点启动时的门控初值: run/stop'),
        Node(
            package='logistics_vision_capture',
            executable='camera_capture',
            name='camera_capture',
            output='screen',
            condition=IfCondition(camera),
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
            condition=IfCondition(camera),
            parameters=[{'display': False}],
        ),
        Node(
            package='logistics_ring_v2',
            executable='ring_pipeline',
            name='ring2_pipeline',
            output='screen',
            parameters=[{'display': display, 'display_debug': display_debug,
                         'roi_enabled': LaunchConfiguration('ring_roi_enabled'),
                         'fast_enabled': LaunchConfiguration('ring_fast_enabled'),
                         'opencv_threads': LaunchConfiguration('ring_opencv_threads'),
                         'start_mode': start}],
        ),
    ])
