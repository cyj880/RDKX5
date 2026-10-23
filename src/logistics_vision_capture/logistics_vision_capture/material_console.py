"""终端调试接口：模拟主控向 material_detect 发命令，并实时打印识别结果。

命令（回车发送）：
  locate <色1> <色2> <色3>    例: locate 2 3 5   （进入定位模式，盘心锁定）
  track <色>                  例: track 3        （只追踪单个颜色）
  idle                                           （停止识别）
  quit                                           （退出本工具）
颜色编号：红1 黄2 蓝3 绿4 黑5 浅蓝6

未来单片机接入时，由串口桥节点向同一话题 /vision/material_command 发相同
格式的命令即可，本工具保留用于调试。
"""

import json
import threading
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .color_features import color_name


HELP_TEXT = """命令:
  locate <色1> <色2> <色3>   例: locate 3 4 6
  track <色>                 例: track 3
  idle
  digits                     圆环+数字定位开始(小车到达粗加工/暂存区后)
  dstop                      圆环+数字定位停止
  quit
颜色编号: 红1 黄2 蓝3 绿4 黑5 浅蓝6

工作时序调试(视觉侧职责):
  1) 臂展开后主控通知 -> locate <色1> <色2> <色3>  (三色齐自动锁盘心并回传)
  2) 车微校准正对圆心 -> 重发 locate               (重锁, 看重锁偏差=车对准误差)
  3) track <目标色>                                (输出抓取目标底心)
  4) stationary=True 后主控抓取 -> 回home -> 换 track <下一色> 循环
     (第二批同色直接换 track; 全程结束发 idle)
  小车到达粗加工区/暂存区 -> digits (圆环定位+数字识别)
  -> 数字回传完成 -> dstop"""


class MaterialConsoleNode(Node):
    """把终端输入转发为物料识别命令，并把结果打印到终端。"""

    def __init__(self) -> None:
        super().__init__('material_console')
        self.declare_parameter('command_topic', '/vision/material_command')
        self.declare_parameter('result_topic', '/vision/camera/material_results')
        self.declare_parameter('region_command_topic', '/vision/region_command')
        self.declare_parameter('number_result_topic', '/vision/camera/number_results')
        self.declare_parameter('print_results', True)
        command_topic = str(self.get_parameter('command_topic').value)
        result_topic = str(self.get_parameter('result_topic').value)
        region_command_topic = str(self.get_parameter('region_command_topic').value)
        number_result_topic = str(self.get_parameter('number_result_topic').value)
        # 多终端用法: 看结果的终端保持默认(全打); 发命令的终端加
        # -p print_results:=false, 输入提示符不被刷屏
        self.print_results = bool(self.get_parameter('print_results').value)
        self.publisher = self.create_publisher(String, command_topic, 10)
        self.region_publisher = self.create_publisher(String, region_command_topic, 10)
        self.create_subscription(String, result_topic, self._result_callback, 10)
        self.create_subscription(String, number_result_topic, self._number_result_callback, 10)
        mode_text = '输入+结果' if self.print_results else '纯输入(结果请看另一终端)'
        self.get_logger().info(
            '终端接口已启动[%s]: 物料命令 -> %s, 区域命令(digits/dstop) -> %s' % (mode_text, command_topic, region_command_topic)
        )
        print(HELP_TEXT)

    def _number_result_callback(self, message: String) -> None:
        if not self.print_results:
            return
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            print('数字结果解析失败: %s' % message.data, flush=True)
            return
        digits = payload.get('digits', [])
        if digits:
            parts = ['[digits] 数字: ' + '  '.join(
                '%d@(%d,%d) %.2f' % (d['digit'], d['center_u'], d['center_v'], d.get('confidence', 0))
                for d in digits
            )]
        else:
            parts = ['[digits] 数字: 无']
        print(' '.join(parts), flush=True)

    def _result_callback(self, message: String) -> None:
        if not self.print_results:
            return
        import json
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            print('结果解析失败: %s' % message.data)
            return
        parts = ['[%s]' % payload.get('mode', '?')]
        motion = payload.get('motion_px')
        parts.append('静止=%s(运动%s)' % (
            payload.get('stationary', '?'),
            'n/a' if motion is None else '%.1fpx' % motion,
        ))
        materials = payload.get('materials', [])
        if materials:
            parts.append('物料: ' + '  '.join(
                '%d(%s)@底心(%d,%d) %.2f' % (
                    m['color'], color_name(m['color']), m['center_u'], m['center_v'],
                    m.get('confidence', 0),
                ) for m in materials
            ))
        else:
            parts.append('物料: 无')
        center = payload.get('turntable_center')
        if center:
            if center.get('locked'):
                text = '盘心: (%d,%d) 已锁定 盘半径=%.0fpx' % (
                    center['u'], center['v'], center.get('ring_radius_px', 0),
                )
                if 'deviation_px' in center:
                    text += ' 实时偏差=%.1fpx' % center['deviation_px']
                if 'relock_deviation_px' in center:
                    text += ' 重锁偏差=%.1fpx' % center['relock_deviation_px']
                parts.append(text)
            else:
                parts.append('盘心: 锁定中[%s/%s] 即时(%s,%s)' % (
                    center.get('progress', 0), center.get('frames', 0),
                    center.get('u', '?'), center.get('v', '?'),
                ))
        target = payload.get('target')
        if target:
            parts.append('目标: (%d,%d) %.2f' % (
                target['center_u'], target['center_v'], target['confidence'],
            ))
        print(' '.join(parts), flush=True)

    def send(self, text: str) -> None:
        message = String()
        message.data = text.strip()
        self.publisher.publish(message)

    def send_region(self, text: str) -> None:
        """区域命令(digits/dstop)发到 /vision/region_command。"""
        message = String()
        message.data = text.strip()
        self.region_publisher.publish(message)


def _spin_node(node: Node) -> None:
    """后台 spin 线程入口: 捕获关闭异常, 保证退出时线程可 join、无噪音。"""
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[MaterialConsoleNode] = None
    spin_thread = None
    try:
        node = MaterialConsoleNode()
        # 线程必须在节点实例化之后创建: 传 None 会在 spin 线程里崩
        # (AttributeError: 'NoneType' object has no attribute 'executor')
        spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
        spin_thread.start()
        while rclpy.ok():
            try:
                line = input('material> ').strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in ('quit', 'exit', 'q'):
                break
            if line.lower() in ('help', 'h', '?'):
                print(HELP_TEXT)
                continue
            if line.lower() in ('digits', 'dstop'):
                node.send_region(line.lower())
                continue
            node.send(line)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # 先 shutdown 让 spin 线程退出, join 后再销毁, 避免 C 层析构报错
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if node is not None:
            node.destroy_node()


if __name__ == '__main__':
    main()
