"""Single debug window for all runtime vision nodes."""

import json
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


class DebugDisplayNode(Node):
    def __init__(self) -> None:
        super().__init__('debug_display')
        self.declare_parameter('display', True)
        self.declare_parameter('image_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('window_name', 'Logistics Vision Debug')
        self.bridge = CvBridge()
        self.window_name = str(self.get_parameter('window_name').value)
        self.enabled = bool(self.get_parameter('display').value)
        self.active = 'raw'
        self.track_color = 5
        self.mode_command_pub = self.create_publisher(String, '/vision/mode_command', 10)
        self.gates = {'material_all': 'stop', 'material_track': 'stop',
                      'ring': 'stop', 'turntable': 'stop', 'stacked': 'stop'}
        self.frames = {}
        self.results = {}
        self.last_update = {}
        self.image_subscription = None
        self.pending_image = None
        self.topics = {
            'raw': str(self.get_parameter('image_topic').value),
            'material_all': '/vision/material_all/annotated',
            'material_track': '/vision/material_track/annotated',
            'ring': '/vision/ring2/annotated',
            'turntable': '/vision/turntable/annotated',
            'stacked': '/vision/stacked/annotated',
        }
        retained = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/vision/active_mode', self._mode, retained)
        self._mode(String(data='raw'))
        for mode in ('material_all', 'material_track'):
            self.create_subscription(String, '/vision/%s/results' % mode,
                                     lambda msg, source=mode: self._result(source, msg), 10)
        self.create_subscription(String, '/vision/ring2/sites',
                                 lambda msg: self._result('ring', msg), 10)
        self.create_subscription(String, '/vision/turntable/results',
                                 lambda msg: self._result('turntable', msg), 10)
        self.create_subscription(String, '/vision/stacked/results',
                                 lambda msg: self._result('stacked', msg), 10)
        self.timer = self.create_timer(0.03, self._refresh)
        self.get_logger().info('统一调试窗口已启动，当前显示 raw，源码=%s。' % __file__)

    def _mode(self, message: String) -> None:
        source = message.data
        if source not in self.topics:
            return
        if self.image_subscription is not None:
            self.destroy_subscription(self.image_subscription)
        self.active = source
        pending = getattr(self, 'pending_command', None)
        expected = 'material_track' if pending and pending.startswith('material track ') else pending
        if source == expected:
            self.pending_command = None
            self.request_time = None
        self.frames.clear()
        self.results.clear()
        self.pending_image = None
        self.gates = {source: 'stop' if source == 'raw' else 'run'}
        if self.enabled:
            self.image_subscription = self.create_subscription(
                Image, self.topics[source], lambda msg: self._image(source, msg), 1)

    def _image(self, source: str, message: Image) -> None:
        if source == self.active:
            self.pending_image = message

    def _result(self, source: str, message: String) -> None:
        if source != self.active:
            return
        try:
            self.results[source] = json.loads(message.data)
            if source == 'material_track':
                color = self.results[source].get('track_color')
                if color in range(1, 7):
                    self.track_color = color
        except json.JSONDecodeError:
            self.results[source] = {'detail': 'invalid result'}

    def _refresh(self) -> None:
        if not self.enabled:
            return
        if self.pending_image is not None:
            message, self.pending_image = self.pending_image, None
            try:
                self.frames[self.active] = self.bridge.imgmsg_to_cv2(message, 'bgr8')
                self.last_update[self.active] = time.monotonic()
            except CvBridgeError as error:
                self.get_logger().warn('调试图像转换失败: %s' % error)
        frame = self.frames.get(self.active)
        if frame is None:
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, 'WAITING FOR FRAME', (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if time.monotonic() - self.last_update.get(self.active, 0) > 2:
            frame = frame.copy()
            cv2.putText(frame, 'NO RECENT FRAME', (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        view = frame.copy()
        result = self.results.get(self.active, {})
        gate = self.gates.get(self.active, 'idle')
        lines = ['MODE: %s   GATE: %s' % (self.active.upper(), str(gate).upper())]
        if getattr(self, 'pending_command', None) and time.monotonic() - self.request_time > 1.0:
            lines.append('REQUEST: %s - WAITING FOR MODE GATE' % self.pending_command.upper())
        if self.active == 'turntable':
            lines.append('DETECTED: %s  STABLE: %s' % (result.get('detected', False), result.get('stable', False)))
            lines.append('CENTER: (%s, %s)' % (result.get('center_u', '--'), result.get('center_v', '--')))
        elif self.active == 'material_all':
            lines.append('MATERIALS: %d' % len(result.get('materials', [])))
            colors = [item.get('color') for item in result.get('materials', [])]
            lines.append('COLORS: %s' % (colors if colors else '--'))
        elif self.active == 'material_track':
            lines.append('COLOR: %s  STATIONARY: %s  IN_TURN: %s' % (
                result.get('track_color', '--'), result.get('stationary', False),
                result.get('in_turntable', '--')))
            lines.append('GRASP_READY: %s  TARGET: %s' % (
                result.get('grasp_ready', False),
                result.get('target').get('center_u') if result.get('target') else 'NO'))
        elif self.active == 'ring':
            lines.append('STATUS: %s' % result.get('status', '--'))
        elif self.active == 'stacked':
            lines.append('DETECTED: %s  STABLE: %s' % (result.get('detected', False), result.get('stable', False)))
            lines.append('STATE: %s  YAW: %s' % (result.get('state', '--'), result.get('station_yaw_deg', '--')))
        # 黄色 HUD 放在画面底部: 避开各节点标注图自带的顶部状态文字(否则重叠)
        origin_y = view.shape[0] - len(lines) * 24 - 12
        for index, line in enumerate(lines):
            cv2.putText(view, line, (10, origin_y + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.window_name, view)
        self._handle_key(cv2.waitKey(1) & 0xFF)

    def _handle_key(self, key: int) -> None:
        if key in (ord('q'), ord('Q'), 27):
            rclpy.shutdown()
            return
        if ord('1') <= key <= ord('6'):
            color = key - ord('0')
            if self.active == 'material_track' and self.track_color == color:
                return
            self.track_color = color
            command = 'material track %d' % color
        else:
            modes = {'i': 'raw', 'r': 'ring', 'm': 'material_all',
                     't': 'material_track', 'a': 'turntable', 's': 'stacked'}
            mode = modes.get(chr(key).lower()) if 0 <= key <= 255 else None
            if mode is None or (mode == self.active and not getattr(self, 'pending_command', None)):
                return
            command = 'material track %d' % self.track_color if mode == 'material_track' else mode
        # All keyboard commands use the same public gate as the external controller.
        self.pending_command = command
        self.request_time = time.monotonic()
        self.mode_command_pub.publish(String(data=command))
        self.get_logger().info('窗口按键命令: %s' % command)

    def destroy_node(self) -> bool:
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DebugDisplayNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
