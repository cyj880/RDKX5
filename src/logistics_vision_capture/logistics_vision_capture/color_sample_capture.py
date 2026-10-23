"""ROS 2 node for collecting labeled material color samples (1-6).

用法：窗口里鼠标拖框选中物料主体（紧贴边缘），按 1~6 保存为对应颜色样本
（红1 黄2 蓝3 绿4 黑5 浅蓝6），Backspace 撤销上一张，Q/Esc 退出。
样本为框选区域 BGR 图，存到 ~/logistics_vision_data/color_dataset/color_N/。
多光照采样指南：正常光、侧光、半阴影、偏暗、偏亮、不同角度距离都要覆盖。
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

from .color_features import color_abbr, color_name


WINDOW_NAME = 'Color Sample Capture'
MIN_BOX_SIZE = 8  # 框选区域最小边长（像素）


class ColorSampleCaptureNode(Node):
    """拖框选物料并按键保存带标签的颜色样本。"""

    def __init__(self) -> None:
        super().__init__('color_sample_capture')
        self.declare_parameter('input_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('data_root', '~/logistics_vision_data/color_dataset')

        self.data_root = Path(
            str(self.get_parameter('data_root').value)
        ).expanduser()
        for color in range(1, 7):
            (self.data_root / ('color_%d' % color)).mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.frame: Optional[np.ndarray] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_end: Optional[Tuple[int, int]] = None
        self.saved_stack: List[Path] = []  # 支持连续撤销
        self.window_created = False

        input_topic = str(self.get_parameter('input_topic').value)
        self.subscription = self.create_subscription(
            Image, input_topic, self._image_callback, 10
        )
        self.get_logger().info(
            '颜色样本采集已启动: 订阅 %s，保存目录 %s。拖框后按 1~6 保存。'
            % (input_topic, self.data_root)
        )

    def _image_callback(self, message: Image) -> None:
        try:
            self.frame = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error('无法转换图像: %s' % error)
            return

        if not self.window_created:
            # AUTOSIZE: 窗口与 640x480 图像 1:1, 与 digit_dataset_capture 相同的
            # 交互表现; WINDOW_NORMAL 拉伸后 GTK 鼠标坐标不换算, 拖框会错位
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)
            self.window_created = True
        self._show(self.frame)

    def _mouse_callback(self, event: int, x: int, y: int, flags, param) -> None:
        # 与 digit_dataset_capture 相同: MOUSEMOVE 仅在左键按住时跟框
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_end = (x, y)

    def _current_box(self) -> Optional[Tuple[int, int, int, int]]:
        if self.drag_start is None or self.drag_end is None:
            return None
        x0 = min(self.drag_start[0], self.drag_end[0])
        y0 = min(self.drag_start[1], self.drag_end[1])
        x1 = max(self.drag_start[0], self.drag_end[0])
        y1 = max(self.drag_start[1], self.drag_end[1])
        if x1 - x0 < MIN_BOX_SIZE or y1 - y0 < MIN_BOX_SIZE:
            return None
        return x0, y0, x1, y1

    def _save_sample(self, color: int) -> None:
        box = self._current_box()
        if box is None or self.frame is None:
            self.get_logger().warn('先用鼠标拖框选中物料，再按数字保存。')
            return
        x0, y0, x1, y1 = box
        crop = self.frame[y0:y1, x0:x1]
        path = self.data_root / ('color_%d' % color) / (
            'color_%d_%s.png' % (color, datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        )
        if cv2.imwrite(str(path), crop):
            self.saved_stack.append(path)
            self.drag_start = None  # 保存后清框, 防止残留选框误保存(与数字采集一致)
            self.drag_end = None
            self.get_logger().info(
                '已保存 %s 样本: %s (共 %d 张)' % (color_name(color), path.name, len(self.saved_stack))
            )
        else:
            self.get_logger().error('保存失败: %s' % path)

    def _undo(self) -> None:
        if not self.saved_stack:
            self.get_logger().warn('没有可撤销的样本。')
            return
        path = self.saved_stack.pop()
        path.unlink(missing_ok=True)
        self.get_logger().info('已撤销: %s' % path.name)

    def _show(self, frame: np.ndarray) -> None:
        preview = frame.copy()
        box = self._current_box()
        if box is not None:
            x0, y0, x1, y1 = box
            cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)
        counters = {}
        for path in self.saved_stack:
            color = int(path.parent.name.split('_')[1])
            counters[color] = counters.get(color, 0) + 1
        # cv2.putText 不支持中文, 窗口文字用 ASCII
        lines = ['Drag box, press 1-6 save, Backspace undo, Q quit']
        lines.append(' '.join(
            '%s:%d' % (color_abbr(c), counters.get(c, 0)) for c in range(1, 7)
        ))
        for index, text in enumerate(lines):
            cv2.putText(
                preview, text, (10, 25 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        if ord('1') <= key <= ord('6'):
            self._save_sample(key - ord('0'))
        elif key in (8, 255 + 8):  # Backspace
            self._undo()
        elif key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令。')
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ColorSampleCaptureNode] = None
    try:
        node = ColorSampleCaptureNode()
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
