"""ROS 2 node for collecting a labeled digit dataset (1/2/3) for validation.

用法：窗口里用鼠标拖框选中数字，按 1/2/3 保存为对应标签的样本（自动递增编号）；
按 Backspace/Delete 撤销上一张；按 Q/Esc 退出。
样本按标签分目录保存：~/logistics_vision_data/digit_dataset/digit_N/img_XXXX.png
（保存的是框选区域的灰度图，供后续评测脚本按识别流水线重新处理。）
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


WINDOW_NAME = 'Digit Dataset Capture'


class DigitDatasetCaptureNode(Node):
    """鼠标框选数字并批量保存带标签的灰度样本。"""

    def __init__(self) -> None:
        super().__init__('digit_dataset_capture')

        self.declare_parameter('input_topic', '/vision/camera/image_undistorted')
        self.declare_parameter(
            'data_root', '~/logistics_vision_data/digit_dataset'
        )
        self.declare_parameter('processing_scale', 0.8)

        self.data_root = Path(
            str(self.get_parameter('data_root').value)
        ).expanduser()
        for digit in (1, 2, 3):
            (self.data_root / ('digit_%d' % digit)).mkdir(parents=True, exist_ok=True)
        self.processing_scale = min(
            1.0, max(0.1, float(self.get_parameter('processing_scale').value))
        )

        self.bridge = CvBridge()
        self.frame: Optional[np.ndarray] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_end: Optional[Tuple[int, int]] = None
        self.window_created = False
        self.last_saved: Optional[Path] = None

        input_topic = str(self.get_parameter('input_topic').value)
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self._image_callback,
            10,
        )
        self.get_logger().info(
            '数字数据集采集节点已启动：拖框选中数字，按 1/2/3 保存，'
            'Backspace 撤销，Q 退出。'
        )

    def _image_callback(self, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error('无法将 ROS 图像转换为 bgr8: %s' % error)
            return

        # 与识别时一致，压缩到 processing_scale。
        if self.processing_scale < 0.999:
            image = cv2.resize(
                image,
                None,
                fx=self.processing_scale,
                fy=self.processing_scale,
                interpolation=cv2.INTER_AREA,
            )
        self.frame = image

        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)
            self.window_created = True

        cv2.imshow(WINDOW_NAME, self._draw(image))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('退出，共采集 %s。' % self._summary())
            rclpy.shutdown()
        elif key in (ord('1'), ord('2'), ord('3')):
            self._save(int(chr(key)))
        elif key in (8, 127):  # Backspace / Delete
            self._undo()

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_end = (x, y)

    def _draw(self, image: np.ndarray) -> np.ndarray:
        display = image.copy()
        if self.drag_start is not None and self.drag_end is not None:
            cv2.rectangle(display, self.drag_start, self.drag_end, (0, 255, 0), 2)
        counts = self._counts()
        cv2.putText(
            display,
            '1:%d  2:%d  3:%d' % (counts[1], counts[2], counts[3]),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            'drag + 1/2/3 save, Backspace undo, Q quit',
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return display

    def _counts(self) -> dict:
        counts = {}
        for digit in (1, 2, 3):
            counts[digit] = len(
                list((self.data_root / ('digit_%d' % digit)).glob('img_*.png'))
            )
        return counts

    def _next_path(self, digit: int) -> Path:
        directory = self.data_root / ('digit_%d' % digit)
        existing = [
            int(path.stem.split('_')[1])
            for path in directory.glob('img_*.png')
        ]
        number = max(existing, default=0) + 1
        return directory / ('img_%04d.png' % number)

    def _save(self, digit: int) -> None:
        if self.frame is None:
            self.get_logger().error('还没有画面。')
            return
        if self.drag_start is None or self.drag_end is None:
            self.get_logger().warn('请先用鼠标拖框选中数字。')
            return
        x1, y1 = self.drag_start
        x2, y2 = self.drag_end
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 5 or y2 - y1 < 5:
            self.get_logger().warn('选框太小，请重新框选。')
            return
        roi = self.frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        path = self._next_path(digit)
        if not cv2.imwrite(str(path), gray):
            self.get_logger().error('样本保存失败: %s' % path)
            return
        self.last_saved = path
        self.get_logger().info('已保存 %s' % path)
        self.drag_start = None
        self.drag_end = None

    def _undo(self) -> None:
        if self.last_saved is None:
            self.get_logger().warn('没有可撤销的样本。')
            return
        try:
            self.last_saved.unlink()
            self.get_logger().info('已删除 %s' % self.last_saved)
        except FileNotFoundError:
            pass
        self.last_saved = None

    def _summary(self) -> str:
        counts = self._counts()
        return '1:%d, 2:%d, 3:%d' % (counts[1], counts[2], counts[3])

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[DigitDatasetCaptureNode] = None
    try:
        node = DigitDatasetCaptureNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('digit_dataset_capture 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
