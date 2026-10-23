"""Material2 颜色样本采集工具: 框选物料, 保存 crop + 背景基准 sidecar。

相对特征需要每帧的白背景基准 (μL, μa, μb)。旧数据集是紧贴物料的裁剪图,
不含背景像素, 无法离线补算基准——本工具在采集瞬间用与在线节点完全相同的
方式(blur → HSV/LAB → S<阈值像素取 Lab 中位数)现算基准, 与裁剪图一起存:

    ~/logistics_vision_data/color_dataset_v2/color_{1..6}/
        color_{N}_{时间戳}.png    裁剪图(原始 BGR)
        color_{N}_{时间戳}.json   sidecar: {color_id, rect, bg_ref, image, saved_at}

训练脚本 material2_color_train 据此提取与在线一致的相对特征。

用法(需先启动相机链):
    ros2 run logistics_ring_v2 material2_sample_capture
操作: 左键拖框圈住物料 → 按 1~6 以对应颜色保存 → Backspace 撤销上一张 →
      Q/Esc 退出。采完用 material2_color_train 训练。

注意: saturation_threshold 参数应与 material2_detect 保持一致(默认 40),
否则采集时的 bg_ref 与在线运行时的基准口径不同。
"""

from datetime import datetime
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

from .material_features import (
    color_abbr,
    color_name,
    estimate_background_ref,
)


WINDOW_NAME = 'Material2 Sample Capture'
MIN_BOX_SIZE = 8  # 最小框边长(px), 防误触


class Material2SampleCapture(Node):
    """订阅去畸变图, 框选物料保存训练样本(裁剪图 + 背景基准)。"""

    def __init__(self) -> None:
        super().__init__('material2_sample_capture')
        self.declare_parameter('input_topic', '/vision/camera/image_undistorted')
        self.declare_parameter(
            'data_root', '~/logistics_vision_data/color_dataset_v2')
        # 必须与 material2_detect 的同名参数保持一致(bg_ref 口径)
        self.declare_parameter('saturation_threshold', 40)
        self.saturation_threshold = int(
            self.get_parameter('saturation_threshold').value)

        self.data_root = Path(
            str(self.get_parameter('data_root').value)).expanduser()
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.frame: Optional[np.ndarray] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        # 已保存样本记录, 供撤销
        self._saved: List[Tuple[Path, Path]] = []
        self.counts = {color: 0 for color in range(1, 7)}
        self.window_created = False

        input_topic = str(self.get_parameter('input_topic').value)
        self.subscription = self.create_subscription(
            Image, input_topic, self._image_callback, 10)
        self.get_logger().info(
            '样本采集已启动: 订阅 %s, 保存到 %s。拖框后按 1~6 保存对应颜色。'
            % (input_topic, self.data_root))

    def _image_callback(self, message: Image) -> None:
        try:
            self.frame = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error('无法转换图像: %s' % error)
            return
        self._show()

    def _compute_bg_ref(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """与在线节点完全一致的背景基准口径: blur → HSV/LAB → S<阈值中位数。"""
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        return estimate_background_ref(lab, hsv[:, :, 1], self.saturation_threshold)

    def _save(self, color_id: int) -> None:
        rect = self._current_rect()
        if rect is None:
            self.get_logger().warn('未框选区域: 先左键拖框圈住物料再按键保存。')
            return
        if self.frame is None:
            return
        x0, y0, x1, y1 = rect
        if x1 - x0 < MIN_BOX_SIZE or y1 - y0 < MIN_BOX_SIZE:
            self.get_logger().warn('框太小(最小 %dpx)。' % MIN_BOX_SIZE)
            return
        bg_ref = self._compute_bg_ref(self.frame)
        if bg_ref is None:
            self.get_logger().error(
                '画面白背景像素不足(S<%d), 无法计算 bg_ref; 请确认相机对准白盘/白地图。'
                % self.saturation_threshold)
            return

        color_dir = self.data_root / ('color_%d' % color_id)
        color_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        stem = 'color_%d_%s' % (color_id, timestamp)
        png_path = color_dir / (stem + '.png')
        json_path = color_dir / (stem + '.json')
        crop = self.frame[y0:y1, x0:x1]
        if not cv2.imwrite(str(png_path), crop):
            self.get_logger().error('裁剪图保存失败: %s' % png_path)
            return
        sidecar = {
            'color_id': color_id,
            'rect': [int(x0), int(y0), int(x1), int(y1)],
            'bg_ref': [round(value, 4) for value in bg_ref],
            'image': png_path.name,
            'saved_at': timestamp,
        }
        json_path.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2), encoding='utf-8')
        self._saved.append((png_path, json_path))
        self.counts[color_id] += 1
        self.drag_start = None
        self.drag_current = None
        self.get_logger().info(
            '已保存 %s 样本: %s (累计 %s)'
            % (color_name(color_id), png_path.name,
               ' '.join('%s:%d' % (color_abbr(c), n) for c, n in self.counts.items())))

    def _undo(self) -> None:
        if not self._saved:
            self.get_logger().warn('没有可撤销的样本。')
            return
        png_path, json_path = self._saved.pop()
        sidecar = json.loads(json_path.read_text(encoding='utf-8'))
        png_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        self.counts[int(sidecar['color_id'])] -= 1
        self.get_logger().info('已撤销: %s' % png_path.name)

    def _current_rect(self) -> Optional[Tuple[int, int, int, int]]:
        if self.drag_start is None or self.drag_current is None:
            return None
        x0, y0 = self.drag_start
        x1, y1 = self.drag_current
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_current = (x, y)

    def _show(self) -> None:
        if self.frame is None:
            return
        if not self.window_created:
            # 必须显式建窗并挂鼠标回调: imshow 隐式创建的窗口收不到拖拽事件
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
            self.window_created = True
        preview = self.frame.copy()
        rect = self._current_rect()
        if rect is not None:
            cv2.rectangle(preview, rect[:2], rect[2:], (0, 255, 0), 2)
        lines = [
            'DRAG box on material, press 1-6 to save (R Y B G K LB)',
            'BACKSPACE: undo last   Q/ESC: quit',
            'saved: %s' % ' '.join(
                '%s:%d' % (color_abbr(c), n) for c, n in self.counts.items()),
        ]
        for index, text in enumerate(lines):
            cv2.putText(preview, text, (10, 25 + index * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                        cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, preview)
        # WINDOW_AUTOSIZE: 拉伸窗口后鼠标坐标错位(见旧包教训), 不允许拉伸
        key = cv2.waitKey(1) & 0xFF
        if ord('1') <= key <= ord('6'):
            self._save(key - ord('0'))
        elif key in (8, 127):
            self._undo()
        elif key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令。')
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[Material2SampleCapture] = None
    try:
        node = Material2SampleCapture()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('material2_sample_capture 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
