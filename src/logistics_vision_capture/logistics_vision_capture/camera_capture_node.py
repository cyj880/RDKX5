"""ROS 2 node for USB camera preview, publishing, and image capture."""

from datetime import datetime
from pathlib import Path
import re
from typing import Optional

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

from .chessboard import find_chessboard_corners


WINDOW_NAME = 'Logistics Vision Camera'


class CameraCaptureNode(Node):
    """Publishes a USB camera image and saves calibration or site images."""

    def __init__(self) -> None:
        super().__init__('camera_capture')

        self.declare_parameter('camera_device', '/dev/video0')
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 30)
        self.declare_parameter('data_root', '~/logistics_vision_data')
        self.declare_parameter('scene_label', 'empty_site')
        self.declare_parameter('chessboard_columns', 9)
        self.declare_parameter('chessboard_rows', 6)
        self.declare_parameter('image_topic', '/vision/camera/image_raw')
        self.declare_parameter('display', True)
        self.declare_parameter('max_read_failures', 30)
        # MJPG 压缩流带宽 ~2MB/s, YUYV 未压缩 ~18MB/s 在 VMware USB 直通下
        # 会周期性僵死(select timeout); OpenCV 自动解码, 下游仍拿 BGR 帧
        self.declare_parameter('pixel_format', 'MJPG')

        self.camera_device = str(self.get_parameter('camera_device').value)
        self.requested_width = int(self.get_parameter('width').value)
        self.requested_height = int(self.get_parameter('height').value)
        self.requested_fps = int(self.get_parameter('fps').value)
        self.pattern_size = (
            int(self.get_parameter('chessboard_columns').value),
            int(self.get_parameter('chessboard_rows').value),
        )
        self.display = bool(self.get_parameter('display').value)
        self.max_read_failures = int(self.get_parameter('max_read_failures').value)
        self.scene_label = self._safe_label(
            str(self.get_parameter('scene_label').value)
        )

        data_root = Path(str(self.get_parameter('data_root').value)).expanduser()
        self.calibration_dir = data_root / 'calibration_images'
        self.empty_site_dir = data_root / 'empty_site_images'
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.empty_site_dir.mkdir(parents=True, exist_ok=True)

        topic = str(self.get_parameter('image_topic').value)
        self.publisher = self.create_publisher(Image, topic, 10)
        self.bridge = CvBridge()
        self.read_failures = 0
        self.window_created = False

        self.capture = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                f'无法打开摄像头 {self.camera_device}。请确认 USB 已直通并存在该设备。'
            )

        pixel_format = str(self.get_parameter('pixel_format').value)
        if pixel_format:
            fourcc = cv2.VideoWriter_fourcc(*pixel_format)
            if not self.capture.set(cv2.CAP_PROP_FOURCC, fourcc):
                self.get_logger().warn(
                    '像素格式 %s 设置失败，使用相机默认格式。' % pixel_format
                )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        self.capture.set(cv2.CAP_PROP_FPS, self.requested_fps)
        self.actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        actual_format = ''.join(
            chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4)
        )

        self.get_logger().info(
            '摄像头已打开: %s, 格式: %s, 实际分辨率: %dx%d, 实际帧率: %.1f FPS'
            % (
                self.camera_device,
                actual_format,
                self.actual_width,
                self.actual_height,
                self.actual_fps,
            )
        )
        if (
            self.actual_width != self.requested_width
            or self.actual_height != self.requested_height
        ):
            self.get_logger().warn(
                '摄像头未使用请求的 %dx%d，而是 %dx%d。请以实际分辨率重新标定。'
                % (
                    self.requested_width,
                    self.requested_height,
                    self.actual_width,
                    self.actual_height,
                )
            )

        period = 1.0 / max(self.requested_fps, 1)
        self.timer = self.create_timer(period, self._capture_frame)

    @staticmethod
    def _safe_label(label: str) -> str:
        safe = re.sub(r'[^A-Za-z0-9_-]+', '_', label).strip('_')
        return safe or 'empty_site'

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    def _capture_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.read_failures += 1
            if self.read_failures == 1:
                self.get_logger().error('摄像头读取失败，正在重试。')
            if self.read_failures >= self.max_read_failures:
                self._reopen_capture()
            return

        self.read_failures = 0
        self.publisher.publish(
            self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        )

        if self.display:
            self._show_preview(frame)

    def _reopen_capture(self) -> None:
        """连续读取失败后释放并重开摄像头: USB 僵死(V4L2 select timeout)自救。

        比赛中相机僵死不能退出节点——重开成功则继续发流; 失败则保持重试,
        待 USB 恢复(复位/重新插拔)后自动接回。注意复位后设备号可能变化。
        """
        self.read_failures = 0
        self.get_logger().error(
            '摄像头连续读取失败 %d 次，尝试重新打开。' % self.max_read_failures
        )
        self.capture.release()
        self.capture = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)
        if self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
            self.capture.set(cv2.CAP_PROP_FPS, self.requested_fps)
            self.get_logger().info('摄像头已重新打开。')
        else:
            self.get_logger().error('摄像头重新打开失败，将继续重试。')

    def _show_preview(self, frame) -> None:
        preview = frame.copy()
        lines = [
            '%dx%d  %.1f FPS' % (self.actual_width, self.actual_height, self.actual_fps),
            'C: CALIBRATION  S: SAVE IMAGE  Q/ESC: QUIT',
        ]
        for index, text in enumerate(lines):
            cv2.putText(
                preview,
                text,
                (15, 30 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('c'), ord('C')):
            self._save_calibration_image(frame)
        elif key in (ord('s'), ord('S')):
            self._save_site_image(frame)
        elif key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令，正在关闭摄像头。')
            rclpy.shutdown()

    def _save_calibration_image(self, frame) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, _ = find_chessboard_corners(gray, self.pattern_size)
        if not found:
            self.get_logger().warn(
                '未检测到完整的 9x6 内角点棋盘格，未保存标定图片。'
            )
            return
        path = self.calibration_dir / ('chessboard_' + self._timestamp() + '.png')
        if cv2.imwrite(str(path), frame):
            self.get_logger().info('已保存标定图片: %s' % path)
        else:
            self.get_logger().error('标定图片保存失败: %s' % path)

    def _save_site_image(self, frame) -> None:
        path = self.empty_site_dir / (
            self.scene_label + '_' + self._timestamp() + '.png'
        )
        if cv2.imwrite(str(path), frame):
            self.get_logger().info('已保存空场图片: %s' % path)
        else:
            self.get_logger().error('空场图片保存失败: %s' % path)

    def destroy_node(self) -> bool:
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[CameraCaptureNode] = None
    try:
        node = CameraCaptureNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('camera_capture 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
