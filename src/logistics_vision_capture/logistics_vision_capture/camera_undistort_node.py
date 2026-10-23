"""ROS 2 node that removes lens distortion from the USB camera image topic."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


WINDOW_NAME = 'Logistics Vision Undistort'


class CameraUndistortNode(Node):
    """Subscribes to raw images, undistorts them, and republishes the result."""

    def __init__(self) -> None:
        super().__init__('camera_undistort')

        self.declare_parameter(
            'calibration_path',
            '~/logistics_vision_data/calibration/camera_640x480.yaml',
        )
        self.declare_parameter('input_topic', '/vision/camera/image_raw')
        self.declare_parameter('output_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('data_root', '~/logistics_vision_data')
        self.declare_parameter('display', True)

        self.calibration_path = Path(
            str(self.get_parameter('calibration_path').value)
        ).expanduser()
        self.data_root = Path(str(self.get_parameter('data_root').value)).expanduser()
        self.save_dir = self.data_root / 'undistorted_images'
        self.display = bool(self.get_parameter('display').value)
        self.bridge = CvBridge()
        self.window_created = False
        self.size_error_reported = False

        (
            self.camera_matrix,
            self.distortion,
            self.image_size,
        ) = self._load_calibration(self.calibration_path)
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.distortion,
            self.image_size,
            0,
            self.image_size,
        )
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.distortion,
            None,
            self.new_camera_matrix,
            self.image_size,
            cv2.CV_16SC2,
        )

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self._image_callback,
            10,
        )
        self.get_logger().info(
            '去畸变节点已启动: 订阅 %s，发布 %s，标定分辨率 %dx%d。'
            % (input_topic, output_topic, *self.image_size)
        )

    def _load_calibration(
        self,
        path: Path,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        if not path.is_file():
            raise RuntimeError('未找到标定文件: %s' % path)

        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise RuntimeError('无法读取标定文件: %s' % path)
        try:
            camera_matrix = storage.getNode('camera_matrix').mat()
            distortion = storage.getNode('distortion_coefficients').mat()
            width = int(storage.getNode('image_width').real())
            height = int(storage.getNode('image_height').real())
        finally:
            storage.release()

        if camera_matrix is None or camera_matrix.shape != (3, 3):
            raise RuntimeError('标定文件中的 camera_matrix 无效。')
        if distortion is None or distortion.size != 5:
            raise RuntimeError('标定文件中的 distortion_coefficients 无效。')
        if width <= 0 or height <= 0:
            raise RuntimeError('标定文件中的图像分辨率无效。')
        return camera_matrix, distortion, (width, height)

    def _image_callback(self, message: Image) -> None:
        try:
            raw_image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error('无法将 ROS 图像转换为 bgr8: %s' % error)
            return

        height, width = raw_image.shape[:2]
        if (width, height) != self.image_size:
            if not self.size_error_reported:
                self.get_logger().error(
                    '输入图像分辨率为 %dx%d，但标定文件要求 %dx%d；已停止处理该尺寸图像。'
                    % (width, height, *self.image_size)
                )
                self.size_error_reported = True
            return
        self.size_error_reported = False

        undistorted = cv2.remap(
            raw_image,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
        )
        output = self.bridge.cv2_to_imgmsg(undistorted, encoding='bgr8')
        output.header = message.header
        self.publisher.publish(output)

        if self.display:
            self._show_preview(undistorted)

    def _show_preview(self, undistorted: np.ndarray) -> None:
        preview = undistorted.copy()
        cv2.putText(
            preview,
            'UNDISTORTED',
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            'U: SAVE UNDISTORTED  Q/ESC: QUIT',
            (15, preview.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('u'), ord('U')):
            self._save_undistorted(undistorted)
        elif key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令，正在关闭去畸变节点。')
            rclpy.shutdown()

    def _save_undistorted(self, image: np.ndarray) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = self.save_dir / ('undistorted_' + timestamp + '.png')
        if cv2.imwrite(str(path), image):
            self.get_logger().info('已保存去畸变图片: %s' % path)
        else:
            self.get_logger().error('去畸变图片保存失败: %s' % path)

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[CameraUndistortNode] = None
    try:
        node = CameraUndistortNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('camera_undistort 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
