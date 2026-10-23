"""ROS 2 node that creates OpenCV camera calibration YAML from chessboard images."""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from .chessboard import find_chessboard_corners


MINIMUM_VALID_IMAGES = 20


class CameraCalibrateNode(Node):
    """Calibrates the camera from saved 9x6 chessboard images."""

    def __init__(self) -> None:
        super().__init__('camera_calibrate')
        self.declare_parameter('data_root', '~/logistics_vision_data')
        self.declare_parameter('chessboard_columns', 9)
        self.declare_parameter('chessboard_rows', 6)
        self.declare_parameter('square_size_mm', 18.0)
        self.declare_parameter('minimum_valid_images', MINIMUM_VALID_IMAGES)
        self.declare_parameter('output_path', '')

        self.data_root = Path(str(self.get_parameter('data_root').value)).expanduser()
        self.pattern_size = (
            int(self.get_parameter('chessboard_columns').value),
            int(self.get_parameter('chessboard_rows').value),
        )
        self.square_size_mm = float(self.get_parameter('square_size_mm').value)
        self.minimum_valid_images = int(
            self.get_parameter('minimum_valid_images').value
        )
        self.output_path_parameter = str(self.get_parameter('output_path').value)

    def calibrate(self) -> bool:
        if self.pattern_size[0] <= 1 or self.pattern_size[1] <= 1:
            self.get_logger().error('棋盘格内角点数量必须大于 1。')
            return False
        if self.square_size_mm <= 0.0:
            self.get_logger().error('棋盘格单格边长必须大于 0。')
            return False

        image_dir = self.data_root / 'calibration_images'
        image_paths = sorted(
            path
            for path in image_dir.glob('*')
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
        )
        if not image_paths:
            self.get_logger().error('未找到标定图片目录或有效图片: %s' % image_dir)
            return False

        object_points: List[np.ndarray] = []
        image_points: List[np.ndarray] = []
        image_size: Optional[Tuple[int, int]] = None
        expected_object_points = self._object_points()

        for path in image_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warn('无法读取图片，已跳过: %s' % path)
                continue
            height, width = image.shape[:2]
            current_size = (width, height)
            if image_size is None:
                image_size = current_size
            elif current_size != image_size:
                self.get_logger().warn(
                    '图片分辨率 %dx%d 与首张 %dx%d 不一致，已跳过: %s'
                    % (*current_size, *image_size, path.name)
                )
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found, corners = find_chessboard_corners(gray, self.pattern_size)
            if not found or corners is None:
                self.get_logger().warn('未检测到完整棋盘格，已跳过: %s' % path.name)
                continue
            object_points.append(expected_object_points.copy())
            image_points.append(corners)

        valid_count = len(image_points)
        self.get_logger().info(
            '标定图片统计: 共 %d 张，完整棋盘格 %d 张。'
            % (len(image_paths), valid_count)
        )
        if valid_count < self.minimum_valid_images or image_size is None:
            self.get_logger().error(
                '有效标定图片不足：需要至少 %d 张，当前只有 %d 张。'
                % (self.minimum_valid_images, valid_count)
            )
            return False

        rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
        output_path = self._output_path(image_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._write_yaml(
            output_path,
            camera_matrix,
            distortion,
            image_size,
            valid_count,
            rms,
        ):
            return False

        self.get_logger().info(
            '标定完成。RMS 重投影误差: %.4f 像素，结果已保存: %s'
            % (rms, output_path)
        )
        return True

    def _object_points(self) -> np.ndarray:
        columns, rows = self.pattern_size
        points = np.zeros((columns * rows, 3), np.float32)
        points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
        points *= self.square_size_mm
        return points

    def _output_path(self, image_size: Tuple[int, int]) -> Path:
        if self.output_path_parameter:
            return Path(self.output_path_parameter).expanduser()
        width, height = image_size
        return self.data_root / 'calibration' / ('camera_%dx%d.yaml' % (width, height))

    def _write_yaml(
        self,
        output_path: Path,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        image_size: Tuple[int, int],
        valid_count: int,
        rms: float,
    ) -> bool:
        storage = cv2.FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
        if not storage.isOpened():
            self.get_logger().error('无法创建标定文件: %s' % output_path)
            return False
        width, height = image_size
        storage.write('camera_matrix', camera_matrix)
        storage.write('distortion_coefficients', distortion)
        storage.write('image_width', float(width))
        storage.write('image_height', float(height))
        storage.write('chessboard_columns', float(self.pattern_size[0]))
        storage.write('chessboard_rows', float(self.pattern_size[1]))
        storage.write('square_size_mm', self.square_size_mm)
        storage.write('valid_image_count', valid_count)
        storage.write('rms_reprojection_error', float(rms))
        storage.release()
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[CameraCalibrateNode] = None
    try:
        node = CameraCalibrateNode()
        if not node.calibrate():
            raise SystemExit(1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
