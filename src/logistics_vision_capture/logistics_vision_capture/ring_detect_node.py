"""ROS 2 node for locating rings: threshold, contour finding, circle fitting and grouping.

圆环识别第三步（调试版）：压缩 → 灰度 → 全局大津法(Otsu) 找黑 → 找轮廓 →
对每个轮廓做最小二乘圆拟合，按半径范围、圆形度、圆周覆盖率筛选；再把圆心相近的
圆聚类成同心圆组（组圆心取成员圆心平均值），各成员圆画绿色、最内环画黑色、组
圆心画十字，并发布 JSON 结果。
"""

from dataclasses import dataclass
import json
from typing import List, Optional, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


BINARY_WINDOW = 'Logistics Vision Binary'
CIRCLES_WINDOW = 'Logistics Vision Circles'


@dataclass(frozen=True)
class Circle:
    """一个拟合出来的候选圆（坐标为压缩后图像坐标）。"""

    center: Tuple[int, int]
    radius: int


@dataclass(frozen=True)
class RingGroup:
    """一个同心圆组：组圆心和成员圆环（按半径升序）。"""

    center: Tuple[int, int]
    circles: List[Circle]


class RingDetectNode(Node):
    """阈值找黑后，用轮廓 + 圆拟合定位圆环。"""

    def __init__(self) -> None:
        super().__init__('ring_detect')

        self.declare_parameter('input_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('binary_topic', '/vision/camera/ring_binary')
        self.declare_parameter('result_topic', '/vision/camera/ring_results')
        # 命令驱动: 小车到达区域后主控发 digits 开始识别, dstop 停止;
        # 默认 idle(未通知不发布结果), 调试窗口照常显示
        self.declare_parameter('command_topic', '/vision/region_command')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('processing_scale', 0.8)
        self.declare_parameter('min_radius_px', 12)
        self.declare_parameter('max_radius_px', 120)
        self.declare_parameter('max_radial_error', 0.15)
        self.declare_parameter('min_coverage', 0.4)
        self.declare_parameter('center_cluster_distance_px', 12.0)
        self.declare_parameter('min_concentric_circles', 2)
        self.declare_parameter('display', True)

        self.image_size = (
            int(self.get_parameter('image_width').value),
            int(self.get_parameter('image_height').value),
        )
        self.processing_scale = min(
            1.0, max(0.1, float(self.get_parameter('processing_scale').value))
        )
        self.min_radius = int(self.get_parameter('min_radius_px').value)
        self.max_radius = int(self.get_parameter('max_radius_px').value)
        self.max_radial_error = float(self.get_parameter('max_radial_error').value)
        self.min_coverage = float(self.get_parameter('min_coverage').value)
        self.center_cluster_distance = float(
            self.get_parameter('center_cluster_distance_px').value
        )
        self.min_concentric_circles = int(
            self.get_parameter('min_concentric_circles').value
        )
        self.display = bool(self.get_parameter('display').value)
        self.mode = 'idle'  # digits→run 开始识别发布, dstop→idle 停止

        self.bridge = CvBridge()
        self.window_created = False
        self.size_error_reported = False

        input_topic = str(self.get_parameter('input_topic').value)
        binary_topic = str(self.get_parameter('binary_topic').value)
        result_topic = str(self.get_parameter('result_topic').value)
        command_topic = str(self.get_parameter('command_topic').value)
        self.binary_publisher = self.create_publisher(Image, binary_topic, 10)
        self.result_publisher = self.create_publisher(String, result_topic, 10)
        self.create_subscription(String, command_topic, self._command_callback, 10)
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self._image_callback,
            10,
        )
        self.get_logger().info(
            '圆环识别节点已启动[idle,等 digits 命令]: 订阅 %s + %s，发布二值图 %s、结果 %s。'
            % (input_topic, command_topic, binary_topic, result_topic)
        )

    def _command_callback(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == 'digits':
            self.mode = 'run'
            self.get_logger().info('进入数字定位模式: 开始发布圆环结果。')
        elif command == 'dstop':
            self.mode = 'idle'
            self.get_logger().info('退出数字定位模式: 停止发布圆环结果。')
        else:
            self.get_logger().warn('无法识别命令 "%s"。可用: digits / dstop' % message.data)

    def _image_callback(self, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error('无法将 ROS 图像转换为 bgr8: %s' % error)
            return

        height, width = image.shape[:2]
        if (width, height) != self.image_size:
            if not self.size_error_reported:
                self.get_logger().error(
                    '输入图像分辨率为 %dx%d，预期 %dx%d。'
                    % (width, height, *self.image_size)
                )
                self.size_error_reported = True
            return
        self.size_error_reported = False

        corrected, binary, circles = self._detect(image)
        scale_x = self.image_size[0] / float(corrected.shape[1])
        scale_y = self.image_size[1] / float(corrected.shape[0])
        groups = self._group(circles, scale_x)

        if self.mode == 'run':
            binary_message = self.bridge.cv2_to_imgmsg(binary, encoding='mono8')
            binary_message.header = message.header
            self.binary_publisher.publish(binary_message)
            self._publish_results(groups, scale_x, scale_y)

        if self.display:
            annotated = self._annotate(corrected, groups)
            self._show(binary, annotated)

    def _detect(self, image) -> tuple:
        # 按参数压缩分辨率，降低后续计算量。
        if self.processing_scale < 0.999:
            resized = cv2.resize(
                image,
                None,
                fx=self.processing_scale,
                fy=self.processing_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = image

        # 彩色转灰度，全局大津法把黑色抠成前景（白）。
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        # 找轮廓。
        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        # 半径范围从全分辨率换算到压缩后坐标。
        process_height, process_width = gray.shape[:2]
        scale_x = self.image_size[0] / float(process_width)
        scale_y = self.image_size[1] / float(process_height)
        process_min_radius = max(3, int(round(self.min_radius / scale_x)))
        process_max_radius = max(
            process_min_radius, int(round(self.max_radius / scale_x))
        )

        circles: List[Circle] = []
        for contour in contours:
            if len(contour) < 20:
                continue
            points = contour.reshape(-1, 2).astype(np.float32)
            fitted = self._fit_circle(points)
            if fitted is None:
                continue
            cx, cy, radius = fitted
            if radius < process_min_radius or radius > process_max_radius:
                continue
            # 圆形度：轮廓点到圆心的径向标准差 / 半径，越小越像圆。
            distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
            radial_error = float(np.std(distances)) / max(1.0, radius)
            if radial_error > self.max_radial_error:
                continue
            # 圆周覆盖率：剔除短弧、半圆等不完整轮廓。
            coverage = self._angular_coverage(points, cx, cy)
            if coverage < self.min_coverage:
                continue
            circles.append(
                Circle((int(round(cx)), int(round(cy))), int(round(radius)))
            )

        return resized, binary, circles

    @staticmethod
    def _fit_circle(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """最小二乘圆拟合（Kasa 法），返回 (cx, cy, r)；退化时返回 None。"""
        x = points[:, 0]
        y = points[:, 1]
        matrix = np.column_stack((x, y, np.ones(len(x), dtype=np.float32)))
        target = x * x + y * y
        solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
        cx = solution[0] / 2.0
        cy = solution[1] / 2.0
        radius_sq = solution[2] + cx * cx + cy * cy
        if not np.isfinite(radius_sq) or radius_sq <= 0:
            return None
        return float(cx), float(cy), float(np.sqrt(radius_sq))

    @staticmethod
    def _angular_coverage(
        points: np.ndarray, cx: float, cy: float, sector_count: int = 36
    ) -> float:
        """返回轮廓点覆盖的圆周扇区比例（按角度分 sector_count 个扇区）。"""
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        sectors = ((angles + 2.0 * np.pi) % (2.0 * np.pi)) // (2.0 * np.pi / sector_count)
        return len(np.unique(sectors.astype(int))) / float(sector_count)

    def _group(self, circles: List[Circle], scale_x: float) -> List[RingGroup]:
        """按圆心距离聚类为同心圆组，组圆心取各成员圆心的平均值。"""
        cluster_distance = self.center_cluster_distance / scale_x

        clusters: List[List[Circle]] = []
        for circle in sorted(circles, key=lambda item: item.radius):
            cluster = None
            for existing in clusters:
                group_x = float(np.mean([item.center[0] for item in existing]))
                group_y = float(np.mean([item.center[1] for item in existing]))
                if np.hypot(
                    circle.center[0] - group_x,
                    circle.center[1] - group_y,
                ) <= cluster_distance:
                    cluster = existing
                    break
            if cluster is None:
                clusters.append([circle])
            else:
                cluster.append(circle)

        groups: List[RingGroup] = []
        for cluster in clusters:
            if len(cluster) < self.min_concentric_circles:
                continue
            sorted_circles = sorted(cluster, key=lambda item: item.radius)
            center_x = int(round(np.mean([item.center[0] for item in sorted_circles])))
            center_y = int(round(np.mean([item.center[1] for item in sorted_circles])))
            groups.append(RingGroup((center_x, center_y), sorted_circles))
        return groups

    def _annotate(self, corrected: np.ndarray, groups: List[RingGroup]) -> np.ndarray:
        # 各成员圆环画绿色；最内环用黑色覆盖；只标组圆心的十字。
        annotated = corrected.copy()
        for group in groups:
            for circle in group.circles:
                cv2.circle(annotated, circle.center, circle.radius, (0, 220, 0), 2)
            innermost = group.circles[0]
            cv2.circle(annotated, innermost.center, innermost.radius, (0, 0, 0), 3)
            cv2.drawMarker(
                annotated,
                group.center,
                (0, 0, 255),
                cv2.MARKER_CROSS,
                18,
                2,
            )
        return annotated

    def _publish_results(
        self,
        groups: List[RingGroup],
        scale_x: float,
        scale_y: float,
    ) -> None:
        payload = {
            'image_width': self.image_size[0],
            'image_height': self.image_size[1],
            'groups': [
                {
                    'center_u': int(round(group.center[0] * scale_x)),
                    'center_v': int(round(group.center[1] * scale_y)),
                    'innermost_radius_px': int(round(group.circles[0].radius * scale_x)),
                    'circle_count': len(group.circles),
                }
                for group in groups
            ],
        }
        output = String()
        output.data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        self.result_publisher.publish(output)

    def _show(self, binary: np.ndarray, annotated: np.ndarray) -> None:
        if not self.window_created:
            cv2.namedWindow(BINARY_WINDOW, cv2.WINDOW_NORMAL)
            cv2.namedWindow(CIRCLES_WINDOW, cv2.WINDOW_NORMAL)
            self.window_created = True
        cv2.imshow(BINARY_WINDOW, binary)
        cv2.imshow(CIRCLES_WINDOW, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令，正在关闭圆环识别节点。')
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[RingDetectNode] = None
    try:
        node = RingDetectNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('ring_detect 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
