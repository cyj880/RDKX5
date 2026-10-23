"""ROS 2 node for detecting site digits 1, 2, and 3 inside ring ROIs.

主路径：订阅 ring_detect 的圆环组结果，把每组最内环的圆形区域作为
数字识别 ROI（圆外像素清零，环线与弧线被排除），圆内前景整体归一化
后送 HOG+SVM 分类——断裂笔画在圆内重聚，弧线不再误判。

回退：无圆环结果且 fallback_full_frame=true 时，走全图轮廓扫描。
"""

from collections import Counter, deque
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .digit_features import (
    binarize_digit,
    create_hog,
    hog_features,
    normalize_mask,
    svm_class_scores,
)


WINDOW_NAME = 'Logistics Vision Number Detect'
DIGIT_BY_INDEX = (1, 2, 3)  # SVM 类别索引对应训练标签（排序后）


@dataclass(frozen=True)
class RingGroup:
    """圆环检测结果中的一组同心圆。"""

    center: Tuple[int, int]
    radius: int


@dataclass(frozen=True)
class Detection:
    """A recognized digit with its position in pixels."""

    digit: int
    center: Tuple[int, int]
    confidence: float
    box: Tuple[int, int, int, int]
    roi: Optional[Tuple[Tuple[int, int], int]] = None  # (圆心, 半径)，圆环 ROI 模式有效


class NumberDetectNode(Node):
    """Classifies the digit inside each detected ring group."""

    def __init__(self) -> None:
        super().__init__('number_detect')
        self.declare_parameter('input_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('output_topic', '/vision/camera/number_annotated')
        self.declare_parameter('result_topic', '/vision/camera/number_results')
        self.declare_parameter('ring_result_topic', '/vision/camera/ring_results')
        self.declare_parameter('roi_radius_ratio', 0.85)
        self.declare_parameter('fallback_full_frame', False)
        self.declare_parameter(
            'model_path', '~/logistics_vision_data/digit_model/digits_svm.yml'
        )
        self.declare_parameter('display', True)
        self.declare_parameter('stable_frames', 5)
        self.declare_parameter('position_tolerance_px', 8)
        self.declare_parameter('minimum_confidence', 0.6)
        self.declare_parameter('minimum_digit_area', 250)
        self.declare_parameter('maximum_digit_area', 8000)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        # 命令驱动: 与 ring_detect 共用 /vision/region_command, digits 开始识别
        # 发布, dstop 停止; 默认 idle(未通知不发布结果), 调试窗口照常显示
        self.declare_parameter('command_topic', '/vision/region_command')

        self.image_size = (
            int(self.get_parameter('image_width').value),
            int(self.get_parameter('image_height').value),
        )
        model_path = Path(
            str(self.get_parameter('model_path').value)
        ).expanduser()
        if not model_path.is_file():
            raise RuntimeError(
                '未找到数字分类模型: %s。请先运行 ros2 run '
                'logistics_vision_capture digit_trainer 训练生成。' % model_path
            )
        self.svm = cv2.ml.SVM_load(str(model_path))
        self.hog = create_hog()

        self.roi_radius_ratio = float(self.get_parameter('roi_radius_ratio').value)
        self.fallback_full_frame = bool(
            self.get_parameter('fallback_full_frame').value
        )
        self.display = bool(self.get_parameter('display').value)
        self.stable_frames = max(1, int(self.get_parameter('stable_frames').value))
        self.position_tolerance = max(
            0, int(self.get_parameter('position_tolerance_px').value)
        )
        self.minimum_confidence = float(
            self.get_parameter('minimum_confidence').value
        )
        self.minimum_digit_area = int(self.get_parameter('minimum_digit_area').value)
        self.maximum_digit_area = int(self.get_parameter('maximum_digit_area').value)

        self.bridge = CvBridge()
        self.latest_groups: List[RingGroup] = []
        self.window_created = False
        self.size_error_reported = False
        self.history: deque[Tuple[Tuple[int, int, int], ...]] = deque(
            maxlen=self.stable_frames
        )
        self.last_printed: Optional[Tuple[Tuple[int, int, int], ...]] = None

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        result_topic = str(self.get_parameter('result_topic').value)
        ring_topic = str(self.get_parameter('ring_result_topic').value)
        command_topic = str(self.get_parameter('command_topic').value)
        self.mode = 'idle'  # digits→run 开始识别发布, dstop→idle 停止
        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.result_publisher = self.create_publisher(String, result_topic, 10)
        self.create_subscription(String, command_topic, self._command_callback, 10)
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self._image_callback,
            10,
        )
        self.ring_subscription = self.create_subscription(
            String,
            ring_topic,
            self._ring_callback,
            10,
        )
        self.get_logger().info(
            '数字识别节点已启动[idle,等 digits 命令]: 订阅 %s + %s + %s，发布标注图 %s、结果 %s。'
            % (input_topic, ring_topic, command_topic, output_topic, result_topic)
        )

    def _command_callback(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == 'digits':
            self.mode = 'run'
            self.history.clear()
            self.last_printed = None
            self.get_logger().info('进入数字识别模式: 开始发布识别结果。')
        elif command == 'dstop':
            self.mode = 'idle'
            self.get_logger().info('退出数字识别模式: 停止发布识别结果。')
        else:
            self.get_logger().warn('无法识别命令 "%s"。可用: digits / dstop' % message.data)

    def _ring_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            groups = [
                RingGroup(
                    (int(group['center_u']), int(group['center_v'])),
                    int(group['innermost_radius_px']),
                )
                for group in payload.get('groups', [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.get_logger().warn('圆环结果 JSON 解析失败，沿用上一次结果。')
            return
        self.latest_groups = groups

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
                    '输入图像分辨率为 %dx%d，预期 %dx%d，当前帧已跳过。'
                    % (width, height, *self.image_size)
                )
                self.size_error_reported = True
            return
        self.size_error_reported = False

        detections = self._detect(image)
        annotated = self._annotate(image, detections)
        output = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        output.header = message.header
        self.publisher.publish(output)
        if self.mode == 'run':
            self._publish_results(detections)
            self._update_stable_result(detections)

        if self.display:
            self._show(annotated)

    def _detect(self, image: np.ndarray) -> List[Detection]:
        if self.latest_groups:
            return self._detect_in_rings(image)
        if self.fallback_full_frame:
            return self._detect_full_frame(image)
        return []

    def _detect_in_rings(self, image: np.ndarray) -> List[Detection]:
        """把每个圆环组的最内环圆形区域作为 ROI 做数字分类。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_height, image_width = gray.shape[:2]
        detections: List[Detection] = []
        for group in self.latest_groups:
            center_x, center_y = group.center
            radius = int(group.radius * self.roi_radius_ratio)
            if radius < 8:
                continue
            # 方形外接框只是取图手段，越界部分裁剪掉。
            x0 = max(0, center_x - radius)
            y0 = max(0, center_y - radius)
            x1 = min(image_width, center_x + radius)
            y1 = min(image_height, center_y + radius)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            roi_binary = binarize_digit(gray[y0:y1, x0:x1])
            # 实心圆盘掩膜：只保留内环圆内的前景。
            disk = np.zeros(roi_binary.shape, dtype=np.uint8)
            cv2.circle(disk, (center_x - x0, center_y - y0), radius, 255, -1)
            roi_binary = cv2.bitwise_and(roi_binary, disk)

            normalized = normalize_mask(roi_binary)
            if normalized is None:
                continue
            scores = svm_class_scores(self.svm, hog_features(self.hog, normalized))
            index = int(np.argmax(scores))
            confidence = float(scores[index])
            if confidence < self.minimum_confidence:
                continue
            detections.append(
                Detection(
                    DIGIT_BY_INDEX[index],
                    (center_x, center_y),
                    confidence,
                    (x0, y0, x1 - x0, y1 - y0),
                    ((center_x, center_y), radius),
                )
            )
        return self._deduplicate(detections)

    def _detect_full_frame(self, image: np.ndarray) -> List[Detection]:
        """回退路径：全图轮廓扫描 + 逐候选分类。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = binarize_digit(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections: List[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.minimum_digit_area or area > self.maximum_digit_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if not self._is_valid_digit_box(x, y, width, height):
                continue
            roi_mask = normalize_mask(binarize_digit(gray[y:y + height, x:x + width]))
            if roi_mask is None:
                continue
            scores = svm_class_scores(self.svm, hog_features(self.hog, roi_mask))
            index = int(np.argmax(scores))
            confidence = float(scores[index])
            if confidence < self.minimum_confidence:
                continue
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            center = (
                int(moments['m10'] / moments['m00']),
                int(moments['m01'] / moments['m00']),
            )
            detections.append(
                Detection(DIGIT_BY_INDEX[index], center, confidence, (x, y, width, height))
            )

        return self._deduplicate(detections)

    def _is_valid_digit_box(self, x: int, y: int, width: int, height: int) -> bool:
        margin = 8
        if x <= margin or y <= margin:
            return False
        if x + width >= self.image_size[0] - margin:
            return False
        if y + height >= self.image_size[1] - margin:
            return False
        ratio = width / float(height)
        return 0.15 <= ratio <= 1.25 and height >= 25

    @staticmethod
    def _deduplicate(detections: List[Detection]) -> List[Detection]:
        selected: dict = {}
        for detection in detections:
            old = selected.get(detection.digit)
            if old is None or detection.confidence > old.confidence:
                selected[detection.digit] = detection
        return sorted(selected.values(), key=lambda detection: detection.center[0])

    def _annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated = image.copy()
        for detection in detections:
            if detection.roi is not None:
                (center_x, center_y), radius = detection.roi
                cv2.circle(annotated, (center_x, center_y), radius, (0, 255, 0), 2)
            else:
                x, y, width, height = detection.box
                cv2.rectangle(
                    annotated, (x, y), (x + width, y + height), (0, 255, 0), 2
                )
            center_x, center_y = detection.center
            cv2.drawMarker(
                annotated,
                (center_x, center_y),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                16,
                2,
            )
            label = '%d (%d,%d) %.2f' % (
                detection.digit,
                center_x,
                center_y,
                detection.confidence,
            )
            cv2.putText(
                annotated,
                label,
                (detection.box[0], max(22, detection.box[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _publish_results(self, detections: List[Detection]) -> None:
        payload = {
            'digits': [
                {
                    'digit': detection.digit,
                    'center_u': detection.center[0],
                    'center_v': detection.center[1],
                    'confidence': round(detection.confidence, 3),
                }
                for detection in detections
            ],
        }
        output = String()
        output.data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        self.result_publisher.publish(output)

    def _update_stable_result(self, detections: List[Detection]) -> None:
        signature = tuple(
            sorted(
                (
                    detection.digit,
                    detection.center[0] // max(1, self.position_tolerance),
                    detection.center[1] // max(1, self.position_tolerance),
                )
                for detection in detections
            )
        )
        self.history.append(signature)
        if len(self.history) < self.stable_frames:
            return
        stable_signature, count = Counter(self.history).most_common(1)[0]
        if count != self.stable_frames or stable_signature == self.last_printed:
            return
        self.last_printed = stable_signature
        result = ', '.join(
            '%d@(%d,%d)' % (
                detection.digit,
                detection.center[0],
                detection.center[1],
            )
            for detection in detections
        )
        self.get_logger().info('稳定识别结果: %s' % (result or '未识别到完整数字'))

    def _show(self, image: np.ndarray) -> None:
        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令，正在关闭数字识别节点。')
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[NumberDetectNode] = None
    try:
        node = NumberDetectNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('number_detect 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
