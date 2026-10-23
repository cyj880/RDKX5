"""Ring V2 pipeline step 4: CLAHE image -> HoughCircles, publish + debug window.

圆环组识别第二版 · 第四步(霍夫检圆): 订阅 CLAHE 节点输出, HoughCircles 提取
候选圆并发布 JSON, 同时在观察窗口画出全部检出圆方便人工调试。

定位目标与参数依据:
    本节点是"候选生成器"(低门槛捞), 圆的身份与筛选由后续几何后验节点完成,
    因此 param2 取低值让虚线外圈也能进候选, 代价是混入假圆——由后验筛兜底。
    初参来自 2026-10-02 640p 实拍实验: 组距约 167px, 内圈实线半径约 28px,
    外圈虚线半径约 50px; dp=1, min_dist=60, min_radius=20, max_radius=60。
    内圈 phi1/phi2 半径差仅约 2.6px, 半径累加器分不开属物理限制:
    两圈同心边缘加倍投票使组心更稳, 后验聚类只取组心, 不受影响。

处理门控: 订阅 /vision/ring2/command, run 开始处理 / stop 跳过处理(省 CPU)。
"""

import json
from typing import List, Optional

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from .ring_processing import detect_circles


WINDOW_NAME = 'Ring V2 Hough'


class HoughNode(Node):
    """对增强图做霍夫圆检测, 发布候选圆 JSON 并画出全部检出圆。"""

    def __init__(self) -> None:
        super().__init__('ring2_hough')
        self.declare_parameter('input_topic', '/vision/ring2/clahe')
        self.declare_parameter('output_topic', '/vision/ring2/circles')
        self.declare_parameter('dp', 1.0)
        self.declare_parameter('min_dist', 60.0)
        self.declare_parameter('param1', 100.0)
        self.declare_parameter('param2', 0.4)
        self.declare_parameter('min_radius', 20)
        self.declare_parameter('max_radius', 60)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('display', False)
        self.declare_parameter('command_topic', '/vision/ring2/command')
        self.declare_parameter('start_mode', 'run')

        self.dp = float(self.get_parameter('dp').value)
        self.min_dist = float(self.get_parameter('min_dist').value)
        self.param1 = float(self.get_parameter('param1').value)
        self.param2 = float(self.get_parameter('param2').value)
        self.min_radius = int(self.get_parameter('min_radius').value)
        self.max_radius = int(self.get_parameter('max_radius').value)
        self.image_size = (
            int(self.get_parameter('image_width').value),
            int(self.get_parameter('image_height').value),
        )
        self.display = bool(self.get_parameter('display').value)
        # 处理门控: run 才处理, stop 跳过(进程常驻, 省算力)
        self.running = str(self.get_parameter('start_mode').value).lower() == 'run'
        self.window_created = False
        self.size_error_reported = False

        self.bridge = CvBridge()
        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        command_topic = str(self.get_parameter('command_topic').value)
        self.publisher = self.create_publisher(String, output_topic, 10)
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self._image_callback,
            10,
        )
        self.create_subscription(String, command_topic, self._command_callback, 10)
        self.get_logger().info(
            'Ring V2 霍夫节点已启动: 订阅 %s, 发布 %s, dp=%.1f minDist=%.0f '
            'param1=%.0f param2=%.2f 半径 %d~%d, 门控 %s(%s), 观察窗口 %s。'
            % (input_topic, output_topic, self.dp, self.min_dist, self.param1,
               self.param2, self.min_radius, self.max_radius, command_topic,
               'run' if self.running else 'stop', '开' if self.display else '关')
        )

    def _command_callback(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == 'run' and not self.running:
            self.running = True
            self.get_logger().info('收到 run: 开始处理。')
        elif command == 'stop' and self.running:
            self.running = False
            self.get_logger().info('收到 stop: 跳过处理, 节省算力。')
        elif command not in ('run', 'stop'):
            self.get_logger().warn('无法识别命令 "%s"。可用: run / stop' % message.data)

    def _poll_quit(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令，正在关闭霍夫节点。')
            rclpy.shutdown()

    def _image_callback(self, message: Image) -> None:
        if not self.running:
            # stop 门控: 跳过处理省 CPU; 已开的窗口保持响应(q 可退出)
            if self.display and self.window_created:
                self._poll_quit()
            return
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='mono8')
        except CvBridgeError as error:
            self.get_logger().error('无法将 ROS 图像转换为 mono8: %s' % error)
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

        circles = self._detect(image)
        self._publish(circles)

        if self.display:
            self._show(image, circles)

    def _detect(self, gray: np.ndarray) -> List[tuple]:
        """霍夫检圆, 返回 [(x, y, r), ...]。"""
        return detect_circles(gray, self.dp, self.min_dist, self.param1, self.param2,
                              self.min_radius, self.max_radius)

    def _publish(self, circles: List[tuple]) -> None:
        payload = {
            'circles': [
                {'x': round(x, 1), 'y': round(y, 1), 'r': round(r, 1)}
                for x, y, r in circles
            ],
        }
        output = String()
        output.data = json.dumps(payload, separators=(',', ':'))
        self.publisher.publish(output)

    def _show(self, gray: np.ndarray, circles: List[tuple]) -> None:
        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, r in circles:
            cv2.circle(annotated, (int(round(x)), int(round(y))),
                       int(round(r)), (0, 255, 0), 2)
            cv2.drawMarker(annotated, (int(round(x)), int(round(y))),
                           (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(annotated, '%.0f' % r,
                        (int(round(x)) + 6, int(round(y)) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.putText(annotated, 'circles: %d' % len(circles), (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, annotated)
        self._poll_quit()

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[HoughNode] = None
    try:
        node = HoughNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('ring2_hough 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
