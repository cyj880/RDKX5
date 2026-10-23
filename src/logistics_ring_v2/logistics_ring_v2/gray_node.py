"""Ring V2 pipeline step 1: undistorted BGR -> grayscale, publish + debug window.

圆环组识别第二版 · 第一步(灰度化): 订阅旧包去畸变节点的输出, 转灰度后发布,
供后续 Hough 检圆节点订阅, 同时打开观察窗口方便人工调试。

灰度化是固定加权 (Y = 0.299R + 0.587G + 0.114B), 没有任何可调参数;
模糊等后续处理由后续节点承担, 本节点保持零处理参数。

处理门控: 订阅 /vision/ring2/command, run 开始处理 / stop 跳过处理(省 CPU)。
常驻+门控架构下, 节点全程存活, 主控到工位板才发 run, 离开发 stop。
"""

from typing import Optional

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


WINDOW_NAME = 'Ring V2 Gray'


class GrayNode(Node):
    """把去畸变图转为灰度并发布, 附调试观察窗口。"""

    def __init__(self) -> None:
        super().__init__('ring2_gray')
        self.declare_parameter('input_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('output_topic', '/vision/ring2/gray')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('display', False)
        self.declare_parameter('command_topic', '/vision/ring2/command')
        self.declare_parameter('start_mode', 'run')

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
        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self._image_callback,
            10,
        )
        self.create_subscription(String, command_topic, self._command_callback, 10)
        self.get_logger().info(
            'Ring V2 灰度节点已启动: 订阅 %s, 发布 %s, 门控 %s(%s), 观察窗口 %s。'
            % (input_topic, output_topic, command_topic,
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
            self.get_logger().info('收到退出指令，正在关闭灰度节点。')
            rclpy.shutdown()

    def _image_callback(self, message: Image) -> None:
        if not self.running:
            # stop 门控: 跳过处理省 CPU; 已开的窗口保持响应(q 可退出)
            if self.display and self.window_created:
                self._poll_quit()
            return
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

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output = self.bridge.cv2_to_imgmsg(gray, encoding='mono8')
        output.header = message.header
        self.publisher.publish(output)

        if self.display:
            self._show(gray)

    def _show(self, gray) -> None:
        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        cv2.imshow(WINDOW_NAME, gray)
        self._poll_quit()

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[GrayNode] = None
    try:
        node = GrayNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('ring2_gray 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
