"""Ring V2 pipeline step 2: grayscale -> Gaussian blur, publish + debug window.

圆环组识别第二版 · 第二步(模糊): 订阅灰度节点输出, 高斯模糊后发布,
供后续 Hough 检圆节点订阅; 同时打开观察窗口方便人工调试。

参数依据(2026-10-02 实拍 640x480 工位板实测):
    1.5mm 环线宽仅约 1.6px, 模糊是抑噪保边权衡。对 6 档参数离线对比:
    3x3 sigma=1.0 时环内 Canny 边缘点保留 97%, 白板区纹理能量下降 63%;
    5x5 起环开始变软, 7x7 sigma=2.0 边缘只剩 22%(虚线糊死), 故取 3x3 sigma=1.0。

处理门控: 订阅 /vision/ring2/command, run 开始处理 / stop 跳过处理(省 CPU)。
"""

from typing import Optional

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


WINDOW_NAME = 'Ring V2 Blur'


class BlurNode(Node):
    """把灰度图做高斯模糊后发布, 附调试观察窗口。"""

    def __init__(self) -> None:
        super().__init__('ring2_blur')
        self.declare_parameter('input_topic', '/vision/ring2/gray')
        self.declare_parameter('output_topic', '/vision/ring2/blur')
        self.declare_parameter('blur_ksize', 3)
        self.declare_parameter('blur_sigma', 1.0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('display', False)
        self.declare_parameter('command_topic', '/vision/ring2/command')
        self.declare_parameter('start_mode', 'run')

        # OpenCV 要求核为正奇数; 误传偶数时自动进一位, 避免运行期报错。
        ksize = int(self.get_parameter('blur_ksize').value)
        self.blur_ksize = ksize + 1 if ksize % 2 == 0 else ksize
        self.blur_sigma = float(self.get_parameter('blur_sigma').value)
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
            'Ring V2 模糊节点已启动: 订阅 %s, 发布 %s, 核 %dx%d sigma=%.1f, '
            '门控 %s(%s), 观察窗口 %s。'
            % (input_topic, output_topic, self.blur_ksize, self.blur_ksize,
               self.blur_sigma, command_topic,
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
            self.get_logger().info('收到退出指令，正在关闭模糊节点。')
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

        blurred = cv2.GaussianBlur(
            image, (self.blur_ksize, self.blur_ksize), self.blur_sigma
        )
        output = self.bridge.cv2_to_imgmsg(blurred, encoding='mono8')
        output.header = message.header
        self.publisher.publish(output)

        if self.display:
            self._show(blurred)

    def _show(self, blurred) -> None:
        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        cv2.imshow(WINDOW_NAME, blurred)
        self._poll_quit()

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[BlurNode] = None
    try:
        node = BlurNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('ring2_blur 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
