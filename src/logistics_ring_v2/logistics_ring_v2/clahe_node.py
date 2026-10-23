"""Ring V2 pipeline step 3: blurred gray -> CLAHE, publish + debug window.

圆环组识别第二版 · 第三步(CLAHE 限幅): 订阅模糊节点输出, 局部自适应对比度
增强后发布, 供后续 Hough 检圆节点订阅; 同时打开观察窗口方便人工调试。

目的与参数依据(2026-10-02 实拍 640x480 工位板实测):
    现场存在反光(数字 2/3 上方高光)与亮度不均, 被反光压灰的环边缘梯度弱,
    Hough 投票吃不到。CLAHE 局部拉对比度恢复梯度; clipLimit 是"恢复量"与
    "纹理噪声放大"的限幅权衡。实测(tileGridSize=8x8):
    clip 3.0 时环对比度(p95-p5)恢复约 50%, 高光区 90->133;
    clip 4/6 起斜纹被明显拉出而环增益放缓, 故取 clip=3.0。
顺序: 模糊在前抑噪, CLAHE 在后增强, 避免直接放大原始噪声。

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


WINDOW_NAME = 'Ring V2 CLAHE'


class ClaheNode(Node):
    """对模糊后的灰度图做 CLAHE 增强并发布, 附调试观察窗口。"""

    def __init__(self) -> None:
        super().__init__('ring2_clahe')
        self.declare_parameter('input_topic', '/vision/ring2/blur')
        self.declare_parameter('output_topic', '/vision/ring2/clahe')
        self.declare_parameter('clip_limit', 3.0)
        self.declare_parameter('tile_cols', 8)
        self.declare_parameter('tile_rows', 8)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('display', False)
        self.declare_parameter('command_topic', '/vision/ring2/command')
        self.declare_parameter('start_mode', 'run')

        self.clahe = cv2.createCLAHE(
            clipLimit=float(self.get_parameter('clip_limit').value),
            tileGridSize=(
                int(self.get_parameter('tile_cols').value),
                int(self.get_parameter('tile_rows').value),
            ),
        )
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
            'Ring V2 CLAHE 节点已启动: 订阅 %s, 发布 %s, clip=%.1f tile=%dx%d, '
            '门控 %s(%s), 观察窗口 %s。'
            % (input_topic, output_topic, self.clahe.getClipLimit(),
               self.clahe.getTilesGridSize()[0], self.clahe.getTilesGridSize()[1],
               command_topic, 'run' if self.running else 'stop',
               '开' if self.display else '关')
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
            self.get_logger().info('收到退出指令，正在关闭 CLAHE 节点。')
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

        enhanced = self.clahe.apply(image)
        output = self.bridge.cv2_to_imgmsg(enhanced, encoding='mono8')
        output.header = message.header
        self.publisher.publish(output)

        if self.display:
            self._show(enhanced)

    def _show(self, enhanced) -> None:
        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        cv2.imshow(WINDOW_NAME, enhanced)
        self._poll_quit()

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ClaheNode] = None
    try:
        node = ClaheNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('ring2_clahe 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
