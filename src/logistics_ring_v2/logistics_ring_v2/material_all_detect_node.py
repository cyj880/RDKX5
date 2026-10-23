"""Material all-recognize node: 全物料识别(六色全部输出, 无抓取目标)。

物料识别拆分后的并列节点之一(与 turntable/stacked 同级, 接入统一门控):
命令只有 run/stop, 图像进来跑共享管线(material_pipeline.MaterialPipeline),
输出全部置信度达标的物料坐标。调试/无任务码时用; 逐个抓取请用
material_track 节点。

话题(经 vision_all.launch 重映射后命令为 /vision/gated/material_all/command):
    订阅 /vision/camera/image_undistorted
    发布 /vision/material_all/results  (JSON: mode/stationary/motion_px/materials)
    发布 /vision/material_all/annotated (标注图)
结果字段: materials[{color, area, confidence}]。坐标只用于标注图显示，
不作为全物料调试节点的结果接口输出；抓取坐标由 material_track 节点提供。
color=0(kNN 未加载)或置信度不足的轨迹只画窗口不进 JSON。
"""

import json
import time
from typing import Optional

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .material_pipeline import (
    attach_parameter_hot_tune,
    declare_pipeline_parameters,
    pipeline_from_node,
    stamp_seconds,
)


WINDOW_NAME = 'Material All Detect'


class MaterialAllDetectNode(Node):
    """全物料识别节点: 六色全部输出, 不生成抓取目标。"""

    def __init__(self) -> None:
        super().__init__('material_all_detect')
        self.declare_parameter('input_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('output_topic', '/vision/material_all/annotated')
        self.declare_parameter('result_topic', '/vision/material_all/results')
        self.declare_parameter('command_topic', '/vision/material_all/command')
        self.declare_parameter('display', False)
        self.declare_parameter('display_debug', False)
        self.declare_parameter('start_mode', 'stop')
        declare_pipeline_parameters(self)

        self.display = bool(self.get_parameter('display').value)
        self.display_debug = bool(self.get_parameter('display_debug').value)
        # 门控: run 才处理, stop 跳过(进程常驻, 省算力); 与 turntable 同款
        self.running = str(self.get_parameter('start_mode').value).lower() == 'run'
        self.pipeline = pipeline_from_node(self)
        attach_parameter_hot_tune(self, self.pipeline)
        for detail in filter(None, (self.pipeline.model_detail,)):
            self.get_logger().warn(detail)

        self.bridge = CvBridge()
        self.window_created = False
        self.mask_window_created = False
        self.size_error_reported = False

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        result_topic = str(self.get_parameter('result_topic').value)
        command_topic = str(self.get_parameter('command_topic').value)
        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.result_publisher = self.create_publisher(String, result_topic, 10)
        self.create_subscription(String, command_topic, self._command_callback, 10)
        self.create_subscription(Image, input_topic, self._image_callback, 10)
        self.get_logger().info(
            '全物料识别节点已启动: 图像 %s, 命令 %s, 结果 %s, 门控 %s。'
            % (input_topic, command_topic, result_topic,
               'run' if self.running else 'stop'))

    def _command_callback(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == 'run' and not self.running:
            self.running = True
            self.pipeline.reset()  # 新会话: 重新累计投票与停稳
            self.get_logger().info('收到 run: 开始全物料识别。')
        elif command == 'stop' and self.running:
            self.running = False
            self.pipeline.reset()
            self.get_logger().info('收到 stop: 跳过处理, 节省算力。')
        elif command not in ('run', 'stop', ''):
            self.get_logger().warn('无法识别命令 "%s"。可用: run / stop' % message.data)

    def _image_callback(self, message: Image) -> None:
        if not self.running:
            # stop 门控: 跳过处理省 CPU; 本地窗口保持响应(q 可退出)
            if self.display and self.window_created:
                self._poll_quit()
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error('无法转换图像: %s' % error)
            return
        height, width = frame.shape[:2]
        if (width, height) != self.pipeline.image_size:
            if not self.size_error_reported:
                self.get_logger().error(
                    '输入图像 %dx%d 与预期 %dx%d 不符，跳过该帧。'
                    % (width, height, *self.pipeline.image_size)
                )
                self.size_error_reported = True
            return
        self.size_error_reported = False

        # 速度计算需要时间戳: header 零值(合成源)时回退单调时钟
        timestamp = stamp_seconds(message) or time.monotonic()
        frame_result = self.pipeline.process(frame, timestamp)
        # 只输出当帧新鲜匹配(age==0)的轨迹, 漏检旧轨迹不进结果(交接文档§5 风险)
        visible = [item for item in frame_result.decided if item[0].age == 0]
        payload = self._fill(frame_result, visible)
        annotated = self.pipeline.annotate(frame, visible, payload)
        output = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        output.header = message.header
        self.publisher.publish(output)
        result = String()
        result.data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        self.result_publisher.publish(result)
        if self.display:
            self._show(annotated)
        if self.display_debug:
            self._show_layer1(frame, frame_result)

    def _fill(self, frame_result, visible) -> dict:
        """全识别调试输出: 六色确认项按 color 排序，不输出坐标和 target。"""
        payload = {'mode': 'material_all', 'gate': 'run',
                   'stationary': frame_result.stationary}
        if frame_result.motion_speed is not None:
            payload['motion_speed_px_s'] = round(frame_result.motion_speed, 1)
        materials = []
        for track, color, confidence in visible:
            if color == 0 or confidence < self.pipeline.min_confidence:
                continue  # 未确认色(kNN 未加载/低置信)只画窗口不进 JSON
            materials.append({
                'color': color,
                'area': track.area,
                'confidence': round(confidence, 3),
            })
        materials.sort(key=lambda m: m['color'])
        payload['materials'] = materials
        return payload

    def _poll_quit(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令。')
            rclpy.shutdown()

    def _show(self, annotated) -> None:
        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        cv2.imshow(WINDOW_NAME, annotated)
        self._poll_quit()

    def _show_layer1(self, frame, frame_result) -> None:
        """层1 调试窗口: 绿=彩色掩膜命中, 蓝=黑料掩膜命中。"""
        if frame_result.mask_color is None or frame_result.mask_dark is None:
            return
        vis = frame.copy()
        overlay = vis.copy()
        overlay[frame_result.mask_color > 0] = (0, 255, 0)
        overlay[frame_result.mask_dark > 0] = (255, 0, 0)
        vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)
        cv2.putText(
            vis, 'Layer1: GREEN=color mask  BLUE=dark mask',
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME + ' Layer1', vis)
        self._poll_quit()

    def destroy_node(self) -> bool:
        if self.window_created or self.mask_window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[MaterialAllDetectNode] = None
    try:
        node = MaterialAllDetectNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('material_all_detect 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
