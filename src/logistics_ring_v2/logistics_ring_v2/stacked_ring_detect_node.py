"""ROS transport and mode gate for the stacked-station tracker."""

import json

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .stacked_ring_algorithm import StackedTracker


class StackedRingDetectNode(Node):
    def __init__(self):
        super().__init__('stacked_ring_detect')
        defaults = {'outer_diameter_mm': 95., 'line_offset_mm': 75.,
                    'expected_spacing_px': 152.1, 'stable_frames': 5,
                    'center_jitter_px': 5., 'angle_jitter_deg': 1.,
                    'line_band_px': 18, 'line_min_span_px': 120,
                    'middle_radius_min': 20, 'middle_radius_max': 60,
                    'line_disagreement_deg': 2., 'triple_confirm_frames': 8}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.declare_parameter('image_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('command_topic', '/vision/gated/stacked/command')
        self.declare_parameter('result_topic', '/vision/stacked/results')
        self.declare_parameter('annotated_topic', '/vision/stacked/annotated')
        self.declare_parameter('start_mode', 'stop')
        self.declare_parameter('display', False)
        params = {name: self.get_parameter(name).value for name in defaults}
        self.tracker = StackedTracker(expected_spacing=params['expected_spacing_px'],
                                     outer_diameter=params['outer_diameter_mm'],
                                     line_offset=params['line_offset_mm'],
                                     stable_frames=params['stable_frames'],
                                     center_jitter=params['center_jitter_px'],
                                     angle_jitter=params['angle_jitter_deg'],
                                     line_band=params['line_band_px'],
                                     min_line_span=params['line_min_span_px'],
                                     min_radius=params['middle_radius_min'],
                                     max_radius=params['middle_radius_max'],
                                     line_disagreement=params['line_disagreement_deg'],
                                     confirm_frames=params['triple_confirm_frames'])
        self.running = str(self.get_parameter('start_mode').value).lower() == 'run'
        self.display = bool(self.get_parameter('display').value)
        self.window = False
        self.bridge = CvBridge()
        self.result_pub = self.create_publisher(String, self.get_parameter('result_topic').value, 10)
        self.annotated_pub = self.create_publisher(Image, self.get_parameter('annotated_topic').value, 1)
        self.create_subscription(Image, self.get_parameter('image_topic').value, self._image, 1)
        self.create_subscription(String, self.get_parameter('command_topic').value, self._command, 10)

    def _command(self, message):
        command = message.data.strip().lower()
        if command == 'stop' or command == 'run' and not self.running:
            self.tracker.reset()
        if command in ('run', 'stop'):
            self.running = command == 'run'

    def _image(self, message):
        if not self.running:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().warn('码垛图像转换失败: %s' % error)
            return
        result, annotated = self.tracker.process(frame)
        result.update(mode='stacked', gate='run')
        self.result_pub.publish(String(data=json.dumps(result, ensure_ascii=False, separators=(',', ':'))))
        output = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        output.header = message.header
        self.annotated_pub.publish(output)
        if self.display:
            if not self.window:
                cv2.namedWindow('Stacked Ring', cv2.WINDOW_NORMAL)
                self.window = True
            cv2.imshow('Stacked Ring', annotated)
            cv2.waitKey(1)

    def destroy_node(self):
        if self.window:
            cv2.destroyWindow('Stacked Ring')
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StackedRingDetectNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
