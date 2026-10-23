"""ROS adapter for low-latency ring recognition; camera and other modes are untouched."""
import json
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .ring_pipeline import LatestFrameWorker, RingPipeline
from .ring_processing import RingValidation, annotate_ring


class RingPipelineNode(Node):
    def __init__(self):
        super().__init__('ring2_pipeline')
        defaults = dict(input_topic='/vision/camera/image_undistorted',
                        output_topic='/vision/ring2/sites',
                        annotated_topic='/vision/ring2/annotated',
                        command_topic='/vision/ring2/command', start_mode='run',
                        display=False, display_debug=False, smooth_alpha=0.8,
                        roi_enabled=True, roi_margin_px=32, roi_full_scan_interval=15,
                        fast_enabled=True, opencv_threads=2,
                        detector_backend='reference',
                        model_path='~/logistics_vision_data/digit_model/digits_svm.yml')
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        param = lambda name: self.get_parameter(name).value
        threads=int(param('opencv_threads'))
        if threads not in (1,2,4):
            raise ValueError('opencv_threads must be 1, 2 or 4')
        # This executable has its own process; do not alter other detector nodes.
        cv2.setNumThreads(threads)
        self.bridge = CvBridge()
        self.display = bool(param('display'))
        self.display_debug = bool(param('display_debug'))
        self.preview = None
        self.processing_errors = 0
        self.last_processing_error = ''
        self.last_search_mode = 'waiting'
        self.last_stage_ms = {}
        self.windows_created = False
        self.publisher = self.create_publisher(String, str(param('output_topic')), 1)
        self.annotated_publisher = self.create_publisher(Image, str(param('annotated_topic')), 1)
        validator = RingValidation(float(param('smooth_alpha')))
        validator.load_digit_model(str(param('model_path')), self.get_logger())
        self.pipeline = RingPipeline(validator, roi_enabled=bool(param('roi_enabled')),
                                     roi_margin_px=int(param('roi_margin_px')),
                                     roi_full_scan_interval=int(param('roi_full_scan_interval')),
                                     fast_enabled=bool(param('fast_enabled')),
                                     detector_backend=str(param('detector_backend')))
        self.worker = LatestFrameWorker(self._process, self._deliver,
                                        self.pipeline.reset, self._processing_error)
        self.running = str(param('start_mode')).lower() == 'run'
        self.last_stats_time = time.monotonic()
        self.last_stats = self.worker.snapshot()
        # Compatible with reliable camera publishers, but never accumulate DDS history.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, str(param('input_topic')), self._image, qos)
        self.create_subscription(String, str(param('command_topic')), self._command, 10)
        self.create_timer(5.0, self._stats)
        if self.display or self.display_debug:
            self.create_timer(0.03, self._show)
        if self.running:
            self.worker.start()
        self.get_logger().info('圆环backend=%s，OpenCV线程=%d。' %
                               (self.pipeline.detector_backend,threads))

    def _image(self, message):
        ingress_ns = time.monotonic_ns()
        if self.running:
            self.worker.submit(message, ingress_ns)

    def _command(self, message):
        command = message.data.strip().lower()
        if command == 'run' and not self.running:
            self.worker.start()
            self.running = True
        elif command == 'stop':
            self.running = False
            self.worker.stop()
            self.preview = None
        elif command != 'run':
            self.get_logger().warning('圆环命令只支持run/stop: %s' % message.data)

    def _process(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        return self.pipeline.process(frame)

    def _deliver(self, source, output, sequence, ingress_ns, process_ms):
        result, gray, rejected = output
        self.last_search_mode = result.get('search_mode', 'unknown')
        self.last_stage_ms = result.get('stage_ms', {})
        result.update(frame_sequence=sequence, ingress_monotonic_ns=ingress_ns,
                      processing_ms=round(process_ms, 3))
        self.publisher.publish(String(data=json.dumps(result, separators=(',', ':'))))
        if self.display or self.display_debug or self.annotated_publisher.get_subscription_count():
            annotated = annotate_ring(gray, result['status'], result['sites'],
                                      result['station_yaw_deg'], rejected)
            message = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            message.header = source.header
            if self.annotated_publisher.get_subscription_count():
                self.annotated_publisher.publish(message)
            if self.display or self.display_debug:
                self.preview = (annotated, gray)

    def _processing_error(self, error):
        # No fresh target is emitted; bridge expiry sends the existing empty frame.
        self.processing_errors += 1
        self.last_processing_error = str(error)

    def _show(self):
        preview, self.preview = self.preview, None
        if preview is not None:
            if self.display:
                cv2.imshow('Ring V2 Final', preview[0])
            if self.display_debug:
                cv2.imshow('Ring V2 CLAHE', preview[1])
            self.windows_created = True
        if self.windows_created and cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
            rclpy.shutdown()

    def _stats(self):
        now = time.monotonic()
        stats = self.worker.snapshot(reset_window=True)
        dt = max(now-self.last_stats_time, 0.001)
        self.get_logger().info(
            'RING PERF input=%.1fHz process=%.1fHz overwritten=%d discarded=%d process_p95=%.1fms search=%s stage_ms=%s' %
            ((stats['received']-self.last_stats['received'])/dt,
             (stats['processed']-self.last_stats['processed'])/dt,
             stats['overwritten']-self.last_stats['overwritten'],
             stats['discarded']-self.last_stats['discarded'], stats['process_p95_ms'], self.last_search_mode,
             self.last_stage_ms))
        self.last_stats, self.last_stats_time = stats, now
        if self.processing_errors:
            self.get_logger().error('圆环处理失败%d次，已清空跟踪，最近错误: %s' %
                                    (self.processing_errors, self.last_processing_error))
            self.processing_errors = 0

    def destroy_node(self):
        self.worker.close()
        if self.windows_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RingPipelineNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
