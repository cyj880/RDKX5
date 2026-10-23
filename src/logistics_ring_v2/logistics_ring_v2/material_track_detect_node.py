"""Material track node: 单目标追踪 + 抓取准备(停稳状态机 + 转盘区域判定)。

物料识别拆分后的并列节点之一: 逐个抓取的工作模式。命令 run/stop, 另支持
`track <色>` 在运行中覆盖目标颜色(同时复位会话)。目标颜色默认来自 ROS
参数 track_color(默认 5=黑), launch/param set 可改。

停稳状态机(借鉴 cyqm material_pickup_detector, 单位 px/s):
  速度 = 各匹配轨迹位移/帧间隔 取最大值; 速度≥moving_speed_px_s 连续
  motion_confirm_frames 帧 → 确认转盘在转; 之后速度≤stillness_speed_px_s
  连续 stillness_frames 帧 → stationary=true, 超阈立即 false;
  从未见运动且持续低速满 initial_still_frames 帧 → 直接判停稳(车到时
  转盘恰好停着/未启动的旁路, 防死锁)。

转盘区域判定(解决"转盘没识别出来就不能抓"的死锁):
  运动时按帧间隔检测，停稳后仍定时复核；缓存超时失效。
  in_turntable 三值:
      true  = 目标原始质心在缓存圆内
      false = 明确在圆外 → grasp_ready=false(防抓错)
      null  = 没有有效转盘圆或区域判定关闭
  自动测距基于垂直向下相机、水平转盘、直径300mm; 使用去畸变后内参。
  无有效测距时底心坐标与距离输出null, 不生成抓取target。

话题(经 vision_all.launch 重映射后命令为 /vision/gated/material_track/command):
    订阅 /vision/camera/image_undistorted
    发布 /vision/material_track/results
    发布 /vision/material_track/annotated
"""

import json
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .material_features import color_abbr
from .material_pipeline import (
    attach_parameter_hot_tune,
    declare_pipeline_parameters,
    pipeline_from_node,
    stamp_seconds,
)


WINDOW_NAME = 'Material Track Detect'


class MaterialTrackDetectNode(Node):
    """单目标追踪节点: 停稳+转盘区域判定通过后输出目标色夹取底心。"""

    def __init__(self) -> None:
        super().__init__('material_track_detect')
        self.declare_parameter('input_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('output_topic', '/vision/material_track/annotated')
        self.declare_parameter('result_topic', '/vision/material_track/results')
        self.declare_parameter('command_topic', '/vision/material_track/command')
        self.declare_parameter('display', False)
        self.declare_parameter('display_debug', False)
        self.declare_parameter('start_mode', 'stop')
        self.declare_parameter('track_color', 5)
        # 转盘区域判定: 半径范围按当前相机高度折算(Z=200mm → 转盘半径约218px)
        self.declare_parameter('turntable_check', True)
        self.declare_parameter('turntable_min_radius_px', 150)
        self.declare_parameter('turntable_max_radius_px', 280)
        self.declare_parameter('turntable_interval', 3)
        self.declare_parameter('turntable_recheck_interval_s', 0.5)
        self.declare_parameter('turntable_cache_ttl_s', 1.5)
        declare_pipeline_parameters(self)

        self.display = bool(self.get_parameter('display').value)
        self.display_debug = bool(self.get_parameter('display_debug').value)
        self.running = str(self.get_parameter('start_mode').value).lower() == 'run'
        self.track_color: Optional[int] = int(self.get_parameter('track_color').value)
        self.turntable_check = bool(self.get_parameter('turntable_check').value)
        self.turntable_min_radius = int(self.get_parameter('turntable_min_radius_px').value)
        self.turntable_max_radius = int(self.get_parameter('turntable_max_radius_px').value)
        self.turntable_interval = max(1, int(self.get_parameter('turntable_interval').value))
        self.turntable_recheck_interval = float(self.get_parameter('turntable_recheck_interval_s').value)
        self.turntable_cache_ttl = float(self.get_parameter('turntable_cache_ttl_s').value)
        self.pipeline = pipeline_from_node(self)
        self.measurement_matrix = self._load_measurement_matrix()
        if self.measurement_matrix is not None:
            self.pipeline.principal_point = (float(self.measurement_matrix[0, 2]),
                                             float(self.measurement_matrix[1, 2]))
        attach_parameter_hot_tune(self, self.pipeline)
        # 停稳速度只统计目标色轨迹: 影子/杂物块的关联churn不污染停稳判定
        self.pipeline.stillness_track_color = self.track_color
        # 节点本地热调: 转盘判定参数(不属于共享管线, 现场按视野/盘径调)
        self._node_tunable = {
            'turntable_min_radius_px': ('turntable_min_radius', int, 50, 400),
            'turntable_max_radius_px': ('turntable_max_radius', int, 100, 500),
            'turntable_interval': ('turntable_interval', int, 1, 30),
            'turntable_recheck_interval_s': ('turntable_recheck_interval', float, 0.1, 10.0),
            'turntable_cache_ttl_s': ('turntable_cache_ttl', float, 0.1, 30.0),
        }

        def _on_node_parameters(parameters):
            from rcl_interfaces.msg import SetParametersResult
            for parameter in parameters:
                entry = self._node_tunable.get(parameter.name)
                if entry is None:
                    continue
                attr_name, expected_type, low, high = entry
                value = parameter.value
                if not isinstance(value, (int, float)) or isinstance(value, bool) or (
                        expected_type is int and not isinstance(value, int)):
                    return SetParametersResult(
                        successful=False, reason='%s 类型不正确' % parameter.name)
                value = expected_type(value)
                if not low <= value <= high:
                    return SetParametersResult(
                        successful=False,
                        reason='%s 超出范围 [%s, %s]' % (parameter.name, low, high))
                if (parameter.name == 'turntable_min_radius_px'
                        and value >= self.turntable_max_radius):
                    return SetParametersResult(
                        successful=False, reason='最小半径必须小于最大半径')
                if (parameter.name == 'turntable_max_radius_px'
                        and value <= self.turntable_min_radius):
                    return SetParametersResult(
                        successful=False, reason='最大半径必须大于最小半径')
                setattr(self, attr_name, value)
                self.get_logger().info('参数 %s 热更新为 %s' % (parameter.name, value))
            return SetParametersResult(successful=True)

        self.add_on_set_parameters_callback(_on_node_parameters)
        for detail in filter(None, (self.pipeline.model_detail,)):
            self.get_logger().warn(detail)
        self.last_target: Optional[Tuple[float, float]] = None
        # 转盘圆缓存: (center_u, center_v, radius_px), 定时复核并按TTL失效。
        self.turntable_cache: Optional[Tuple[int, int, float]] = None
        self._turntable_last_seen = None
        self._turntable_last_check = None
        self._turntable_seen = False
        self._frame_count = 0

        self.bridge = CvBridge()
        self.window_created = False
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
            '目标追踪节点已启动: 图像 %s, 命令 %s, 结果 %s, 目标色 %s, '
            '转盘判定 %s, 门控 %s。'
            % (input_topic, command_topic, result_topic, self.track_color,
               '开' if self.turntable_check else '关',
               'run' if self.running else 'stop'))

    def _command_callback(self, message: String) -> None:
        tokens = message.data.strip().lower().split()
        if not tokens:
            return
        command = tokens[0]
        if command == 'run' and len(tokens) == 1 and not self.running:
            self.running = True
            self.pipeline.reset()
            self.pipeline.stillness_track_color = self.track_color
            self.last_target = None
            self.turntable_cache = None
            self._reset_turntable()
            self.get_logger().info(
                '收到 run: 开始追踪(目标色 %s)。' % self.track_color)
        elif command == 'stop' and len(tokens) == 1:
            self.running = False
            self.pipeline.reset()
            self.last_target = None
            self.turntable_cache = None
            self._reset_turntable()
            self.get_logger().info('收到 stop: 跳过处理, 节省算力。')
        elif command == 'track' and len(tokens) == 2:
            try:
                color = int(tokens[1])
            except ValueError:
                self.get_logger().warn('track 参数必须是数字: %s' % message.data)
                return
            if not 1 <= color <= 6:
                self.get_logger().warn('颜色编号必须在 1~6: %s' % message.data)
                return
            self.track_color = color
            self.pipeline.reset()  # 换目标色 = 新会话, 重新确认停稳
            self.pipeline.stillness_track_color = color
            self.last_target = None
            self._reset_turntable()
            self.running = True
            self.get_logger().info(
                '追踪目标更新: %d(%s)(会话已复位)' % (color, color_abbr(color)))
        elif command != 'run' or len(tokens) != 1:
            self.get_logger().warn(
                '无法识别命令 "%s"。可用: run / stop / track <色>' % message.data)

    def _reset_turntable(self):
        self.turntable_cache = None
        self._turntable_last_seen = None
        self._turntable_last_check = None
        self._turntable_seen = False
        self._frame_count = 0

    def _image_callback(self, message: Image) -> None:
        if not self.running:
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

        timestamp = stamp_seconds(message) or time.monotonic()
        frame_result = self.pipeline.process(frame, timestamp)
        self._maybe_update_turntable(frame)
        # 只输出当帧新鲜匹配(age==0)的轨迹: 漏检保留的旧轨迹(age>0)不进
        # 结果与抓取候选——修复"漏检后旧坐标/旧 target 残留"(交接文档§5 风险);
        # 轨迹老化机制保留在管线内部, 短暂遮挡后重匹配即恢复输出
        visible = [item for item in frame_result.decided
                   if item[1] == self.track_color and item[0].age == 0]
        payload = self._fill(frame_result, visible)
        annotated = self.pipeline.annotate(frame, visible, payload,
                                           self._hud_lines(payload))
        output = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        output.header = message.header
        self.publisher.publish(output)
        result = String()
        result.data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        self.result_publisher.publish(result)
        if self.display:
            self._show(annotated)

    # ---------- 转盘区域判定 ----------
    def _maybe_update_turntable(self, frame: np.ndarray) -> None:
        """Recheck even at rest; a previously observed circle cannot remain valid forever."""
        self._frame_count += 1
        if not self.turntable_check:
            return
        now = time.monotonic()
        if (self._turntable_last_seen is not None and
                now - self._turntable_last_seen > self.turntable_cache_ttl):
            self.turntable_cache = None
        if (self.pipeline.stationary and self._turntable_last_check is not None and
                now - self._turntable_last_check < self.turntable_recheck_interval):
            return
        if not self.pipeline.stationary and self._frame_count % self.turntable_interval:
            return
        self._turntable_last_check = now
        cached = self._detect_turntable_circle(frame)
        if cached is not None:
            if self.turntable_cache is None:
                self.get_logger().info(
                    '转盘圆已锁定: (%d,%d) r=%.0fpx。' % cached)
            self.turntable_cache = cached
            self._turntable_last_seen = time.monotonic()
            self._turntable_seen = True

    def _detect_turntable_circle(self, frame: np.ndarray):
        """0.5 缩放图 Hough 大圆, 取最靠近画面中心的合格候选(原图坐标)。"""
        scale = 0.5
        small = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        min_radius = max(10, int(self.turntable_min_radius * scale))
        max_radius = max(min_radius + 10, int(self.turntable_max_radius * scale))
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(60.0, min_radius * 0.8),
            param1=60.0, param2=40.0,
            minRadius=min_radius, maxRadius=max_radius,
        )
        if circles is None:
            return None
        height, width = small.shape[:2]
        center_prior = np.asarray((width / 2.0, height / 2.0))
        best = None
        best_distance = math.inf
        for local_x, local_y, radius in circles[0]:
            candidate = np.asarray((float(local_x), float(local_y)))
            distance = float(np.linalg.norm(candidate - center_prior))
            if distance < best_distance:
                best_distance = distance
                best = (candidate, float(radius))
        if best is None:
            return None
        (center_x, center_y), radius = best
        return (int(round(center_x / scale)), int(round(center_y / scale)),
                radius / scale)

    def _judge_in_turntable(self, centroid: Tuple[int, int]):
        """目标原始质心是否在当前有效转盘圆内。

        余量 1.0(不外扩): 实测 1.05 会把恰在物理边缘外的物料误判圆内。
        """
        if not self.turntable_check or self.turntable_cache is None:
            return None
        center_u, center_v, radius = self.turntable_cache
        distance = math.hypot(centroid[0] - center_u, centroid[1] - center_v)
        return distance <= radius

    # ---------- 输出填充 ----------
    def _load_measurement_matrix(self):
        """Match camera_undistort's alpha=0 output intrinsics and resolution."""
        storage = cv2.FileStorage(str(Path(self.get_parameter('calibration_path').value)
                                     .expanduser()), cv2.FILE_STORAGE_READ)
        try:
            if not storage.isOpened():
                raise ValueError('无法读取测高标定文件')
            matrix = storage.getNode('camera_matrix').mat()
            distortion = storage.getNode('distortion_coefficients').mat()
            size = (int(storage.getNode('image_width').real()),
                    int(storage.getNode('image_height').real()))
            if (matrix is None or matrix.shape != (3, 3) or distortion is None
                    or distortion.size != 5 or size != self.pipeline.image_size):
                raise ValueError('测高标定内容或分辨率无效')
            updated, _ = cv2.getOptimalNewCameraMatrix(matrix, distortion, size, 0, size)
            if not np.isfinite(updated).all() or min(updated[0, 0], updated[1, 1]) <= 0:
                raise ValueError('测高焦距无效')
            return updated
        except (ValueError, cv2.error) as error:
            self.get_logger().warn(str(error))
            return None
        finally:
            storage.release()

    def _optical_distance(self):
        if (not self.turntable_check or self.turntable_cache is None
                or self.measurement_matrix is None):
            return None
        radius = self.turntable_cache[2]
        # A fronto-parallel 300 mm disk: Z = f_px * 150 mm / radius_px.
        fx, fy = self.measurement_matrix[0, 0], self.measurement_matrix[1, 1]
        if not math.isfinite(radius) or radius <= 0 or abs(fx - fy) / max(fx, fy) > 0.05:
            return None
        distance = float((fx + fy) * 0.5 * 150.0 / radius)
        return distance if distance > self.pipeline.material_height_mm else None

    def _fill(self, frame_result, visible) -> dict:
        """Return measured optical distance and corrected coordinates; invalid depth blocks grasp."""
        distance = self._optical_distance()
        if distance is not None:
            self.pipeline.camera_to_tray_mm = distance
        reported_distance = round(distance, 1) if distance is not None else None
        payload = {'mode': 'material_track', 'gate': 'run',
                   'track_color': self.track_color,
                   'stationary': frame_result.stationary,
                   'camera_to_turntable_distance_mm': reported_distance,
                   'distance_valid': distance is not None}
        if frame_result.motion_speed is not None:
            payload['motion_speed_px_s'] = round(frame_result.motion_speed, 1)
        materials = []
        for track, color, confidence in visible:
            if confidence < self.pipeline.min_confidence:
                continue
            bottom = self.pipeline.bottom_center(track.center) if distance is not None else (None, None)
            materials.append({
                'color': color,
                'center_u': bottom[0],
                'center_v': bottom[1],
                'centroid_u': track.center[0],
                'centroid_v': track.center[1],
                'camera_to_turntable_distance_mm': reported_distance,
                'confidence': round(confidence, 3),
            })
        payload['materials'] = materials

        # 转盘圆缓存信息与 in_turntable 三值判定(对置信度最高的候选)
        candidates = [
            (track, confidence) for track, _color, confidence in visible
            if confidence >= self.pipeline.min_confidence
        ]
        if self.pipeline.motion_gate_px > 0 and self.last_target is not None:
            gated = [
                (track, confidence) for track, confidence in candidates
                if np.hypot(track.center[0] - self.last_target[0],
                            track.center[1] - self.last_target[1]) <= self.pipeline.motion_gate_px
            ]
            if gated:
                candidates = gated
        in_turntable = None
        if candidates:
            best_track = max(candidates, key=lambda item: item[1])[0]
            # 转盘归属判定用原始质心(图像成员关系: 色块是否落在转盘圆内),
            # 不用视差修正后的底心——底心向主点收拢, 会把角落物料拉进圆内
            in_turntable = self._judge_in_turntable(best_track.center)
        if self.turntable_cache is not None:
            center_u, center_v, radius = self.turntable_cache
            payload['turntable'] = {'detected': True, 'center_u': center_u,
                                    'center_v': center_v, 'radius_px': round(radius, 1)}
        else:
            payload['turntable'] = None
        payload['in_turntable'] = in_turntable

        # 抓取需要有效测距; 未知转盘不能用默认高度生成底心目标。
        cache_lost = self.turntable_check and self._turntable_seen and self.turntable_cache is None
        grasp_ready = (frame_result.stationary and bool(candidates) and
                       in_turntable is not False and not cache_lost and distance is not None)
        payload['turntable_state'] = ('disabled' if not self.turntable_check else
                                     'lost' if cache_lost else
                                     'fresh' if self.turntable_cache is not None else 'unknown')
        payload['grasp_ready'] = grasp_ready

        target = None
        if grasp_ready and candidates:
            best_track, best_confidence = max(candidates, key=lambda item: item[1])
            position = best_track.ema if best_track.ema is not None else best_track.center
            self.last_target = (float(position[0]), float(position[1]))
            bottom = self.pipeline.bottom_center((int(round(position[0])), int(round(position[1]))))
            target = {
                'center_u': bottom[0],
                'center_v': bottom[1],
                'camera_to_turntable_distance_mm': reported_distance,
                'confidence': round(best_confidence, 3),
            }
        payload['target'] = target
        return payload

    def _hud_lines(self, payload: dict) -> Tuple[str, ...]:
        lines = []
        if self.track_color is None:
            return lines
        lines.append('target: %d(%s)' % (self.track_color, color_abbr(self.track_color)))
        distance = payload.get('camera_to_turntable_distance_mm')
        lines.append('Z: %s mm' % (distance if distance is not None else 'UNKNOWN'))
        lines.append('grasp: %s' % ('READY TO GRASP' if payload.get('grasp_ready')
                                    else 'WAIT (turntable/stop)'))
        turntable = payload.get('turntable')
        if turntable:
            lines.append('TURN: (%d,%d) r=%.0f  IN: %s'
                         % (turntable['center_u'], turntable['center_v'],
                            turntable['radius_px'], payload.get('in_turntable')))
        else:
            lines.append('TURN: %s  IN: null' % payload.get('turntable_state', 'unknown'))
        return lines

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

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[MaterialTrackDetectNode] = None
    try:
        node = MaterialTrackDetectNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('material_track_detect 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
