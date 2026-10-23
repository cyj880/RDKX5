"""Material V2: turntable material detection, color vote, grasp target.

物料识别第二版 · 单节点(与旧包 material_detect 同构, 判色特征相对化):
订阅旧包去畸变图, 白背景 Lab 分割 → 形状筛选 → 跨帧轨迹+投票判色
(kNN 吃相对白背景的 Lab 特征) → locate/track/idle 模式机 → 视差底心
输出与静止检测, 发布 /vision/material2/results。

相对 v1 的三处改动:
    1. 判色特征相对化: kNN 特征从物料块绝对 Lab 改为相对每帧白背景基准
       (共享 material_features 模块), 模型跨光照迁移, 现场不易失配。
    2. 参数热调: 分割/形状/判色门槛等参数注册 on_set 回调,
       `ros2 param set` 即时生效, 不重启节点(现场调参即"我们的滑条")。
    3. idle 跳过处理: 命令驱动模式下 idle 只刷窗口不分割, X5 上省 CPU。

命令(主控或 material_console 发布到 /vision/material2/command):
    locate <色1> <色2> <色3> / track <色> / all / idle
    all = 六色全识别(调试/无任务码时用); idle = 跳过处理省 CPU。
颜色编号按规则: 红1 黄2 蓝3 绿4 黑5 浅蓝6。

停稳门控(逐个抓取的工作模式=track): 静止判据=物料轨迹帧间位移连续
stillness_frames 帧 ≤ stillness_motion_px(两参数可热调)。圆盘未停稳时
track 模式输出 grasp_ready=false 且 target=null——主控拿不到夹取坐标,
从源头保证"圆盘停止后才开始夹取"; 停稳后 grasp_ready=true, target 自动
恢复。重发任何命令都会复位静止态, 新会话必须重新确认停稳。
转盘圆心定位/锁定不在本节点范围(本轮不做)。
"""

from collections import deque
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .material_features import (
    color_abbr,
    color_feature_vector_relative,
    color_name,
    estimate_background_ref,
    FEATURE_TAG,
)


WINDOW_NAME = 'Material2 Detect'


@dataclass
class BlobTrack:
    """一个物料块的跨帧轨迹: 位置 + 颜色投票。"""

    track_id: int
    center: Tuple[int, int]
    area: int
    bbox: Tuple[int, int, int, int]
    votes: List[Tuple[int, float]] = field(default_factory=list)
    ema: Optional[Tuple[float, float]] = None
    age: int = 0  # 连续未被匹配的帧数


class Material2DetectNode(Node):
    """识别转盘上的物料: 定位、定色与目标追踪。"""

    def __init__(self) -> None:
        super().__init__('material2_detect')
        self.declare_parameter('input_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('output_topic', '/vision/material2/annotated')
        self.declare_parameter('result_topic', '/vision/material2/results')
        self.declare_parameter('command_topic', '/vision/material2/command')
        self.declare_parameter(
            'model_path', '~/logistics_vision_data/color_model/colors_knn_v2.yml'
        )
        self.declare_parameter('display', False)
        self.declare_parameter('display_debug', False)
        self.declare_parameter('start_mode', 'idle')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        # 层3 视差修正: 质心→底心(锥轴落地点=夹取目标), Z/h/r_ratio 参数化
        self.declare_parameter(
            'calibration_path',
            '~/logistics_vision_data/calibration/camera_640x480.yaml',
        )
        self.declare_parameter('camera_to_tray_mm', 200.0)
        self.declare_parameter('material_height_mm', 70.0)
        self.declare_parameter('material_radius_ratio', 0.6)

        # 分割与形状筛选(均为热调参数, 见 self._tunable)
        self.declare_parameter('saturation_threshold', 40)
        self.declare_parameter('color_delta_ab', 25.0)
        self.declare_parameter('dark_l_ratio', 0.55)
        self.declare_parameter('dark_delta_ab', 15.0)
        self.declare_parameter('dark_open_kernel', 7)
        self.declare_parameter('merge_distance_px', 40)
        self.declare_parameter('min_blob_area', 500)
        self.declare_parameter('max_blob_area', 40000)
        self.declare_parameter('min_solidity', 0.70)
        self.declare_parameter('min_circularity', 0.30)

        # 判定与稳定性
        self.declare_parameter('min_confidence', 0.6)
        self.declare_parameter('vote_frames', 5)
        self.declare_parameter('match_distance_px', 40)
        self.declare_parameter('track_max_age', 8)
        self.declare_parameter('smoothing_alpha', 0.0)
        self.declare_parameter('motion_gate_px', 0)
        self.declare_parameter('stillness_motion_px', 1.5)
        self.declare_parameter('stillness_frames', 6)

        self.image_size = (
            int(self.get_parameter('image_width').value),
            int(self.get_parameter('image_height').value),
        )
        self.display = bool(self.get_parameter('display').value)
        # 层1 掩膜调试窗口独立开关(默认关, 只开主结果窗口)
        self.display_debug = bool(self.get_parameter('display_debug').value)
        self._read_tunable_parameters()

        self.tracks: List[BlobTrack] = []
        self.next_track_id = 0
        self.last_target: Optional[Tuple[float, float]] = None
        self.stationary = False
        self._last_motion_px: Optional[float] = None
        self.still_count = 0
        # 模式状态
        self.mode = 'idle'
        start_mode = str(self.get_parameter('start_mode').value).lower()
        if start_mode in ('all', 'idle'):
            self.mode = start_mode
        self.locate_colors: List[int] = []
        self.track_color: Optional[int] = None
        # 白背景 Lab 基准 (μL, μa, μb), 每帧刷新; 画面被大面积遮挡时沿用旧值
        self._bg_ref: Optional[Tuple[float, float, float]] = None

        self.bridge = CvBridge()
        self.principal_point = self._load_principal_point(
            Path(str(self.get_parameter('calibration_path').value)).expanduser()
        )
        self._vis_mask_color: Optional[np.ndarray] = None
        self._vis_mask_dark: Optional[np.ndarray] = None
        self.window_created = False
        self.size_error_reported = False

        # 热调参数: ros2 param set 即时生效, 不重启节点
        self._tunable = {
            'saturation_threshold': ('saturation_threshold', int, 0, 255),
            'color_delta_ab': ('color_delta_ab', float, 0.0, 510.0),
            'dark_l_ratio': ('dark_l_ratio', float, 0.0, 1.0),
            'dark_delta_ab': ('dark_delta_ab', float, 0.0, 510.0),
            'dark_open_kernel': ('dark_open_kernel', int, 1, 31),
            'merge_distance_px': ('merge_distance_px', float, 0.0, 300.0),
            'min_blob_area': ('min_blob_area', int, 0, 100000),
            'max_blob_area': ('max_blob_area', int, 0, 500000),
            'min_solidity': ('min_solidity', float, 0.0, 1.0),
            'min_circularity': ('min_circularity', float, 0.0, 1.0),
            'min_confidence': ('min_confidence', float, 0.0, 1.0),
            'match_distance_px': ('match_distance_px', float, 1.0, 300.0),
            # 视差三参数热调: 现场看红十字是否落在物料底面圆心, 直接 param set
            'camera_to_tray_mm': ('camera_to_tray_mm', float, 50.0, 500.0),
            'material_height_mm': ('material_height_mm', float, 10.0, 200.0),
            'material_radius_ratio': ('material_radius_ratio', float, 0.1, 1.0),
            # 静止判据热调: 位移阈值(px)与确认帧数, 现场按转盘惯性调
            'stillness_motion_px': ('stillness_motion_px', float, 0.1, 50.0),
            'stillness_frames': ('stillness_frames', int, 1, 60),
        }
        self.add_on_set_parameters_callback(self._on_set_parameters)

        model_path = Path(str(self.get_parameter('model_path').value)).expanduser()
        self.knn = None
        self.knn_k = 3
        if model_path.is_file():
            self.knn, self.knn_k = self._load_knn(model_path)
        else:
            self.get_logger().warn(
                '未找到颜色模型: %s。彩色物料块将标记为 color=0(仅分割调试);'
                '用 material2_sample_capture 采集并 material2_color_train 训练后重启。'
                % model_path
            )

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        result_topic = str(self.get_parameter('result_topic').value)
        command_topic = str(self.get_parameter('command_topic').value)
        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.result_publisher = self.create_publisher(String, result_topic, 10)
        self.command_subscription = self.create_subscription(
            String, command_topic, self._command_callback, 10
        )
        self.subscription = self.create_subscription(
            Image, input_topic, self._image_callback, 10
        )
        self.get_logger().info(
            '物料识别节点已启动: 图像 %s, 命令 %s, 结果 %s。模式: locate/track/idle。'
            % (input_topic, command_topic, result_topic)
        )

    # ---------- 热调参数 ----------
    def _read_tunable_parameters(self) -> None:
        self.saturation_threshold = int(self.get_parameter('saturation_threshold').value)
        self.color_delta_ab = float(self.get_parameter('color_delta_ab').value)
        self.dark_l_ratio = float(self.get_parameter('dark_l_ratio').value)
        self.dark_delta_ab = float(self.get_parameter('dark_delta_ab').value)
        self.dark_open_kernel = max(1, int(self.get_parameter('dark_open_kernel').value))
        self.merge_distance_px = float(self.get_parameter('merge_distance_px').value)
        self.min_blob_area = int(self.get_parameter('min_blob_area').value)
        self.max_blob_area = int(self.get_parameter('max_blob_area').value)
        self.min_solidity = float(self.get_parameter('min_solidity').value)
        self.min_circularity = float(self.get_parameter('min_circularity').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.vote_frames = max(1, int(self.get_parameter('vote_frames').value))
        self.match_distance = float(self.get_parameter('match_distance_px').value)
        self.track_max_age = int(self.get_parameter('track_max_age').value)
        self.smoothing_alpha = float(self.get_parameter('smoothing_alpha').value)
        self.motion_gate_px = int(self.get_parameter('motion_gate_px').value)
        self.stillness_motion_px = float(self.get_parameter('stillness_motion_px').value)
        self.stillness_frames = int(self.get_parameter('stillness_frames').value)
        # 层3 视差修正参数
        self.camera_to_tray_mm = float(self.get_parameter('camera_to_tray_mm').value)
        self.material_height_mm = float(self.get_parameter('material_height_mm').value)
        self.material_radius_ratio = float(
            self.get_parameter('material_radius_ratio').value)

    def _on_set_parameters(self, parameters) -> SetParametersResult:
        """热调回调: 校验范围后直接更新成员变量, 当帧即生效。"""
        for parameter in parameters:
            entry = self._tunable.get(parameter.name)
            if entry is None:
                continue
            attr_name, expected_type, low, high = entry
            value = parameter.value
            if expected_type is int:
                if not isinstance(value, int) or isinstance(value, bool):
                    return SetParametersResult(
                        successful=False, reason='%s 必须是整数' % parameter.name)
            else:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return SetParametersResult(
                        successful=False, reason='%s 必须是数值' % parameter.name)
            value = expected_type(value)
            if not low <= value <= high:
                return SetParametersResult(
                    successful=False,
                    reason='%s 超出范围 [%s, %s]' % (parameter.name, low, high))
            if parameter.name == 'dark_open_kernel' and value % 2 == 0:
                value += 1
            setattr(self, attr_name, value)
            self.get_logger().info('参数 %s 热更新为 %s' % (parameter.name, value))
        return SetParametersResult(successful=True)

    # ---------- 层3: 视差修正(质心→底心) ----------
    def _load_knn(self, path: Path):
        """加载 kNN 颜色模型; 特征口径不符时拒绝加载(退化为 color=0 调试模式)。"""
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise RuntimeError('无法读取颜色模型: %s' % path)
        tag = storage.getNode('features').string()
        if tag != FEATURE_TAG:
            storage.release()
            self.get_logger().error(
                '颜色模型特征口径不符: %s (期望 %s, 实际 %s), 拒绝加载。'
                '请用 material2_sample_capture + material2_color_train 重新训练。'
                % (path, FEATURE_TAG, tag or '(缺失)'))
            return None, self.knn_k
        samples = storage.getNode('samples').mat()
        labels = storage.getNode('labels').mat()
        k = int(storage.getNode('k').real())
        storage.release()
        if samples is None or labels is None or len(samples) == 0:
            raise RuntimeError('颜色模型内容无效: %s' % path)
        knn = cv2.ml.KNearest_create()
        knn.setDefaultK(k)
        knn.train(samples, cv2.ml.ROW_SAMPLE, labels.astype(np.int32))
        return knn, k

    def _load_principal_point(self, path: Path) -> Tuple[float, float]:
        """从标定文件读主点 (cx, cy); 失败时退回图像中心并告警。"""
        fallback = (self.image_size[0] / 2.0, self.image_size[1] / 2.0)
        try:
            storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            if not storage.isOpened():
                raise RuntimeError('无法打开标定文件: %s' % path)
            matrix = storage.getNode('camera_matrix').mat()
            storage.release()
            if matrix is None or matrix.shape != (3, 3):
                raise RuntimeError('camera_matrix 无效: %s' % path)
            return float(matrix[0][2]), float(matrix[1][2])
        except (RuntimeError, cv2.error) as error:
            self.get_logger().warn(
                '读取主点失败(%s), 视差修正退用图像中心(%.0f, %.0f)。'
                % (error, fallback[0], fallback[1])
            )
            return fallback

    def _parallax_k(self) -> float:
        """质心外偏比例 k 的解析值: k = W·(h/(Z−h))/(1+W), W = (r_ratio·Z/(Z−h))²。

        锥台俯视投影中顶面被放大 Z/(Z−h) 倍, blob 面积权重比 W 即
        (顶面投影半径/底面投影半径)², 质心相对底面轴心沿主点方向外偏 k。
        Z=275,h=70,r_ratio=0.6 → k=0.134; Z=295 → k=0.129。
        """
        z = max(1.0, self.camera_to_tray_mm)
        h = min(self.material_height_mm, z - 1.0)
        ratio = z / (z - h)
        w = (self.material_radius_ratio * ratio) ** 2
        return w * (h / (z - h)) / (1.0 + w)

    def _bottom_center(self, centroid: Tuple[int, int]) -> Tuple[int, int]:
        """质心→底心: 底心 = 主点 + (质心−主点)·(1−k)。

        底心即锥轴在盘面的落点, 是机械臂的夹取目标。
        """
        k = self._parallax_k()
        px, py = self.principal_point
        return (
            int(round(px + (centroid[0] - px) * (1.0 - k))),
            int(round(py + (centroid[1] - py) * (1.0 - k))),
        )

    # ---------- 命令 ----------
    def _command_callback(self, message: String) -> None:
        tokens = message.data.strip().split()
        if not tokens:
            return
        command = tokens[0].lower()
        if command == 'locate' and len(tokens) == 4:
            try:
                colors = [int(value) for value in tokens[1:]]
            except ValueError:
                self.get_logger().warn('locate 参数必须是数字: %s' % message.data)
                return
            if not all(1 <= color <= 6 for color in colors):
                self.get_logger().warn('颜色编号必须在 1~6: %s' % message.data)
                return
            self.mode = 'locate'
            self.locate_colors = colors
            self.track_color = None
            self._reset_session()
            self.get_logger().info(
                '进入定位模式: 本批颜色 %s' % ' '.join(color_name(c) for c in colors)
            )
        elif command == 'track' and len(tokens) == 2:
            try:
                color = int(tokens[1])
            except ValueError:
                self.get_logger().warn('track 参数必须是数字: %s' % message.data)
                return
            if not 1 <= color <= 6:
                self.get_logger().warn('颜色编号必须在 1~6: %s' % message.data)
                return
            self.mode = 'track'
            self.track_color = color
            self._reset_session()
            self.get_logger().info(
                '进入追踪模式: 目标 %d(%s)' % (color, color_name(color)))
        elif command == 'all':
            self.mode = 'all'
            self.track_color = None
            self._reset_session()
            self.get_logger().info('进入全识别模式(六色)。')
        elif command == 'idle':
            self.mode = 'idle'
            self.track_color = None
            self._reset_session()
            self.get_logger().info('进入空闲模式(跳过处理省 CPU)。')
        else:
            self.get_logger().warn(
                '无法识别命令 "%s"。可用: locate <色1><色2><色3> / track <色> / all / idle'
                % message.data
            )

    def _reset_session(self) -> None:
        self.tracks = []
        self.last_target = None
        # 静止态一并复位: 新会话必须重新确认停稳, 杜绝上一轮的
        # stationary=true 残留(转盘可能已在转, 夹取门控不能沿用旧结论)
        self.stationary = False
        self.still_count = 0

    # ---------- 主处理 ----------
    def _image_callback(self, message: Image) -> None:
        if self.mode == 'idle' and not self.display:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error('无法转换图像: %s' % error)
            return
        height, width = frame.shape[:2]
        if (width, height) != self.image_size:
            if not self.size_error_reported:
                self.get_logger().error(
                    '输入图像 %dx%d 与预期 %dx%d 不符，跳过该帧。'
                    % (width, height, *self.image_size)
                )
                self.size_error_reported = True
            return
        self.size_error_reported = False

        if self.mode == 'idle':
            # idle 不分割不发布, 只刷窗口保退出键响应(X5 省 CPU)
            if self.display:
                self._show_idle(frame)
            return

        payload, annotated = self._process_frame(frame)
        output = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        output.header = message.header
        self.publisher.publish(output)
        result = String()
        result.data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        self.result_publisher.publish(result)
        if self.display:
            self._show(annotated)
        if self.display_debug:
            self._show_layer1(frame)

    def _show_idle(self, frame: np.ndarray) -> None:
        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        idle = frame.copy()
        cv2.putText(idle, 'IDLE (send locate/track to /vision/material2/command)',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, idle)
        self._poll_quit()

    def _poll_quit(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令。')
            rclpy.shutdown()

    def _update_stillness(self) -> None:
        """静止判据: 物料轨迹中心的帧间位移(替代旧整幅帧差分)。

        转盘转动时物料绕盘心公转, track 匹配位移即运动量; 夹爪入画或抓走
        一个物料不影响其余物料轨迹, 不会误判。位移连续低于阈值达
        stillness_frames 帧判静止(30fps 下 6 帧=0.2s, 满足 <=0.3s 要求);
        一旦超阈值立即翻回运动, 转盘重新启动尽快响应。
        """
        if self._last_motion_px is None:
            return  # 本帧无轨迹位移数据(无匹配), 保持原状态
        if self._last_motion_px <= self.stillness_motion_px:
            self.still_count += 1
            if self.still_count >= self.stillness_frames:
                self.stationary = True
        else:
            self.still_count = 0
            self.stationary = False

    def _process_frame(self, frame: np.ndarray) -> Tuple[dict, np.ndarray]:
        blobs = self._detect_blobs(frame)
        self._update_tracks(blobs)
        self._update_stillness()
        decided = [
            (track,) + self._decided_color(track)
            for track in self.tracks
            if len(track.votes) >= max(1, self.vote_frames // 2)
        ]

        payload = {'mode': self.mode, 'stationary': self.stationary}
        if self._last_motion_px is not None:
            payload['motion_px'] = round(self._last_motion_px, 2)
        # 识别范围随模式收窄: locate 只认任务三色, track 只认当前目标色——
        # 窗口与 JSON 均只显示范围内物料; 范围外轨迹在内部照常维护(投票/
        # 老化不停), 换色时无需重建
        if self.mode == 'locate':
            wanted = set(self.locate_colors)
            visible = [item for item in decided if item[1] in wanted]
            self._fill_locate(payload, visible)
        elif self.mode == 'track':
            visible = [item for item in decided if item[1] == self.track_color]
            self._fill_track(payload, visible)
        elif self.mode == 'all':
            visible = decided
            self._fill_all(payload, visible)
        else:
            visible = decided

        annotated = self._annotate(frame, visible, payload)
        return payload, annotated

    # ---------- 层1: 候选生成 ----------
    def _detect_blobs(self, frame: np.ndarray) -> List[dict]:
        """相对白背景分割: 物料 = 与白背景 Lab 基准显著偏离的区域。

        彩料掩膜 = 色度差 |Δa|+|Δb| > color_delta_ab —— 覆盖六色中饱和度最低的
        绿/浅蓝(实测 Δab P5=29/31), 白背景自身 Δab max=18;
        黑料掩膜 = 亮度比 L < dark_l_ratio·μL 且无色度 Δab < dark_delta_ab ——
        暗+灰才是黑: 红/蓝物料 L 低(0.26~0.52·μL)但色度高(Δab≥44)不会误判黑,
        黑料 L 比值 0.08、Δab P95=6, 两个条件余量都在 2 倍以上。

        两张掩膜独立成块, 贴画幅边界的连通域剔除(地图边缘边框/大片背景干扰),
        形状筛选后送判色——彩色块走 kNN 相对特征, 暗块按定义判黑。
        """
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lab_u8 = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        lab = lab_u8.astype(np.float32)
        sat = hsv[:, :, 1]

        ref = estimate_background_ref(lab, sat, self.saturation_threshold)
        if ref is not None:
            self._bg_ref = ref
        if self._bg_ref is None:
            # 画面被大面积遮挡且无历史基准: 本帧无法分割
            empty = np.zeros(frame.shape[:2], np.uint8)
            self._vis_mask_color = empty
            self._vis_mask_dark = empty
            return []
        lab_f = lab
        l_ch, a_ch, b_ch = lab_f[:, :, 0], lab_f[:, :, 1], lab_f[:, :, 2]
        mu_l, mu_a, mu_b = self._bg_ref
        delta_ab = np.abs(a_ch - mu_a) + np.abs(b_ch - mu_b)

        mask_color = np.where(
            delta_ab > self.color_delta_ab, 255, 0
        ).astype(np.uint8)
        mask_dark = np.where(
            (l_ch < self.dark_l_ratio * mu_l) & (delta_ab < self.dark_delta_ab),
            255, 0,
        ).astype(np.uint8)

        kernel_open_color = np.ones((3, 3), np.uint8)
        kernel_open_dark = np.ones(
            (self.dark_open_kernel, self.dark_open_kernel), np.uint8
        )
        kernel_close = np.ones((5, 5), np.uint8)
        # 暗掩膜用大开核: 物料压在印刷细线上时, 细线(~4px)被开运算抹断,
        # 物料主体(数十px)无损保留, 避免连成"物料+长线"被圆形度误杀
        mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel_open_color)
        mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel_close)
        mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel_open_dark)
        mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel_close)
        self._vis_mask_color = mask_color
        self._vis_mask_dark = mask_dark

        blobs: List[dict] = []
        blobs += self._extract_blobs(mask_color, 'color', lab_f, self._bg_ref)
        blobs += self._extract_blobs(mask_dark, 'dark', lab_f, self._bg_ref)
        blobs = self._merge_overlapping(blobs)
        return blobs

    def _merge_overlapping(self, blobs: List[dict]) -> List[dict]:
        """合并同一物料身上的暗块与反光彩块。

        黑色物料的受光面 S 会超过彩色阈值, 在彩色掩膜里成为第二个块,
        与暗块近距离并存导致跟踪目标抖动。合并规则: 距离小于
        merge_distance_px 的块归并为一个, 暗块优先提供中心与颜色(判黑)。
        """
        merged: List[dict] = []
        for blob in sorted(blobs, key=lambda b: -b['area']):
            for target in merged:
                if np.hypot(blob['center'][0] - target['center'][0],
                            blob['center'][1] - target['center'][1]) < self.merge_distance_px:
                    if blob['dark']:
                        target['center'] = blob['center']
                        target['color'] = 5
                        target['confidence'] = max(target['confidence'], blob['confidence'])
                    target['area'] = max(target['area'], blob['area'])
                    break
            else:
                merged.append(dict(blob))
        return merged

    def _extract_blobs(self, mask: np.ndarray, kind: str, lab: np.ndarray,
                       bg_ref: Tuple[float, float, float]) -> List[dict]:
        height, width = mask.shape
        # RETR_LIST: 必须取全部轮廓——边框类闭合环绕住画面时, 环内物料轮廓
        # 属于层级子节点, RETR_EXTERNAL 会把环内物料整个吞掉
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        blobs: List[dict] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_blob_area or area > self.max_blob_area:
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            # 贴边剔除: 地图边缘边框、延伸出画面的背景大片区域
            if x <= 0 or y <= 0 or x + box_w >= width or y + box_h >= height:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area <= 0 or area / hull_area < self.min_solidity:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
            if circularity < self.min_circularity:
                continue

            if kind == 'color':
                blob_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(blob_mask, [contour], -1, 255, -1)
                try:
                    features = color_feature_vector_relative(lab, blob_mask, bg_ref)
                except ValueError:
                    continue
                color, confidence = self._classify(features)
            else:
                color, confidence = 5, 1.0  # 黑色物料按"白背景上的暗块"定义判定

            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            center = (
                int(moments['m10'] / moments['m00']),
                int(moments['m01'] / moments['m00']),
            )
            blobs.append({
                'center': center,
                'area': int(area),
                'bbox': (x, y, box_w, box_h),
                'color': color,
                'confidence': confidence,
                'dark': (kind == 'dark'),
            })
        return blobs

    def _classify(self, features: np.ndarray) -> Tuple[int, float]:
        if self.knn is None:
            return 0, 0.0
        _, results, neighbors, _ = self.knn.findNearest(
            features.reshape(1, -1), self.knn_k
        )
        color = int(results[0][0])
        confidence = float(np.mean(neighbors.flatten() == color))
        return color, confidence

    # ---------- 层2: 跨帧轨迹与投票 ----------
    def _update_tracks(self, blobs: List[dict]) -> None:
        matched_distances: List[float] = []
        unmatched = list(blobs)
        for track in self.tracks:
            best = None
            best_distance = self.match_distance
            for blob in unmatched:
                distance = float(np.hypot(
                    blob['center'][0] - track.center[0],
                    blob['center'][1] - track.center[1],
                ))
                if distance < best_distance:
                    best = blob
                    best_distance = distance
            if best is not None:
                unmatched.remove(best)
                matched_distances.append(best_distance)
                track.center = best['center']
                track.area = best['area']
                track.bbox = best['bbox']
                track.age = 0
                track.votes.append((best['color'], best['confidence']))
                if len(track.votes) > self.vote_frames:
                    track.votes.pop(0)
                if self.smoothing_alpha > 0:
                    if track.ema is None:
                        track.ema = (float(best['center'][0]), float(best['center'][1]))
                    else:
                        track.ema = (
                            self.smoothing_alpha * best['center'][0] + (1 - self.smoothing_alpha) * track.ema[0],
                            self.smoothing_alpha * best['center'][1] + (1 - self.smoothing_alpha) * track.ema[1],
                        )
                else:
                    track.ema = None
            else:
                track.age += 1
        self.tracks = [track for track in self.tracks if track.age <= self.track_max_age]
        for blob in unmatched:
            self.tracks.append(BlobTrack(
                track_id=self.next_track_id,
                center=blob['center'],
                area=blob['area'],
                bbox=blob['bbox'],
                votes=[(blob['color'], blob['confidence'])],
            ))
            self.next_track_id += 1
        # 本帧运动量 = 已匹配轨迹的帧间位移均值; 无匹配则无数据(静止状态保持)
        self._last_motion_px = (
            float(np.mean(matched_distances)) if matched_distances else None
        )

    def _decided_color(self, track: BlobTrack) -> Tuple[int, float]:
        """投票定色: 多数色 + 多帧一致性作为置信度。"""
        if not track.votes:
            return 0, 0.0
        counter = {}
        for color, _ in track.votes:
            counter[color] = counter.get(color, 0) + 1
        color = max(counter, key=counter.get)
        confidence = counter[color] / float(len(track.votes))
        return color, confidence

    # ---------- 层3: 模式输出 ----------
    def _fill_locate(self, payload: dict, decided: List[tuple]) -> None:
        wanted = set(self.locate_colors)
        found = {}
        for track, color, confidence in decided:
            if color in wanted and confidence >= self.min_confidence:
                found[color] = track
        payload['locate_colors'] = list(self.locate_colors)
        materials = []
        for color in sorted(found):
            track = found[color]
            bottom = self._bottom_center(track.center)
            materials.append({
                'color': color,
                'center_u': bottom[0],
                'center_v': bottom[1],
                'centroid_u': track.center[0],
                'centroid_v': track.center[1],
                'area': track.area,
                'confidence': round(self._decided_color(track)[1], 3),
            })
        payload['materials'] = materials

    def _fill_all(self, payload: dict, decided: List[tuple]) -> None:
        """全识别模式: 六色全部输出(窗口画全部, JSON 只含确认项)。"""
        materials = []
        for track, color, confidence in decided:
            if color == 0 or confidence < self.min_confidence:
                continue  # 未确认色(kNN 未加载/低置信)只画窗口不进 JSON
            bottom = self._bottom_center(track.center)
            materials.append({
                'color': color,
                'center_u': bottom[0],
                'center_v': bottom[1],
                'centroid_u': track.center[0],
                'centroid_v': track.center[1],
                'area': track.area,
                'confidence': round(confidence, 3),
            })
        materials.sort(key=lambda m: m['color'])
        payload['materials'] = materials

    def _fill_track(self, payload: dict, decided: List[tuple]) -> None:
        payload['track_color'] = self.track_color
        # 停稳门控: 圆盘未停稳不输出夹取目标(target=null)——主控拿到也夹不得;
        # 停稳但目标色暂不可见/置信不够时同样 target=null, 等它转进视野
        grasp_ready = self.stationary
        payload['grasp_ready'] = grasp_ready
        materials = []
        for track, color, confidence in decided:
            if confidence < self.min_confidence:
                continue
            bottom = self._bottom_center(track.center)
            materials.append({
                'color': color,
                'center_u': bottom[0],
                'center_v': bottom[1],
                'centroid_u': track.center[0],
                'centroid_v': track.center[1],
                'confidence': round(confidence, 3),
            })
        payload['materials'] = materials
        candidates = [
            (track, confidence)
            for track, color, confidence in decided
            if color == self.track_color and confidence >= self.min_confidence
        ]
        if self.motion_gate_px > 0 and self.last_target is not None:
            gated = [
                (track, confidence) for track, confidence in candidates
                if np.hypot(track.center[0] - self.last_target[0],
                            track.center[1] - self.last_target[1]) <= self.motion_gate_px
            ]
            if gated:
                candidates = gated
        target = None
        if grasp_ready and candidates:
            best_track, best_confidence = max(candidates, key=lambda item: item[1])
            position = best_track.ema if best_track.ema is not None else best_track.center
            self.last_target = (float(position[0]), float(position[1]))
            bottom = self._bottom_center((int(round(position[0])), int(round(position[1]))))
            target = {
                'center_u': bottom[0],
                'center_v': bottom[1],
                'confidence': round(best_confidence, 3),
            }
        payload['target'] = target

    # ---------- 标注 ----------
    def _annotate(self, frame: np.ndarray, decided: List[tuple], payload: dict) -> np.ndarray:
        annotated = frame.copy()
        for track, color, confidence in decided:
            x, y, w, h = track.bbox
            if confidence >= self.min_confidence:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = '%d(%s) %.2f' % (color, color_abbr(color), confidence)
            else:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 160, 255), 1)
                label = '? %.2f' % confidence
            cv2.putText(
                annotated, label, (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
            )
            # 层3 视差: 黄点=blob质心, 品红线=锥轴, 红十字=底心(夹取目标)
            bottom = self._bottom_center(track.center)
            cv2.line(annotated, track.center, bottom, (255, 0, 255), 1)
            cv2.circle(annotated, track.center, 3, (0, 255, 255), 1)
            cv2.drawMarker(annotated, bottom, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)

        # track 模式且已停稳: 品红斜十字标出最终夹取目标(=JSON 的 target)
        target = payload.get('target')
        if target:
            cv2.drawMarker(annotated, (target['center_u'], target['center_v']),
                           (255, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 2)
            cv2.putText(annotated, 'TARGET',
                        (target['center_u'] + 12, target['center_v'] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2,
                        cv2.LINE_AA)

        lines = ['mode: %s' % self.mode]
        if self.mode == 'locate':
            lines.append('colors: %s' % self.locate_colors)
        elif self.mode == 'track':
            lines.append('target: %d(%s)' % (self.track_color, color_abbr(self.track_color)))
            lines.append('grasp: %s' % ('READY TO GRASP' if self.stationary
                                        else 'WAIT TURNTABLE STOP'))
        motion = 'n/a' if self._last_motion_px is None else '%.1fpx' % self._last_motion_px
        lines.append(
            'stationary: %s  motion: %s [%d/%d]'
            % (self.stationary, motion, self.still_count, self.stillness_frames)
        )
        lines.append('k=%.3f  RED=bottom grasp  YELLOW=centroid' % self._parallax_k())
        lines.append('WHITE=principal pt')
        for index, text in enumerate(lines):
            cv2.putText(
                annotated, text, (10, 25 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return annotated

    def _show_layer1(self, frame: np.ndarray) -> None:
        """层1 调试窗口: 绿=彩色掩膜命中, 蓝=黑料掩膜命中。"""
        if self._vis_mask_color is None or self._vis_mask_dark is None:
            return
        vis = frame.copy()
        overlay = vis.copy()
        overlay[self._vis_mask_color > 0] = (0, 255, 0)
        overlay[self._vis_mask_dark > 0] = (255, 0, 0)
        vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)
        cv2.putText(
            vis, 'Layer1: GREEN=color mask  BLUE=dark mask',
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow('Material2 Layer1 Masks', vis)
        self._poll_quit()

    def _show(self, annotated: np.ndarray) -> None:
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
    node: Optional[Material2DetectNode] = None
    try:
        node = Material2DetectNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('material2_detect 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
