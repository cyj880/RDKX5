"""ROS 2 node for locating turntable materials from a top-down view.

层1 分割（地图场地环境、自动曝光、补光灯）——相对白背景判据:
  白背景基准 = S<阈值的像素(白地图+白盘, 占画面大头)的 Lab 中位数, 每帧自适应
  Mask A 彩色物料 = 色度差 |Δa|+|Δb| > 阈值   —— 偏离背景色度的区域(色度对曝光远比 HSV 的 S 稳定)
  Mask B 黑色物料 = L<阈值×μL 且 Δab 小      —— 暗+灰才是黑: 红/蓝物料 L 低(实测 0.26~0.52·μL)
                    但色度高(Δab≥44), 不会误判黑; 黑料 Δab P95=6、L 比值 0.08, 余量 2 倍以上
  共同步骤: 贴画幅边界的连通域剔除（地图边缘边框/大片背景干扰）→ 形态学 → 形状筛选
  共同步骤: 贴画幅边界的连通域剔除（地图边缘边框/大片背景干扰）→ 形态学 → 形状筛选
层2 判色: 彩色块 Lab 特征 → kNN（模型缺失时 color=0 仅调试）；暗块按定义判黑
层3 视差修正: 底心 = 主点 + (质心−主点)·(1−k), k = W·(h/(Z−h))/(1+W),
  W = (r_ratio·Z/(Z−h))²。底心=锥轴在盘面的落点=夹取目标, 输出的
  center_u/center_v 与盘心锁定均用底心; Z/h/r_ratio 参数化,
  盘高(80~100mm)/决赛物料形状变化时改参数即可, 无需重标。
静止检测: 运动量 = 物料轨迹(track匹配)的帧间位移均值, 连续
  stillness_frames 帧低于 stillness_motion_px 判静止; 夹爪入画/抓取不影响
  其余物料轨迹, 替代旧整幅帧差分, 不再误判。

命令（material_console 或未来单片机桥接发布到 /vision/material_command）：
  locate <色1> <色2> <色3> / track <色> / idle
颜色编号按规则：红1 黄2 蓝3 绿4 黑5 浅蓝6
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
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .color_features import color_abbr, color_feature_vector, color_name


WINDOW_NAME = 'Material Detect'


@dataclass
class BlobTrack:
    """一个物料块的跨帧轨迹：位置 + 颜色投票。"""

    track_id: int
    center: Tuple[int, int]
    area: int
    bbox: Tuple[int, int, int, int]
    votes: List[Tuple[int, float]] = field(default_factory=list)
    ema: Optional[Tuple[float, float]] = None
    age: int = 0  # 连续未被匹配的帧数


class MaterialDetectNode(Node):
    """识别转盘上的物料：定位、定色、盘心锁定与目标追踪。"""

    def __init__(self) -> None:
        super().__init__('material_detect')
        self.declare_parameter('input_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('output_topic', '/vision/camera/material_annotated')
        self.declare_parameter('result_topic', '/vision/camera/material_results')
        self.declare_parameter('command_topic', '/vision/material_command')
        self.declare_parameter(
            'model_path', '~/logistics_vision_data/color_model/colors_knn.yml'
        )
        self.declare_parameter('display', True)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        # 层3 视差修正: 质心→底心(锥轴落地点=夹取目标), Z/h/r_ratio 参数化
        self.declare_parameter(
            'calibration_path',
            '~/logistics_vision_data/calibration/camera_640x480.yaml',
        )
        self.declare_parameter('camera_to_tray_mm', 285.0)
        self.declare_parameter('material_height_mm', 70.0)
        self.declare_parameter('material_radius_ratio', 0.6)

        # 分割与形状筛选
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
        self.declare_parameter('center_lock_frames', 10)
        self.declare_parameter('smoothing_alpha', 0.0)
        self.declare_parameter('motion_gate_px', 0)
        self.declare_parameter('stillness_motion_px', 1.5)
        self.declare_parameter('stillness_frames', 6)

        self.image_size = (
            int(self.get_parameter('image_width').value),
            int(self.get_parameter('image_height').value),
        )
        model_path = Path(str(self.get_parameter('model_path').value)).expanduser()
        self.knn = None
        self.knn_k = 3
        if model_path.is_file():
            self.knn, self.knn_k = self._load_knn(model_path)
        else:
            self.get_logger().warn(
                '未找到颜色模型: %s。彩色物料块将标记为 color=0（仅分割调试）；'
                '完成采样训练后重启节点即可启用判色。' % model_path
            )

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
        self.center_lock_frames = max(1, int(self.get_parameter('center_lock_frames').value))
        self.smoothing_alpha = float(self.get_parameter('smoothing_alpha').value)
        self.motion_gate_px = int(self.get_parameter('motion_gate_px').value)
        self.stillness_motion_px = float(self.get_parameter('stillness_motion_px').value)
        self.stillness_frames = int(self.get_parameter('stillness_frames').value)
        self.display = bool(self.get_parameter('display').value)

        # 层3 视差修正参数
        self.camera_to_tray_mm = float(self.get_parameter('camera_to_tray_mm').value)
        self.material_height_mm = float(self.get_parameter('material_height_mm').value)
        self.material_radius_ratio = float(
            self.get_parameter('material_radius_ratio').value
        )

        # 模式状态
        self.mode = 'idle'
        self.locate_colors: List[int] = []
        self.track_color: Optional[int] = None
        self.tracks: List[BlobTrack] = []
        self.next_track_id = 0
        self.locked_center: Optional[Tuple[int, int]] = None
        self.locked_radius_px: Optional[float] = None
        self.prev_locked_center: Optional[Tuple[int, int]] = None  # 重锁验证用: 上次锁定值
        self.centroid_history: deque = deque(maxlen=self.center_lock_frames)
        self.last_target: Optional[Tuple[float, float]] = None
        self.stationary = False
        self._last_motion_px: Optional[float] = None
        self.still_count = 0
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
            '物料识别节点已启动: 图像 %s，命令 %s，结果 %s。模式: locate/track/idle。'
            % (input_topic, command_topic, result_topic)
        )

    def _load_knn(self, path: Path):
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise RuntimeError('无法读取颜色模型: %s' % path)
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

    # ---------- 层3: 视差修正(质心→底心) ----------
    def _load_principal_point(self, path: Path) -> Tuple[float, float]:
        """从标定文件读主点 (cx, cy)；失败时退回图像中心并告警。"""
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
                '读取主点失败(%s)，视差修正退用图像中心(%.0f, %.0f)。'
                % (error, fallback[0], fallback[1])
            )
            return fallback

    def _parallax_k(self) -> float:
        """质心外偏比例 k 的解析值: k = W·(h/(Z−h))/(1+W), W = (r_ratio·Z/(Z−h))²。

        锥台俯视投影中顶面被放大 Z/(Z−h) 倍, blob 面积权重比 W 即
        (顶面投影半径/底面投影半径)² = (r_ratio·Z/(Z−h))², 质心相对底面轴心
        沿主点方向外偏 k。只依赖 Z/h/r_ratio 三个参数, 盘高/物料变化改参数即可。
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
            self.last_target = None
            self.get_logger().info('进入追踪模式: 目标 %d(%s)' % (color, color_name(color)))
        elif command == 'idle':
            self.mode = 'idle'
            self.track_color = None
            self._reset_session()
            self.get_logger().info('进入空闲模式。')
        else:
            self.get_logger().warn(
                '无法识别命令 "%s"。可用: locate <色1><色2><色3> / track <色> / idle'
                % message.data
            )

    def _reset_session(self) -> None:
        # 重发 locate = 重锁验证: 保留上次锁定值, 重锁完成后输出新旧偏差
        if self.locked_center is not None:
            self.prev_locked_center = self.locked_center
        self.tracks = []
        self.locked_center = None
        self.locked_radius_px = None
        self.centroid_history.clear()
        self.last_target = None

    # ---------- 主处理 ----------
    def _image_callback(self, message: Image) -> None:
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

        payload, annotated = self._process_frame(frame)
        output = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        output.header = message.header
        self.publisher.publish(output)
        if self.mode != 'idle':
            result = String()
            result.data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
            self.result_publisher.publish(result)
        if self.display:
            self._show_layer1(frame)
            self._show(annotated)

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
        cv2.imshow('Material Layer1 Masks', vis)

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
        # 识别范围随模式收窄: idle 六色全识别, locate 只认任务三色, track 只认
        # 当前目标色——调试窗口与 JSON 均只显示范围内物料; 范围外轨迹在内部
        # 照常维护(投票/老化不停), 换色/重发 locate 时无需重建
        if self.mode == 'locate':
            wanted = set(self.locate_colors)
            visible = [item for item in decided if item[1] in wanted]
        elif self.mode == 'track':
            visible = [item for item in decided if item[1] == self.track_color]
        else:
            visible = decided
        # 估计盘心: 已锁定后固定为锁定时刻的值——小车不动盘心不变,
        # 物料被抓减少时圆心/分布圈不再跟着变(锁定前的即时计算仅供预览)
        confirmed = [
            item for item in visible if item[2] >= self.min_confidence
        ]
        if self.locked_center is not None:
            payload['estimated_center'] = {
                'u': self.locked_center[0],
                'v': self.locked_center[1],
                'ring_radius_px': self.locked_radius_px,
                'count': len(confirmed),
            }
        elif len(confirmed) >= 2:
            bottoms = np.array(
                [self._bottom_center(track.center) for track, _, _ in confirmed],
                dtype=np.float64,
            )
            center = bottoms.mean(axis=0)
            radius = float(np.mean(np.hypot(
                bottoms[:, 0] - center[0], bottoms[:, 1] - center[1]
            )))
            payload['estimated_center'] = {
                'u': int(round(center[0])),
                'v': int(round(center[1])),
                'ring_radius_px': round(radius, 1),
                'count': len(bottoms),
            }
        if self.mode == 'locate':
            self._fill_locate(payload, visible)
        elif self.mode == 'track':
            self._fill_track(payload, visible)

        annotated = self._annotate(frame, visible, payload)
        return payload, annotated

    # ---------- 层1: 候选生成 ----------
    def _estimate_background(self, sat: np.ndarray, lab: np.ndarray) -> bool:
        """从画面估计白背景 Lab 基准 (μL, μa, μb)。

        白地图+白转盘是全场恒定的天然参照物: S<饱和度上限的像素(占画面 70%+)
        取 Lab 中位数。基准逐帧自适应光照, 阈值不再依赖具体曝光值。
        画面被大面积遮挡导致白像素不足时沿用上一帧基准。
        """
        bg = sat < self.saturation_threshold
        if int(bg.sum()) < 1000:
            return self._bg_ref is not None
        lab_f = lab.astype(np.float32)
        self._bg_ref = (
            float(np.median(lab_f[:, :, 0][bg])),
            float(np.median(lab_f[:, :, 1][bg])),
            float(np.median(lab_f[:, :, 2][bg])),
        )
        return True

    def _detect_blobs(self, frame: np.ndarray) -> List[dict]:
        """相对白背景分割：物料 = 与白背景 Lab 基准显著偏离的区域。

        彩料掩膜 = 色度差 |Δa|+|Δb| > color_delta_ab —— 覆盖六色中饱和度最低的
        绿/浅蓝(实测 Δab P5=29/31), 白背景自身 Δab max=18;
        黑料掩膜 = 亮度比 L < dark_l_ratio·μL 且无色度 Δab < dark_delta_ab ——
        暗+灰才是黑: 红/蓝物料 L 低(0.26~0.52·μL)但色度高(Δab≥44)不会误判黑,
        黑料 L 比值 0.08、Δab P95=6, 两个条件余量都在 2 倍以上。

        两张掩膜独立成块，贴画幅边界的连通域剔除（地图边缘边框/大片背景干扰），
        形状筛选后送判色——彩色块走 kNN（无模型时 color=0 仅调试），暗块按定义判黑。
        """
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        sat = hsv[:, :, 1]

        if not self._estimate_background(sat, lab):
            empty = np.zeros(frame.shape[:2], np.uint8)
            self._vis_mask_color = empty
            self._vis_mask_dark = empty
            return []
        lab_f = lab.astype(np.float32)
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
        blobs += self._extract_blobs(mask_color, frame, kind='color')
        blobs += self._extract_blobs(mask_dark, frame, kind='dark')
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

    def _extract_blobs(self, mask: np.ndarray, frame: np.ndarray, kind: str) -> List[dict]:
        height, width = mask.shape
        # RETR_LIST: 必须取全部轮廓——边框类闭合环绕住画面时, 环内物料轮廓
        # 属于层级子节点, RETR_EXTERNAL 会把环内物料整个吞掉
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        lab = None
        if kind == 'color' and self.knn is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

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
                if lab is None:
                    color, confidence = 0, 0.0
                else:
                    blob_mask = np.zeros(mask.shape, dtype=np.uint8)
                    cv2.drawContours(blob_mask, [contour], -1, 255, -1)
                    try:
                        features = color_feature_vector(lab, blob_mask)
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
        """投票定色：多数色 + 多帧一致性作为置信度。"""
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

        center_info = None
        if sorted(found.keys()) == sorted(wanted):
            points = np.array(
                [self._bottom_center(found[color].center) for color in sorted(wanted)],
                dtype=np.float64,
            )
            centroid = points.mean(axis=0)
            ring_radius = float(np.mean(np.hypot(
                points[:, 0] - centroid[0], points[:, 1] - centroid[1]
            )))
            if self.locked_center is None:
                # 锁定中: 累积三色齐帧的盘心, 满 center_lock_frames 取平均锁定
                self.centroid_history.append((centroid[0], centroid[1]))
                if len(self.centroid_history) >= self.center_lock_frames:
                    stacked = np.array(self.centroid_history)
                    locked = stacked.mean(axis=0)
                    self.locked_center = (int(round(locked[0])), int(round(locked[1])))
                    self.locked_radius_px = round(ring_radius, 1)
                else:
                    center_info = {
                        'locked': False,
                        'locking': True,
                        'u': int(round(centroid[0])),
                        'v': int(round(centroid[1])),
                        'progress': len(self.centroid_history),
                        'frames': self.center_lock_frames,
                        'ring_radius_px': round(ring_radius, 1),
                    }
            if self.locked_center is not None:
                # 已锁定: 车不动永不重算, 只输出锁定值与当前即时偏差(调试观察)
                deviation = float(np.hypot(
                    centroid[0] - self.locked_center[0],
                    centroid[1] - self.locked_center[1],
                ))
                center_info = {
                    'locked': True,
                    'u': self.locked_center[0],
                    'v': self.locked_center[1],
                    'ring_radius_px': self.locked_radius_px,
                    'deviation_px': round(deviation, 1),
                }
                if self.prev_locked_center is not None:
                    # 重锁验证: 新旧锁定盘心偏差 = 车微校准(正对圆心)的残余误差
                    center_info['relock_deviation_px'] = round(float(np.hypot(
                        self.locked_center[0] - self.prev_locked_center[0],
                        self.locked_center[1] - self.prev_locked_center[1],
                    )), 1)
        payload['turntable_center'] = center_info

    def _fill_track(self, payload: dict, decided: List[tuple]) -> None:
        payload['track_color'] = self.track_color
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
        if candidates:
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
        if self.locked_center is not None:
            # track 模式透出既有锁定盘心(locate 阶段锁的), 供主控继续使用
            payload['turntable_center'] = {
                'locked': True,
                'u': self.locked_center[0],
                'v': self.locked_center[1],
                'ring_radius_px': self.locked_radius_px,
            }

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

        center = payload.get('turntable_center')
        if center:
            if center.get('locked'):
                # 锁定盘心: 绿色大十字 + 物料分布圈
                cv2.circle(
                    annotated, (center['u'], center['v']),
                    int(center.get('ring_radius_px', 0)), (0, 255, 0), 1,
                )
                cv2.drawMarker(
                    annotated, (center['u'], center['v']), (0, 255, 0),
                    cv2.MARKER_CROSS, 26, 2,
                )
            else:
                # 锁定中: 黄色十字(即时平均位置)
                cv2.drawMarker(
                    annotated, (center['u'], center['v']), (0, 220, 220),
                    cv2.MARKER_CROSS, 22, 1,
                )
        estimated = payload.get('estimated_center')
        if estimated:
            eu, ev = estimated['u'], estimated['v']
            # 青色: 估计盘心 + 物料分布圈(观察 k 修正时与红十字对比)
            cv2.circle(annotated, (eu, ev), int(estimated['ring_radius_px']), (255, 255, 0), 1)
            cv2.drawMarker(annotated, (eu, ev), (255, 255, 0), cv2.MARKER_CROSS, 20, 1)
        # 白点: 主点(视差修正的收敛参考, 所有底心都朝它收 1-k)
        cv2.drawMarker(
            annotated,
            (int(round(self.principal_point[0])), int(round(self.principal_point[1]))),
            (255, 255, 255), cv2.MARKER_CROSS, 10, 1,
        )

        lines = ['mode: %s' % self.mode]
        if self.mode == 'locate':
            lines.append('colors: %s' % self.locate_colors)
        elif self.mode == 'track':
            lines.append('target: %d(%s)' % (self.track_color, color_abbr(self.track_color)))
        motion = 'n/a' if self._last_motion_px is None else '%.1fpx' % self._last_motion_px
        lines.append(
            'stationary: %s  motion: %s [%d/%d]'
            % (self.stationary, motion, self.still_count, self.stillness_frames)
        )
        if center is not None and center.get('locked'):
            dev = center.get('deviation_px')
            dev_text = ' dev=%.1fpx' % dev if dev is not None else ''
            relock = center.get('relock_deviation_px')
            relock_text = ' relock=%.1fpx' % relock if relock is not None else ''
            lines.append(
                'turntable: LOCKED @(%d,%d) r=%.0fpx%s%s'
                % (center['u'], center['v'], center.get('ring_radius_px', 0),
                   dev_text, relock_text)
            )
        elif center is not None and center.get('locking'):
            lines.append(
                'turntable: locking [%d/%d]'
                % (center.get('progress', 0), center.get('frames', 0))
            )
        else:
            lines.append('turntable: no lock (locate needs 3 colors)')
        lines.append('k=%.3f  RED=bottom grasp  YELLOW=centroid' % self._parallax_k())
        lines.append('CYAN=tray center est.  WHITE=principal pt')
        for index, text in enumerate(lines):
            cv2.putText(
                annotated, text, (10, 25 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return annotated

    def _show(self, annotated: np.ndarray) -> None:
        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        cv2.imshow(WINDOW_NAME, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令。')
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[MaterialDetectNode] = None
    try:
        node = MaterialDetectNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('material_detect 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
