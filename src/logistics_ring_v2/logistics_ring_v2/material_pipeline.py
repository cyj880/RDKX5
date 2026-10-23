"""共享物料识别管线: 分割/判色/跟踪/停稳/视差, 供 all/locate/track 三节点复用。

从 material2_detect_node.py(已退役)迁出的与模式无关的全部算法, 不依赖 rclpy——
仿 stacked_ring_algorithm 的"算法类 + 薄节点"分层: 本模块可脱离 ROS 单测,
三个物料节点(material_all/material_locate/material_track)各自持有一个
MaterialPipeline 实例, 模式差异只存在于各自的输出填充与命令语义。

管线内容(与退役单节点逐方法一致):
    白背景 Lab 基准估计(每帧自适应) → 彩料/黑料双掩膜 → 形态学 →
    面积/实心度/圆形度筛选 → 近距块合并(暗块优先判黑) → 相对特征 kNN 判色 →
    跨帧最近邻跟踪 + 颜色投票 → 停稳判断 → 视差底心(夹取目标)。

三个节点各自的差异(不在本模块):
    all    = 六色全部输出, 不生成抓取 target
    locate = 指定颜色定位(locate_colors), 每色一条
    track  = 单目标追踪 + 停稳门控(grasp_ready/target)

三方一致性约束(改任何一处必须同步):
    material2_sample_capture  采集时算 bg_ref 存 sidecar
    material2_color_train     训练时读 sidecar 的 bg_ref
    本模块                     在线每帧现算 bg_ref
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .material_features import (
    color_abbr,
    color_feature_vector_relative,
    estimate_background_ref,
    FEATURE_TAG,
)


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


@dataclass
class PipelineFrame:
    """一帧管线输出: 节点在此之上做模式过滤与结果填充。"""

    decided: List[tuple]  # [(BlobTrack, color, confidence), ...] 投票达半数窗口的轨迹
    stationary: bool
    motion_speed: Optional[float]  # 本帧运动量 px/s(各匹配轨迹速度最大值); 无数据为 None
    mask_color: Optional[np.ndarray]
    mask_dark: Optional[np.ndarray]


def stamp_seconds(message) -> float:
    """从 ROS 消息 header 提取秒级时间戳; 零值返回 0(由调用方回退单调时钟)。"""
    stamp = message.header.stamp
    value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return value if value > 0.0 else 0.0


# 热调参数表: 名称 -> (管线属性, 类型, 下限, 上限)。
# 现场调参即"我们的滑条": ros2 param set 经节点回调进入 set_param, 当帧生效。
TUNABLE_PARAMETERS = {
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
    # 视差三参数: 现场看红十字是否落在物料底面圆心
    'camera_to_tray_mm': ('camera_to_tray_mm', float, 50.0, 500.0),
    'material_height_mm': ('material_height_mm', float, 10.0, 200.0),
    'material_radius_ratio': ('material_radius_ratio', float, 0.1, 1.0),
    # 停稳状态机(cyqm 式, 单位 px/s): 转动确认阈值/停稳阈值/转动确认帧数
    'moving_speed_px_s': ('moving_speed_px_s', float, 1.0, 300.0),
    'stillness_speed_px_s': ('stillness_speed_px_s', float, 0.5, 300.0),
    'motion_confirm_frames': ('motion_confirm_frames', int, 1, 30),
    'stillness_frames': ('stillness_frames', int, 1, 60),
    # 初始停稳旁路: 从未见到运动且持续低速满该帧数 → 直接判停稳(防死锁)
    'initial_still_frames': ('initial_still_frames', int, 1, 300),
}

# ROS 参数声明表: 名称 -> 默认值(declare_pipeline_parameters 使用)
PARAMETER_DEFAULTS = {
    'model_path': '~/logistics_vision_data/color_model/colors_knn_v2.yml',
    'calibration_path': '~/logistics_vision_data/calibration/camera_640x480.yaml',
    'image_width': 640,
    'image_height': 480,
    'camera_to_tray_mm': 200.0,
    'material_height_mm': 70.0,
    'material_radius_ratio': 0.6,
    'saturation_threshold': 40,
    'color_delta_ab': 25.0,
    'dark_l_ratio': 0.55,
    'dark_delta_ab': 15.0,
    'dark_open_kernel': 7,
    'merge_distance_px': 40,
    'min_blob_area': 500,
    'max_blob_area': 40000,
    'min_solidity': 0.70,
    'min_circularity': 0.30,
    'min_confidence': 0.6,
    'vote_frames': 5,
    'match_distance_px': 40,
    'track_max_age': 8,
    'smoothing_alpha': 0.4,
    'motion_gate_px': 0,
    'moving_speed_px_s': 40.0,
    'stillness_speed_px_s': 10.0,
    'motion_confirm_frames': 2,
    'stillness_frames': 6,
    'initial_still_frames': 60,
}


class MaterialPipeline:
    """有状态物料识别算法, 独立于 ROS 传输层。"""

    def __init__(
        self,
        *,
        model_path: str,
        calibration_path: str,
        image_size: Tuple[int, int] = (640, 480),
        camera_to_tray_mm: float = 200.0,
        material_height_mm: float = 70.0,
        material_radius_ratio: float = 0.6,
        saturation_threshold: int = 40,
        color_delta_ab: float = 25.0,
        dark_l_ratio: float = 0.55,
        dark_delta_ab: float = 15.0,
        dark_open_kernel: int = 7,
        merge_distance_px: float = 40.0,
        min_blob_area: int = 500,
        max_blob_area: int = 40000,
        min_solidity: float = 0.70,
        min_circularity: float = 0.30,
        min_confidence: float = 0.6,
        vote_frames: int = 5,
        match_distance_px: float = 40.0,
        track_max_age: int = 8,
        smoothing_alpha: float = 0.0,
        motion_gate_px: int = 0,
        moving_speed_px_s: float = 40.0,
        stillness_speed_px_s: float = 10.0,
        motion_confirm_frames: int = 2,
        stillness_frames: int = 6,
        initial_still_frames: int = 60,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._log = log or (lambda message: None)
        self.image_size = tuple(image_size)
        # 数值参数直接落为属性(与 TUNABLE_PARAMETERS 的属性名一致, 热调即改属性)
        self.camera_to_tray_mm = float(camera_to_tray_mm)
        self.material_height_mm = float(material_height_mm)
        self.material_radius_ratio = float(material_radius_ratio)
        self.saturation_threshold = int(saturation_threshold)
        self.color_delta_ab = float(color_delta_ab)
        self.dark_l_ratio = float(dark_l_ratio)
        self.dark_delta_ab = float(dark_delta_ab)
        self.dark_open_kernel = max(1, int(dark_open_kernel))
        self.merge_distance_px = float(merge_distance_px)
        self.min_blob_area = int(min_blob_area)
        self.max_blob_area = int(max_blob_area)
        self.min_solidity = float(min_solidity)
        self.min_circularity = float(min_circularity)
        self.min_confidence = float(min_confidence)
        self.vote_frames = max(1, int(vote_frames))
        self.match_distance = float(match_distance_px)
        self.track_max_age = int(track_max_age)
        self.smoothing_alpha = float(smoothing_alpha)
        self.motion_gate_px = int(motion_gate_px)
        self.moving_speed_px_s = float(moving_speed_px_s)
        self.stillness_speed_px_s = float(stillness_speed_px_s)
        self.motion_confirm_frames = int(motion_confirm_frames)
        self.stillness_frames = int(stillness_frames)
        self.initial_still_frames = int(initial_still_frames)

        self.tracks: List[BlobTrack] = []
        self.next_track_id = 0
        # 停稳状态机(cyqm 式): 先见运动→接近停稳; 初始低速旁路防"转盘恰停"死锁
        self.motion_confirmed = False
        self.motion_frames = 0
        self.still_count = 0
        self.low_speed_total = 0
        self.stationary = False
        self.motion_speed: Optional[float] = None  # 本帧运动量 px/s
        self._last_timestamp: Optional[float] = None
        # 停稳速度统计的颜色过滤: None=统计全部轨迹; track 节点设为目标色,
        # 使影子/杂物块的关联churn不污染停稳判定(抓取关心的是目标物料动没动)
        self.stillness_track_color: Optional[int] = None
        # 白背景 Lab 基准 (μL, μa, μb), 每帧刷新; 画面被大面积遮挡时沿用旧值
        self._bg_ref: Optional[Tuple[float, float, float]] = None
        self._vis_mask_color: Optional[np.ndarray] = None
        self._vis_mask_dark: Optional[np.ndarray] = None

        self.knn = None
        self.knn_k = 3
        self.model_detail = ''
        self._load_knn(Path(model_path).expanduser())
        self.principal_point = self._load_principal_point(
            Path(calibration_path).expanduser())

    # ---------- 参数热调 ----------
    def set_param(self, name: str, value) -> None:
        """热调入口: 按 TUNABLE_PARAMETERS 校验类型与范围后当帧生效。

        校验失败抛 ValueError, 由节点回调转为拒绝该次 param set。
        """
        entry = TUNABLE_PARAMETERS.get(name)
        if entry is None:
            raise ValueError('参数 %s 不支持热调' % name)
        attr_name, expected_type, low, high = entry
        if expected_type is int:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError('%s 必须是整数' % name)
        else:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError('%s 必须是数值' % name)
        value = expected_type(value)
        if not low <= value <= high:
            raise ValueError('%s 超出范围 [%s, %s]' % (name, low, high))
        if name == 'dark_open_kernel' and value % 2 == 0:
            value += 1
        setattr(self, attr_name, value)

    # ---------- 模型与标定 ----------
    def _load_knn(self, path: Path) -> None:
        """加载 kNN 颜色模型; 特征口径不符时拒绝加载(退化为 color=0 调试模式)。"""
        try:
            storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            if not storage.isOpened():
                raise RuntimeError('无法读取颜色模型: %s' % path)
            tag = storage.getNode('features').string()
            if tag != FEATURE_TAG:
                storage.release()
                self.model_detail = (
                    '颜色模型特征口径不符: %s (期望 %s, 实际 %s), 拒绝加载。'
                    '请用 material2_sample_capture + material2_color_train 重新训练。'
                    % (path, FEATURE_TAG, tag or '(缺失)'))
                return
            samples = storage.getNode('samples').mat()
            labels = storage.getNode('labels').mat()
            k = int(storage.getNode('k').real())
            storage.release()
            if samples is None or labels is None or len(samples) == 0:
                raise RuntimeError('颜色模型内容无效: %s' % path)
            knn = cv2.ml.KNearest_create()
            knn.setDefaultK(k)
            knn.train(samples, cv2.ml.ROW_SAMPLE, labels.astype(np.int32))
            self.knn, self.knn_k = knn, k
        except (RuntimeError, cv2.error) as error:
            self.model_detail = str(error)

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
            self._log('读取主点失败(%s), 视差修正退用图像中心(%.0f, %.0f)。'
                      % (error, fallback[0], fallback[1]))
            return fallback

    # ---------- 视差修正(质心→底心) ----------
    def parallax_k(self) -> float:
        """质心外偏比例 k 的解析值: k = W·(h/(Z−h))/(1+W), W = (r_ratio·Z/(Z−h))²。

        锥台俯视投影中顶面被放大 Z/(Z−h) 倍, blob 面积权重比 W 即
        (顶面投影半径/底面投影半径)², 质心相对底面轴心沿主点方向外偏 k。
        Z=200,h=70,r_ratio=0.6 → k=0.248。
        """
        z = max(1.0, self.camera_to_tray_mm)
        h = min(self.material_height_mm, z - 1.0)
        ratio = z / (z - h)
        w = (self.material_radius_ratio * ratio) ** 2
        return w * (h / (z - h)) / (1.0 + w)

    def bottom_center(self, centroid: Tuple[int, int]) -> Tuple[int, int]:
        """质心→底心: 底心 = 主点 + (质心−主点)·(1−k)。

        底心即锥轴在盘面的落点, 是机械臂的夹取目标。
        """
        k = self.parallax_k()
        px, py = self.principal_point
        return (
            int(round(px + (centroid[0] - px) * (1.0 - k))),
            int(round(py + (centroid[1] - py) * (1.0 - k))),
        )

    # ---------- 会话 ----------
    def reset(self) -> None:
        """会话复位: 清轨迹与停稳结论, 新会话必须重新确认停稳。"""
        self.tracks = []
        self.motion_confirmed = False
        self.motion_frames = 0
        self.still_count = 0
        self.low_speed_total = 0
        self.stationary = False
        self.motion_speed = None
        self._last_timestamp = None

    # ---------- 主处理 ----------
    def process(self, frame: np.ndarray,
                timestamp_s: Optional[float] = None) -> PipelineFrame:
        """处理一帧 BGR 图; timestamp_s 用于速度计算(px/s), 缺省则无速度数据。

        帧间隔异常(<5ms 或 >500ms, 丢帧/时钟跳变)时重置测速基线,
        本帧不产生速度数据(停稳状态保持, 与 cyqm 的帧间隔有效性检查一致)。
        """
        interval_s = None
        if timestamp_s is not None and self._last_timestamp is not None:
            interval_s = timestamp_s - self._last_timestamp
            if not 0.005 <= interval_s <= 0.5:
                self._log('帧间隔 %.3fs 无效, 重置测速基线。' % interval_s)
                interval_s = None
        if timestamp_s is not None:
            self._last_timestamp = timestamp_s

        blobs = self._detect_blobs(frame)
        self.motion_speed = self._update_tracks(blobs, interval_s)
        self._update_stillness()
        decided = [
            (track,) + self._decided_color(track)
            for track in self.tracks
            if len(track.votes) >= max(1, self.vote_frames // 2)
        ]
        return PipelineFrame(
            decided=decided,
            stationary=self.stationary,
            motion_speed=self.motion_speed,
            mask_color=self._vis_mask_color,
            mask_dark=self._vis_mask_dark,
        )

    def _update_stillness(self) -> None:
        """停稳状态机(借鉴 cyqm material_pickup_detector 的运动状态机)。

        速度 = 各匹配轨迹的帧间位移/帧间隔 (px/s), 取最大值——单个物料
        运动也会被捕捉, 不会被均值掩盖。三种路径:
          1. 先见运动: 速度≥moving_speed_px_s 连续 motion_confirm_frames 帧
             → motion_confirmed; 之后速度≤stillness_speed_px_s 连续
             stillness_frames 帧 → stationary=true, 超阈立即 false。
          2. 初始停稳旁路: 从未确认运动且持续低速满 initial_still_frames 帧
             → 直接 stationary=true(车到时转盘恰好停着/未启动, 不必等运动)。
          3. 无有效速度数据时取消停稳，清空连续观测计数。
        """
        motion = self.motion_speed
        if motion is None:
            self.stationary = False
            self.motion_frames = 0
            self.still_count = 0
            self.low_speed_total = 0
            return
        if motion >= self.moving_speed_px_s:
            self.motion_frames += 1
            self.still_count = 0
            self.low_speed_total = 0
            self.stationary = False
            if self.motion_frames >= self.motion_confirm_frames:
                self.motion_confirmed = True
        elif motion > self.stillness_speed_px_s:
            # The band between moving and still thresholds is not evidence of rest.
            self.stationary = False
            self.motion_frames = 0
            self.still_count = 0
            self.low_speed_total = 0
        elif self.motion_confirmed:
            self.motion_frames = 0
            self.low_speed_total = 0
            self.still_count += 1
            if self.still_count >= self.stillness_frames:
                self.stationary = True
        else:
            # 尚未确认运动: 低速累计走初始停稳旁路
            self.motion_frames = 0
            self.still_count = 0
            self.low_speed_total += 1
            if self.low_speed_total >= self.initial_still_frames:
                self.motion_confirmed = True
                self.stationary = True

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
        l_ch, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
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
        blobs += self._extract_blobs(mask_color, 'color', lab, self._bg_ref)
        blobs += self._extract_blobs(mask_dark, 'dark', lab, self._bg_ref)
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
    def _update_tracks(self, blobs: List[dict],
                       interval_s: Optional[float]) -> Optional[float]:
        """跨帧轨迹匹配+投票; 返回本帧运动量 px/s(最大轨迹速度), 无数据为 None。"""
        matched_speeds: List[float] = []
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
                track_color, _ = self._decided_color(track)
                if (interval_s and track.age == 0
                        and (self.stillness_track_color is None
                             or track_color == self.stillness_track_color)):
                    # 停稳速度统计可按颜色过滤(track 节点设目标色):
                    # 影子/杂物块的关联churn不再污染停稳判定
                    matched_speeds.append(best_distance / interval_s)
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
        # 本帧运动量 = 已匹配轨迹速度的最大值(单个物料运动也不会被均值掩盖);
        # 无匹配或无有效帧间隔 → None(停稳状态保持)
        return float(np.max(matched_speeds)) if matched_speeds else None

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

    # ---------- 标注 ----------
    def annotate(self, frame: np.ndarray, visible: List[tuple],
                 payload: dict, hud_lines: Tuple[str, ...] = ()) -> np.ndarray:
        """共享标注: 物料框/标签/视差三件套/TARGET/公共 HUD; 模式行由 hud_lines 传入。"""
        annotated = frame.copy()
        for track, color, confidence in visible:
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
            cv2.circle(annotated, track.center, 3, (0, 255, 255), 1)
            if payload.get('distance_valid', True):
                bottom = self.bottom_center(track.center)
                cv2.line(annotated, track.center, bottom, (255, 0, 255), 1)
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

        motion = 'n/a' if self.motion_speed is None else '%.1fpx/s' % self.motion_speed
        lines = ['mode: %s' % payload.get('mode', '--')]
        lines.extend(hud_lines)
        lines.append(
            'stationary: %s  motion: %s [%d/%d]'
            % (self.stationary, motion, self.still_count, self.stillness_frames)
        )
        lines.append('k=%.3f  RED=bottom grasp  YELLOW=centroid' % self.parallax_k())
        lines.append('WHITE=principal pt')
        for index, text in enumerate(lines):
            cv2.putText(
                annotated, text, (10, 25 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return annotated


# ---------- ROS 侧辅助(节点用; 本函数区依赖 rcl_interfaces, 不依赖 rclpy 运行时) ----------
def declare_pipeline_parameters(node) -> None:
    """在节点上声明全部管线参数(三个物料节点复用, 不复制粘贴)。"""
    for name, default in PARAMETER_DEFAULTS.items():
        node.declare_parameter(name, default)


def pipeline_from_node(node) -> MaterialPipeline:
    """从节点已声明的参数装配管线; 加载告警经节点日志输出。"""
    values = {name: node.get_parameter(name).value
              for name in PARAMETER_DEFAULTS}
    values['image_size'] = (
        int(values.pop('image_width')), int(values.pop('image_height')))
    return MaterialPipeline(log=node.get_logger().warn, **values)


def attach_parameter_hot_tune(node, pipeline) -> None:
    """把管线热调参数接到节点 param set 回调; 校验失败拒绝该次设置。"""
    from rcl_interfaces.msg import SetParametersResult

    def _on_set_parameters(parameters):
        for parameter in parameters:
            if parameter.name not in TUNABLE_PARAMETERS:
                continue
            try:
                pipeline.set_param(parameter.name, parameter.value)
            except ValueError as error:
                return SetParametersResult(successful=False, reason=str(error))
            node.get_logger().info(
                '参数 %s 热更新为 %s' % (parameter.name, parameter.value))
        return SetParametersResult(successful=True)

    node.add_on_set_parameters_callback(_on_set_parameters)
