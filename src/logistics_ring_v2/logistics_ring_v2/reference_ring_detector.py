"""Reference ring detector adapted from the user-supplied Vision project.

Source: D:/物流小车/代码/Vision/Vision/pi_vision/detectors/ring_detector.py.
Preserves Hough/contour fallback, concentric grouping, hollow validation and
ellipse refinement. The adapter configures native 640x480 and leaves temporal
smoothing to the existing result validator. No reference serial/UI code is used.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import threading

import cv2
import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class Ring:
    ring_no: int
    center_x: int
    center_y: int
    confidence: int
    radius: float = 0.0


class ReferenceRingDetector:
    def __init__(
        self,
        threshold: int = 160,
        min_radius: float = 25.0,
        max_radius: float = 400.0,
        min_circularity: float = 0.5,
        min_solidity: float = 0.7,
        center_merge_dist: float = 40.0,
        radius_merge_tol: float = 6.0,
        max_rings_per_cluster: int = 5,
        min_rings_per_cluster: int = 2,
        center_tolerance: float = 15.0,
        min_work_width: int = 960,
        min_work_height: int = 720,
        hough_enabled: bool = True,
        use_clahe: bool = True,
        blur_kernel: Tuple[int, int] = (5, 5),
        hough_dp: float = 1.5,
        hough_min_dist: float = 12.0,
        hough_param1: Optional[float] = None,
        hough_param2: float = 0.8,
        hough_method: Optional[int] = None,
        hollow_check: bool = True,
        hollow_threshold: float = 15.0,
        merge_clusters_ratio: float = 0.6,
        smooth_alpha: float = 0.0,
        refine_center: bool = True,
        refine_max_offset_ratio: float = 0.35,
    ) -> None:
        self.threshold = threshold
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_circularity = min_circularity
        self.min_solidity = min_solidity
        self.center_merge_dist = center_merge_dist
        self.radius_merge_tol = radius_merge_tol
        self.max_rings_per_cluster = max_rings_per_cluster
        self.min_rings_per_cluster = min_rings_per_cluster
        self.center_tolerance = center_tolerance
        self.min_work_width = min_work_width
        self.min_work_height = min_work_height
        self.hough_enabled = hough_enabled
        self.use_clahe = use_clahe
        self.blur_kernel = blur_kernel
        self.hough_dp = hough_dp
        self.hough_min_dist = hough_min_dist
        self.hough_param1 = hough_param1
        # 默认用 HOUGH_GRADIENT_ALT（OpenCV>=4.3）：精度高、误检少，对残缺圆弧鲁棒；
        # 旧版 OpenCV 无 ALT 时自动回退经典 GRADIENT。
        if hough_method is None:
            hough_method = getattr(cv2, "HOUGH_GRADIENT_ALT", None) or cv2.HOUGH_GRADIENT
        self.hough_method = hough_method
        self.hough_param2 = hough_param2
        # ALT 的 param2 是 0~1 的质量阈值；经典 GRADIENT 是累加器阈值（>10）。
        # 若用户按 ALT 语义传了 0~1 的值而实际用 GRADIENT，则换算成合理阈值。
        if self.hough_method == cv2.HOUGH_GRADIENT and self.hough_param2 < 1.0:
            self.hough_param2 = 40.0
        self.hollow_check = hollow_check
        self.hollow_threshold = hollow_threshold
        self.merge_clusters_ratio = merge_clusters_ratio
        self.smooth_alpha = smooth_alpha
        self.refine_center = refine_center
        self.refine_max_offset_ratio = refine_max_offset_ratio
        self._prev_rings = []
        self._smooth_lock = threading.Lock()
        self._kernel = np.ones((3, 3), dtype=np.uint8)

    def detect(self, frame: np.ndarray) -> List[Ring]:
        """返回圆环列表。

        检测路径（不完整形态也能识别）：
          1. 滤波 + 霍夫圆：CLAHE 光照均衡 -> 高斯去噪 -> HoughCircles，
             残缺的圆弧段也能投票出圆心（强光洗白/遮挡缺口的圆环同样适用）；
          2. 兜底：自适应阈值 + 轮廓 + 圆度/凸度过滤（霍夫无结果时启用）。

        每一组同心圆合并为一个 Ring 上报（一个中心坐标），ring_no 固定为 1。
        """
        work, scale = self._prepare(frame)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        circles = self._detect_hough(gray)
        if not circles:
            circles = self._detect_contours(work)
        clusters = self._cluster_circles(circles)
        # 上报前二次同心合并：斜俯视下环呈透视椭圆，霍夫圆在不同弧段拟合出的
        # 圆心可能有 60~67px 的系统性偏差，导致同一组同心环被拆成两个中心。
        # 用径向重叠判据（圆心距 < 0.6*(r_max1+r_max2)）把"同组环分裂"重新并回一个中心。
        clusters = self._merge_concentric_clusters(clusters)
        rings = []
        for center_x, center_y, radii, max_deviation in clusters:
            if max_deviation > self.center_tolerance:
                continue
            merged_radii = self._merge_radii(radii)
            if len(merged_radii) < self.min_rings_per_cluster:
                continue
            # 环状空心验证：真环环带是一条深色环线（中心透背景），
            # 而实心圆盘/反光斑/黑边中心与环带无明显深色对比。
            # 用 c - band_p5 >= hollow_threshold 过滤后类假环。
            if not self._hollow_ok(gray, center_x, center_y, max(merged_radii)):
                continue
            # 椭圆心精修：斜俯视下环是透视椭圆，正圆霍夫圆心有系统偏差，
            # 用环带边缘点 fitEllipse 求真实中心（偏移过大/退化则回退霍夫圆心）。
            cx_out = center_x
            cy_out = center_y
            if self.refine_center:
                refined = self._fit_ellipse_center(gray, center_x, center_y, max(merged_radii))
                if refined is not None:
                    cx_out, cy_out = refined
            # 一组同心圆只上报一个中心坐标（含单个圆环的情况）
            rings.append(
                Ring(
                    ring_no=1,
                    center_x=int(round(cx_out / scale)),
                    center_y=int(round(cy_out / scale)),
                    confidence=90,
                    # 半径用合并层的中位数，比最小半径更稳（不被单层候选骤变带偏）
                    radius=float(np.median(merged_radii)) / scale,
                )
            )
        return self._temporal_smooth(rings)

    def _fit_ellipse_center(self, gray, cx: float, cy: float, radius: float):
        """在环带边缘点上拟合椭圆，返回透视下的真实环中心；不可靠则返回 None。

        斜俯视下圆环呈透视椭圆，霍夫正圆拟合的圆心会沿投影轴偏移；环带(0.85~1.15r)
        的 Canny 边缘点 fitEllipse 的椭圆中心才是真实圆环中心。
        若边缘点不足 / 椭圆退化(长短轴比过大) / 中心偏移超阈值，回退 None（用霍夫圆心）。
        """
        if radius < 5:
            return None
        edges = cv2.Canny(gray, 50, 150)
        h, w = gray.shape
        ys, xs = np.ogrid[:h, :w]
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        band = (d2 <= (1.15 * radius) ** 2) & (d2 >= (0.85 * radius) ** 2)
        pts_y, pts_x = np.where(band & (edges > 0))
        pts = np.column_stack([pts_x, pts_y]).astype(np.float32)
        if pts.shape[0] < 12:
            return None
        try:
            (ex, ey), (MA, ma), _angle = cv2.fitEllipse(pts)
        except cv2.error:
            return None
        if MA <= 0 or ma <= 0:
            return None
        # 退化：长短轴比例过大（拟合到别的结构）
        if max(MA, ma) / (min(MA, ma) + 1e-6) > 3.0:
            return None
        # 中心偏移过大 → 不可靠，回退
        if ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5 > self.refine_max_offset_ratio * radius:
            return None
        return float(ex), float(ey)

    def _temporal_smooth(self, current: List[Ring]) -> List[Ring]:
        """时间 EMA 平滑：抑制圆心帧间漂移与半径跳变（被遮挡环突然变小等）。

        用最近邻匹配（中心距离 < max(r1,r2)*match_ratio）把当前环与上帧环关联，
        匹配则 EMA 平滑 中心/半径；未匹配（新出现）直接用当前值。
        上帧环若连续丢失则从状态中移除（避免残影）。
        """
        if self.smooth_alpha <= 0.0:
            self._prev_rings = list(current)
            return current
        with self._smooth_lock:
            prev = list(self._prev_rings)
            alpha = self.smooth_alpha
            out = []
            used_prev = set()
            for ring in current:
                best = None
                best_d = 1e18
                for idx, p in enumerate(prev):
                    d = ((ring.center_x - p.center_x) ** 2
                         + (ring.center_y - p.center_y) ** 2) ** 0.5
                    if d < best_d:
                        best_d = d
                        best = (idx, p)
                match_radius = max(ring.radius, best[1].radius) * 1.5 if best else 0.0
                if best is not None and best_d <= match_radius:
                    idx, p = best
                    used_prev.add(idx)
                    cx = alpha * ring.center_x + (1 - alpha) * p.center_x
                    cy = alpha * ring.center_y + (1 - alpha) * p.center_y
                    rr = alpha * ring.radius + (1 - alpha) * p.radius
                    out.append(
                        Ring(
                            ring_no=ring.ring_no,
                            center_x=int(round(cx)),
                            center_y=int(round(cy)),
                            confidence=ring.confidence,
                            radius=float(rr),
                        )
                    )
                else:
                    out.append(ring)
            # 忽略本轮未匹配的上帧环（让其自然失配清除）
            self._prev_rings = out
            return out

    def _hollow_ok(
        self, gray: np.ndarray, cx: float, cy: float, radius: float
    ) -> bool:
        """真环中心应透背景：环带内有显著比中心更暗的环线（c - band_p5 >= 阈值）。"""
        if not self.hollow_check or radius < 2:
            return True
        h, w = gray.shape
        rr = float(radius)
        ys, xs = np.ogrid[:h, :w]
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        center = gray[d2 <= (0.30 * rr) ** 2].mean()
        # band 采样贴近环线（Hough 的 r 即环线所在半径），用 0.88~1.0 r
        band = gray[(d2 <= (1.0 * rr) ** 2) & (d2 >= (0.88 * rr) ** 2)]
        if band.size == 0:
            return False
        band_p5 = float(np.percentile(band, 5))
        return (center - band_p5) >= self.hollow_threshold

    def _prepare(self, frame: np.ndarray):
        """小图放大到工作分辨率，返回 (work, scale)。"""
        work = frame
        scale = 1.0
        if (
            frame.shape[1] < self.min_work_width
            or frame.shape[0] < self.min_work_height
        ):
            scale = max(
                self.min_work_width / frame.shape[1],
                self.min_work_height / frame.shape[0],
            )
            work = cv2.resize(
                frame,
                (
                    int(round(frame.shape[1] * scale)),
                    int(round(frame.shape[0] * scale)),
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        return work, scale

    def _detect_hough(self, gray: np.ndarray) -> list:
        """滤波 + 霍夫圆：从（可能残缺的）边缘投票检出圆，返回 (x,y,r,area,1,1) 列表。"""
        if not self.hough_enabled:
            return []
        proc = gray
        if self.use_clahe:
            # CLAHE：均衡局部对比度，缓解强光洗白导致的边缘丢失
            proc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        # 高斯滤波去噪，避免把噪声点当成圆
        blurred = cv2.GaussianBlur(proc, self.blur_kernel, 0)
        param1 = self.hough_param1
        if param1 is None:
            # Canny 高阈值按中位数自适应
            med = float(np.median(blurred))
            param1 = max(10.0, min(255.0, 1.33 * med))
        try:
            result = cv2.HoughCircles(
                blurred,
                self.hough_method,
                self.hough_dp,
                self.hough_min_dist,
                param1=param1,
                param2=self.hough_param2,
                minRadius=int(self.min_radius),
                maxRadius=int(self.max_radius),
            )
        except cv2.error:
            return []
        if result is None:
            return []
        out = []
        for cx, cy, r in result[0]:
            out.append((float(cx), float(cy), float(r), float(r * r), 1.0, 1.0))
        return out

    def _detect_contours(self, work: np.ndarray) -> list:
        """兜底路径：自适应阈值 + 轮廓 + 形状过滤，返回 (x,y,r,area,circ,solidity) 列表。"""
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,
            10,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if radius < self.min_radius or radius > self.max_radius:
                continue
            if circularity < self.min_circularity or solidity < self.min_solidity:
                continue
            circles.append((x, y, radius, area, circularity, solidity))
        return circles

    def _merge_radii(self, radii) -> List[float]:
        """把半径相近的重复轮廓合并，按从小到大保留环层。"""
        if not radii:
            return []
        sorted_radii = sorted(radii)
        merged = [sorted_radii[0]]
        for radius in sorted_radii[1:]:
            if abs(radius - merged[-1]) <= self.radius_merge_tol:
                merged[-1] = (merged[-1] + radius) / 2.0
            else:
                merged.append(radius)
        return merged

    def _cluster_circles(self, circles):
        """把同心圆聚成一组，返回 (中心x, 中心y, 半径列表)。"""
        clusters = []
        for x, y, radius, area, _, _ in circles:
            target = None
            for index, (xs, ys, radii, weights) in enumerate(clusters):
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                if (x - cx) ** 2 + (y - cy) ** 2 <= self.center_merge_dist ** 2:
                    target = index
                    break
            if target is None:
                clusters.append([[x], [y], [radius], [area]])
            else:
                clusters[target][0].append(x)
                clusters[target][1].append(y)
                clusters[target][2].append(radius)
                clusters[target][3].append(area)

        result = []
        for xs, ys, radii, weights in clusters:
            weight_sum = sum(weights) or 1.0
            center_x = sum(x * w for x, w in zip(xs, weights)) / weight_sum
            center_y = sum(y * w for y, w in zip(ys, weights)) / weight_sum
            max_deviation = max(
                ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                for x, y in zip(xs, ys)
            )
            result.append((center_x, center_y, radii, max_deviation))
        return result

    def _merge_concentric_clusters(self, clusters):
        """把圆心接近、圆盘显著重叠（径向重叠判据）的同心环组二次合并成一个中心。

        针对斜俯视/透视椭圆下霍夫圆圆心偏差导致"同一组同心环被拆成两个中心"的问题。
        判据：两簇圆心距离 <= merge_ratio * (r_max1 + r_max2)（默认 0.6），
        即两圆盘显著重叠 → 视为同一同心环组；独立靶面圆心距离远，不满足。
        """
        if not clusters:
            return []
        merge_ratio = self.merge_clusters_ratio
        merged = []
        for cx, cy, radii, dev in clusters:
            r_max = max(radii) if radii else 0.0
            placed = False
            for m in merged:
                mx = sum(m[0]) / len(m[0])
                my = sum(m[1]) / len(m[1])
                mr_max = max(sum(m[2], [])) if m[2] else 0.0
                dist = ((cx - mx) ** 2 + (cy - my) ** 2) ** 0.5
                if dist <= merge_ratio * (r_max + mr_max):
                    m[0].append(cx)
                    m[1].append(cy)
                    m[2].append(radii)
                    m[3].append(dev)
                    placed = True
                    break
            if not placed:
                merged.append([[cx], [cy], [radii], [dev]])
        result = []
        for xs, ys, radii_list, dev_list in merged:
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            radii = [_r for radii in radii_list for _r in radii]
            result.append((center_x, center_y, radii, max(dev_list)))
        return result
