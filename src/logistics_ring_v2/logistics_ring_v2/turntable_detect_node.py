"""Turntable outer-circle detection node.

The material foreground is used only as a coarse seed.  The published center
always comes from a refined outer-circle fit, never directly from the seed.
"""

from collections import deque
import json
import math
from itertools import combinations
from typing import Optional, Sequence

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class TurntableDetectNode(Node):
    def __init__(self) -> None:
        super().__init__('turntable_detect')
        self.declare_parameter('input_topic', '/vision/camera/image_undistorted')
        self.declare_parameter('annotated_topic', '/vision/turntable/annotated')
        self.declare_parameter('result_topic', '/vision/turntable/results')
        self.declare_parameter('command_topic', '/vision/turntable/command')
        self.declare_parameter('display', False)
        self.declare_parameter('start_mode', 'stop')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('reference_center_u', 320.0)
        self.declare_parameter('reference_center_v', 240.0)
        self.declare_parameter('reference_radius_px', 185.0)
        self.declare_parameter('seed_roi_margin_px', 90.0)
        self.declare_parameter('seed_min_area_px', 250.0)
        self.declare_parameter('seed_max_area_px', 20000.0)
        self.declare_parameter('seed_min_solidity', 0.55)
        self.declare_parameter('seed_min_spacing_px', 35.0)
        self.declare_parameter('seed_min_orbit_radius_px', 45.0)
        self.declare_parameter('seed_max_orbit_radius_px', 180.0)
        self.declare_parameter('coarse_scale', 0.35)
        self.declare_parameter('coarse_min_radius_px', 120.0)
        self.declare_parameter('coarse_max_radius_px', 230.0)
        self.declare_parameter('coarse_param1', 60.0)
        self.declare_parameter('coarse_param2', 28.0)
        self.declare_parameter('coarse_min_dist_px', 80.0)
        self.declare_parameter('refine_half_width_px', 28.0)
        self.declare_parameter('angular_step_deg', 4.0)
        self.declare_parameter('minimum_visible_arc_deg', 90.0)
        self.declare_parameter('maximum_fit_rms_px', 7.0)
        self.declare_parameter('minimum_edge_strength', 4.0)
        self.declare_parameter('stable_frames', 5)
        self.declare_parameter('maximum_center_jitter_px', 10.0)
        self.declare_parameter('maximum_radius_jitter_px', 5.0)
        self.declare_parameter('local_recheck_confidence_below', 0.70)
        self.declare_parameter('local_recheck_rms_above_px', 4.0)
        self.declare_parameter('local_recheck_interval_frames', 5)
        self.declare_parameter('reacquisition_jump_px', 15.0)
        self.declare_parameter('seed_hold_frames', 3)

        self.width = int(self.get_parameter('image_width').value)
        self.height = int(self.get_parameter('image_height').value)
        self.reference = np.array([
            float(self.get_parameter('reference_center_u').value),
            float(self.get_parameter('reference_center_v').value),
        ], dtype=np.float32)
        self.reference_radius = float(self.get_parameter('reference_radius_px').value)
        self.mode = str(self.get_parameter('start_mode').value).lower()
        self.mode = 'run' if self.mode == 'run' else 'stop'
        self.bridge = CvBridge()
        self.annotated_pub = self.create_publisher(
            Image, str(self.get_parameter('annotated_topic').value), 10)
        self.result_pub = self.create_publisher(
            String, str(self.get_parameter('result_topic').value), 10)
        self.create_subscription(
            String, str(self.get_parameter('command_topic').value),
            self._command_callback, 10)
        self.create_subscription(
            Image, str(self.get_parameter('input_topic').value),
            self._image_callback, 10)
        self.samples = deque(maxlen=max(1, int(self.get_parameter('stable_frames').value)))
        self.seed = None
        self.seed_miss_frames = 0
        self.frames_since_global_recheck = 0
        self.last_result = None
        self._last_shape_error = False
        self.get_logger().info('转盘节点已启动，门控=%s。' % self.mode)

    def _command_callback(self, message: String) -> None:
        command = message.data.strip().lower()
        if command in ('run', 'stop'):
            self.mode = command
            if command == 'stop':
                self.samples.clear()
                self.seed = None
                self.seed_miss_frames = 0
                self.frames_since_global_recheck = 0
                self.last_result = None
            self.get_logger().info('转盘门控: %s' % command)
        else:
            self.get_logger().warn('转盘命令无效: %s，可用 run / stop' % message.data)

    def _image_callback(self, message: Image) -> None:
        if self.mode != 'run':
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error('转盘图像转换失败: %s' % error)
            return
        if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
            if not self._last_shape_error:
                self.get_logger().error('转盘输入分辨率为 %dx%d，预期 %dx%d。' % (
                    frame.shape[1], frame.shape[0], self.width, self.height))
            self._last_shape_error = True
            return
        self._last_shape_error = False
        if self.mode != 'run':
            return
        result, annotated = self._detect(frame)
        result['mode'] = 'turntable'
        result['gate'] = self.mode
        output = String()
        output.data = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        self.result_pub.publish(output)
        image_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        image_msg.header = message.header
        self.annotated_pub.publish(image_msg)
        self.last_result = result

    def _detect(self, frame: np.ndarray) -> tuple[dict, np.ndarray]:
        annotated = frame.copy()
        best = None
        tracking_mode = 'local'
        had_seed = self.seed is not None
        if had_seed:
            best = self._refine(frame, self.seed)
            self.frames_since_global_recheck += 1
            needs_recheck = (best is None or
                             best['confidence'] < float(self.get_parameter('local_recheck_confidence_below').value) or
                             best['fit_rms_px'] > float(self.get_parameter('local_recheck_rms_above_px').value))
            interval = max(1, int(self.get_parameter('local_recheck_interval_frames').value))
            if best is None or (needs_recheck and self.frames_since_global_recheck >= interval):
                best = self._global_refine(frame)
                tracking_mode = 'reacquired'
                self.frames_since_global_recheck = 0
        else:
            best = self._global_refine(frame)
            tracking_mode = 'reacquired'

        if best is None:
            self.samples.clear()
            self.seed_miss_frames += 1
            if self.seed_miss_frames > int(self.get_parameter('seed_hold_frames').value):
                self.seed = None
                tracking_mode = 'search'
            result = self._empty('no reliable outer circle')
            cv2.putText(annotated, 'TURNTABLE: SEARCHING', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            return result, annotated

        if had_seed:
            jump = max(float(np.linalg.norm(np.array([best['u'], best['v']]) - self.seed[:2])),
                       abs(float(best['radius']) - float(self.seed[2])))
            if jump > float(self.get_parameter('reacquisition_jump_px').value):
                self.samples.clear()
        self.seed = np.array([best['u'], best['v'], best['radius']], dtype=np.float32)
        self.seed_miss_frames = 0
        self.samples.append(best)
        center = np.median(np.array([[s['u'], s['v']] for s in self.samples]), axis=0)
        radius = float(np.median([s['radius'] for s in self.samples]))
        center_jitter = max(float(np.linalg.norm(np.array([s['u'], s['v']]) - center))
                            for s in self.samples)
        radius_jitter = max(abs(float(s['radius']) - radius) for s in self.samples)
        stable = (len(self.samples) == self.samples.maxlen and
                  center_jitter <= float(self.get_parameter('maximum_center_jitter_px').value) and
                  radius_jitter <= float(self.get_parameter('maximum_radius_jitter_px').value))
        cv2.circle(annotated, (int(round(center[0])), int(round(center[1]))),
                   int(round(radius)), (0, 220, 0), 2)
        cv2.drawMarker(annotated, (int(round(center[0])), int(round(center[1]))),
                       (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        text = 'TURNTABLE %s center=(%d,%d)' % (
            'STABLE' if stable else 'ACQUIRING', round(center[0]), round(center[1]))
        cv2.putText(annotated, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0) if stable else (0, 200, 255), 2, cv2.LINE_AA)
        result = dict(best)
        result.update({
            'detected': True, 'stable': stable,
            'center_u': int(round(center[0])), 'center_v': int(round(center[1])),
            'radius_px': radius, 'tracking_mode': tracking_mode,
        })
        result.pop('u', None); result.pop('v', None); result.pop('radius', None)
        return result, annotated

    def _global_refine(self, frame: np.ndarray) -> Optional[dict]:
        """全局霍夫粗定位，三物料种子可作为候选约束和回退。"""
        material_seed = self._material_seed(frame)
        candidates = self._coarse_candidates(frame, material_seed)
        if material_seed is not None:
            candidates.append(material_seed)
        best = None
        for candidate in candidates:
            fit = self._refine(frame, candidate)
            if fit is not None and (best is None or fit['confidence'] > best['confidence']):
                best = fit
        # A spurious material seed must not suppress the unrestricted search.
        if best is None and material_seed is not None:
            for candidate in self._coarse_candidates(frame, None):
                fit = self._refine(frame, candidate)
                if fit is not None and (best is None or fit['confidence'] > best['confidence']):
                    best = fit
        return best

    def _empty(self, detail: str) -> dict:
        return {'detected': False, 'stable': False, 'center_u': None,
                'center_v': None, 'radius_px': None, 'visible_arc_deg': 0.0,
                'fit_rms_px': None, 'confidence': 0.0,
                'tracking_mode': 'search', 'detail': detail}

    def _material_seed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        margin = float(self.get_parameter('seed_roi_margin_px').value)
        x0 = max(0, int(self.reference[0] - self.reference_radius - margin))
        y0 = max(0, int(self.reference[1] - self.reference_radius - margin))
        x1 = min(self.width, int(self.reference[0] + self.reference_radius + margin))
        y1 = min(self.height, int(self.reference[1] + self.reference_radius + margin))
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        lab = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = hsv[:, :, 1] < 45
        ref = np.median(lab[white], axis=0) if np.count_nonzero(white) > 100 else np.median(lab.reshape(-1, 3), axis=0)
        delta = np.abs(lab.astype(np.float32) - ref).sum(axis=2)
        mask = (delta > float(self.get_parameter('seed_min_solidity').value) * 35).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (float(self.get_parameter('seed_min_area_px').value) <= area <= float(self.get_parameter('seed_max_area_px').value)):
                continue
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area <= 0 or area / hull_area < float(self.get_parameter('seed_min_solidity').value):
                continue
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            items.append(np.array([moments['m10'] / moments['m00'] + x0,
                                   moments['m01'] / moments['m00'] + y0]))
        if len(items) < 3:
            return None
        best = None
        min_spacing = float(self.get_parameter('seed_min_spacing_px').value)
        min_orbit = float(self.get_parameter('seed_min_orbit_radius_px').value)
        max_orbit = float(self.get_parameter('seed_max_orbit_radius_px').value)
        for points in combinations(items, 3):
            circle = self._fit_circle(np.array(points, dtype=np.float32))
            if circle is None:
                continue
            center, radius = circle
            distances = [np.linalg.norm(a - b) for a, b in combinations(points, 2)]
            if min(distances) < min_spacing or not (min_orbit <= radius <= max_orbit):
                continue
            score = abs(radius - self.reference_radius * 0.65) + np.linalg.norm(center - self.reference)
            if best is None or score < best[0]:
                best = (score, center)
        return None if best is None else np.array([best[1][0], best[1][1], self.reference_radius], dtype=np.float32)

    def _coarse_candidates(self, frame: np.ndarray, seed: Optional[np.ndarray]) -> list[np.ndarray]:
        scale = float(self.get_parameter('coarse_scale').value)
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        min_r = max(2, int(float(self.get_parameter('coarse_min_radius_px').value) * scale))
        max_r = max(min_r + 1, int(float(self.get_parameter('coarse_max_radius_px').value) * scale))
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(3, int(float(self.get_parameter('coarse_min_dist_px').value) * scale)),
            param1=float(self.get_parameter('coarse_param1').value),
            param2=float(self.get_parameter('coarse_param2').value),
            minRadius=min_r, maxRadius=max_r)
        output = []
        if circles is not None:
            for x, y, radius in circles[0][:10]:
                candidate = np.array([x / scale, y / scale, radius / scale], dtype=np.float32)
                if seed is not None and np.linalg.norm(candidate[:2] - seed[:2]) > float(self.get_parameter('seed_roi_margin_px').value) * 1.5:
                    continue
                output.append(candidate)
        if seed is not None and not output:
            output.append(seed.copy())
        return output

    def _refine(self, frame: np.ndarray, candidate: np.ndarray) -> Optional[dict]:
        half = float(self.get_parameter('refine_half_width_px').value)
        step = max(1.0, float(self.get_parameter('angular_step_deg').value))
        angles = np.deg2rad(np.arange(0.0, 360.0, step))
        radii = np.arange(max(5.0, candidate[2] - half), candidate[2] + half + 1.0, 1.0)
        map_x = (candidate[0] + np.cos(angles)[:, None] * radii[None, :]).astype(np.float32)
        map_y = (candidate[1] + np.sin(angles)[:, None] * radii[None, :]).astype(np.float32)
        valid = ((map_x >= 3) & (map_x < self.width - 3) &
                 (map_y >= 3) & (map_y < self.height - 3))
        colors = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
        colors = cv2.GaussianBlur(colors, (5, 1), 1.0)
        gradients = np.linalg.norm(colors[:, 2:, :] - colors[:, :-2, :], axis=2)
        valid = valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
        sample_radii = radii[1:-1]
        prior = np.exp(-0.5 * ((sample_radii - candidate[2]) / max(8.0, half * 0.7)) ** 2)
        scores = gradients * prior[None, :]
        scores[~valid] = -np.inf
        index = np.argmax(scores, axis=1)
        rows = np.arange(len(angles))
        strength = gradients[rows, index]
        minimum = float(self.get_parameter('minimum_edge_strength').value)
        eligible = np.isfinite(scores[rows, index]) & (strength >= minimum)
        if np.count_nonzero(eligible) < 20:
            return None
        selected_angles = angles[eligible]
        selected_radii = sample_radii[index[eligible]]
        points = np.column_stack((candidate[0] + selected_radii * np.cos(selected_angles),
                                  candidate[1] + selected_radii * np.sin(selected_angles)))
        strengths = strength[eligible]
        circle = self._robust_fit_circle(points)
        if circle is None:
            return None
        center, radius, inliers = circle
        distances = np.linalg.norm(points[inliers] - center, axis=1)
        rms = float(np.sqrt(np.mean((distances - radius) ** 2)))
        max_rms = float(self.get_parameter('maximum_fit_rms_px').value)
        if rms > max_rms:
            return None
        point_angles = np.mod(np.arctan2(points[inliers, 1] - center[1], points[inliers, 0] - center[0]), 2 * np.pi)
        longest = self._longest_arc_angles(point_angles)
        minimum_arc = float(self.get_parameter('minimum_visible_arc_deg').value)
        if longest < minimum_arc:
            return None
        edge_score = min(1.0, float(np.median(strengths[inliers])) / 20.0)
        confidence = min(1.0, 0.35 * min(1.0, longest / 180.0) +
                         0.50 * max(0.0, 1.0 - rms / (max_rms * 1.5)) + 0.15 * edge_score)
        return {'u': float(center[0]), 'v': float(center[1]), 'radius': float(radius),
                'visible_arc_deg': float(longest), 'fit_rms_px': rms,
                'confidence': confidence, 'detail': 'outer circle refined'}

    def _robust_fit_circle(self, points: np.ndarray):
        if len(points) < 20:
            return None
        keep = np.ones(len(points), dtype=bool)
        circle = self._fit_circle(points)
        if circle is None:
            return None
        for _ in range(5):
            center, radius = circle
            residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
            active = residuals[keep]
            median = float(np.median(active))
            mad = float(np.median(np.abs(active - median))) + 1e-6
            threshold = max(1.5, min(5.0, median + 3.0 * 1.4826 * mad))
            updated = residuals <= threshold
            if np.count_nonzero(updated) < 20:
                break
            keep = updated
            circle = self._fit_circle(points[keep])
            if circle is None:
                return None
        center, radius = circle
        if not (float(self.get_parameter('coarse_min_radius_px').value) <= radius <=
                float(self.get_parameter('coarse_max_radius_px').value)):
            return None
        return center, radius, keep

    @staticmethod
    def _longest_arc_angles(angles: np.ndarray) -> float:
        if len(angles) == 0:
            return 0.0
        bins = np.zeros(72, dtype=bool)
        bins[np.mod((angles / (2 * np.pi) * 72).astype(int), 72)] = True
        bins = bins | (np.roll(bins, 1) & np.roll(bins, -1))
        best = current = 0
        for value in np.tile(bins, 2):
            current = current + 1 if value else 0
            best = max(best, current)
        return float(min(best, 72) * 5.0)

    @staticmethod
    def _longest_arc(bins: Sequence[int], total: int) -> float:
        present = set(bins)
        if not present:
            return 0.0
        best = current = 0
        for index in range(total * 2):
            if index % total in present:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return min(total, best) * 5.0

    @staticmethod
    def _fit_circle(points: np.ndarray):
        if len(points) < 3:
            return None
        x, y = points[:, 0], points[:, 1]
        matrix = np.column_stack((x, y, np.ones(len(points))))
        target = x * x + y * y
        solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
        cx, cy = solution[0] / 2.0, solution[1] / 2.0
        radius_sq = solution[2] + cx * cx + cy * cy
        if radius_sq <= 0 or not np.isfinite(radius_sq):
            return None
        return np.array([cx, cy], dtype=np.float32), float(math.sqrt(radius_sq))

    def destroy_node(self) -> bool:
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = TurntableDetectNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
