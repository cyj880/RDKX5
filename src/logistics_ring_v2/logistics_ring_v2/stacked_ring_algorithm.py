"""Stateful stacked-station tracker, independent of ROS transport."""

from collections import deque
import math

import cv2
import numpy as np

from .ring_geometry import circle_groups, select_three


class StackedTracker:
    def __init__(self, *, expected_spacing=152.1, outer_diameter=95., line_offset=75.,
                 stable_frames=5, center_jitter=5., angle_jitter=1.,
                 min_radius=20, max_radius=60, line_band=18, min_line_span=120,
                 line_disagreement=2., confirm_frames=8):
        self.expected_spacing = expected_spacing
        self.outer_diameter = outer_diameter
        self.line_offset = line_offset
        self.stable_frames = stable_frames
        self.center_jitter = center_jitter
        self.angle_jitter = angle_jitter
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.line_band = line_band
        self.min_line_span = min_line_span
        self.line_disagreement = line_disagreement
        self.confirm_frames = confirm_frames
        self.reset()
        self.clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    def reset(self):
        self.middle = None
        self.sides = None
        self.initial_angle = None
        self.candidate = None
        self.confirmed = 0
        self.history = deque(maxlen=self.stable_frames)

    @staticmethod
    def _empty(state, reason):
        return dict(detected=False, stable=False, state=state, middle_center_px=None,
                    middle_radius_px=None, top_line_angle_deg=None, bottom_line_angle_deg=None,
                    station_yaw_deg=None, side_centers_inferred=None, quality=0., failure_reason=reason)

    def _initialize(self, gray, edges):
        enhanced = self.clahe.apply(cv2.GaussianBlur(gray, (5, 5), 0))
        found = cv2.HoughCircles(enhanced, cv2.HOUGH_GRADIENT_ALT, dp=1., minDist=60.,
                                 param1=100., param2=.4, minRadius=self.min_radius,
                                 maxRadius=self.max_radius)
        circles = [tuple(map(float, item)) for item in found[0]] if found is not None else []
        groups, _ = circle_groups(circles, cv2.Canny(enhanced, 50, 150), radial_tolerance=2)
        groups = [group for group in groups if group['n'] >= 2 and
                  group['r_outer'] - group['r_inner'] >= 8]
        triple, relation, _ = select_three(groups, self.expected_spacing)
        if triple is None:
            self.candidate, self.confirmed = None, 0
            return False
        center = triple[1]
        if self.candidate is None or math.hypot(center['x'] - self.candidate[0],
                                                center['y'] - self.candidate[1]) > 15:
            self.confirmed = 0
        self.candidate = center['x'], center['y']
        self.confirmed += 1
        if self.confirmed < self.confirm_frames:
            return False
        refined = self._refine(edges, (center['x'], center['y'], center['r_outer']))
        if refined is None:
            return False
        self.middle = refined
        self.sides = (relation[0], relation[1])
        self.initial_angle = relation[3]
        self.history.clear()
        return True

    def _refine(self, edges, seed):
        x, y, radius = seed
        yy, xx = np.nonzero(edges)
        if not len(xx):
            return None
        distance = np.hypot(xx - x, yy - y)
        selected = abs(distance - radius) <= 4
        if selected.sum() < 70:
            return None
        coords = np.column_stack((xx[selected], yy[selected])).astype(np.float64)
        for limit in (3., 2.):
            design = np.column_stack((2 * coords[:, 0], 2 * coords[:, 1], np.ones(len(coords))))
            answer, *_ = np.linalg.lstsq(design, np.sum(coords ** 2, axis=1), rcond=None)
            cx, cy = answer[:2]
            r2 = answer[2] + cx * cx + cy * cy
            if r2 <= 0:
                return None
            fitted = math.sqrt(r2)
            if math.hypot(cx - x, cy - y) > 8 or abs(fitted - radius) > 8:
                return None
            keep = abs(np.hypot(coords[:, 0] - cx, coords[:, 1] - cy) - fitted) < limit
            if keep.sum() < 60:
                return None
            coords = coords[keep]
        arc = np.unique((np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx) * 36 / np.pi).astype(int))
        if len(arc) < 30 or not self.min_radius <= fitted <= self.max_radius:
            return None
        return float(cx), float(cy), float(fitted)

    def _line(self, edges, center, target_y):
        x, y, _ = center
        slope = math.tan(math.radians(self.initial_angle))
        rows = np.arange(edges.shape[0])[:, None]
        cols = np.arange(edges.shape[1])[None, :]
        mask = np.abs(rows - (target_y + slope * (cols - x))) <= self.line_band
        filtered = np.where(mask, edges, 0).astype(np.uint8)
        found = cv2.HoughLinesP(filtered, 1, np.pi / 180, 40,
                                minLineLength=self.min_line_span, maxLineGap=25)
        best = None
        for item in found[:, 0] if found is not None else []:
            x1, y1, x2, y2 = map(int, item)
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle - self.initial_angle) > 15:
                continue
            length = math.hypot(x2 - x1, y2 - y1)
            expected = target_y + slope * ((x1 + x2) / 2 - x)
            score = length - 3 * abs((y1 + y2) / 2 - expected)
            if best is None or score > best[0]:
                best = (score, angle, ((x1, y1), (x2, y2)))
        return best

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        state = 'tracking'
        if self.middle is None:
            if not self._initialize(gray, edges):
                return self._empty('initializing', '等待完整三圆环连续确认'), frame.copy()
            state = 'initialized'
        else:
            updated = self._refine(edges, self.middle)
            if updated is None or math.hypot(updated[0] - self.middle[0], updated[1] - self.middle[1]) > 15:
                self.reset()
                return self._empty('search', '中环失锁，等待完整三环重新初始化'), frame.copy()
            self.middle = updated
        x, y, r = self.middle
        offset = 2 * r * self.line_offset / self.outer_diameter
        top, bottom = self._line(edges, self.middle, y - offset), self._line(edges, self.middle, y + offset)
        annotated = frame.copy()
        cv2.circle(annotated, (round(x), round(y)), round(r), (0, 255, 255), 2)
        for line in (top, bottom):
            if line:
                cv2.line(annotated, *line[2], (255, 0, 255), 2)
        if not top and not bottom:
            self.history.clear()
            return self._empty(state, '上下基准线均不可见'), annotated
        if top and bottom and abs(top[1] - bottom[1]) > self.line_disagreement:
            self.history.clear()
            return self._empty(state, '上下基准线方向冲突'), annotated
        angle = float(np.mean([line[1] for line in (top, bottom) if line]))
        self.history.append((x, y, angle))
        samples = np.asarray(self.history)
        stable = (len(self.history) == self.stable_frames and
                  float(np.ptp(samples[:, 0])) <= self.center_jitter and
                  float(np.ptp(samples[:, 1])) <= self.center_jitter and
                  float(np.ptp(samples[:, 2])) <= self.angle_jitter)
        theta = math.radians(angle)
        sides = [{'x': round(x - self.sides[0] * math.cos(theta), 2),
                  'y': round(y - self.sides[0] * math.sin(theta), 2), 'source': 'inferred'},
                 {'x': round(x + self.sides[1] * math.cos(theta), 2),
                  'y': round(y + self.sides[1] * math.sin(theta), 2), 'source': 'inferred'}]
        for side in sides:
            cv2.drawMarker(annotated, (round(side['x']), round(side['y'])),
                           (0, 165, 255), cv2.MARKER_CROSS, 16, 1)
            cv2.putText(annotated, 'INFERRED', (round(side['x']) - 30, round(side['y']) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 165, 255), 1)
        result = dict(detected=True, stable=stable, state=state,
                      middle_center_px={'x': round(x, 2), 'y': round(y, 2)},
                      middle_radius_px=round(r, 2),
                      top_line_angle_deg=round(top[1], 2) if top else None,
                      bottom_line_angle_deg=round(bottom[1], 2) if bottom else None,
                      station_yaw_deg=round(angle, 2), side_centers_inferred=sides,
                      quality=round(min(1., len(self.history)/self.stable_frames) *
                                    (1. if top and bottom else .7), 2), failure_reason='')
        return result, annotated
