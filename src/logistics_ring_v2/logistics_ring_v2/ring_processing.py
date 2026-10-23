"""Shared ROS-independent ring tracking, validation and drawing algorithms."""
import math
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from .ring_geometry import select_three
from .ring_serial_protocol import ring_yaw


def detect_circles(gray, dp=1.0, min_dist=60.0, param1=100.0, param2=0.4,
                   min_radius=20, max_radius=60):
    """Common detector for the runtime pipeline and original Hough debug node."""
    found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, dp=dp, minDist=min_dist,
                             param1=param1, param2=param2,
                             minRadius=min_radius, maxRadius=max_radius)
    return [] if found is None else [tuple(float(v) for v in c) for c in found.reshape(-1, 3)]


class RingTracker:
    def __init__(self, smooth_alpha=1.0):
        self.tracked = []
        self.track_match_dist = 25.0
        self.track_confirm_hits = 8
        self.track_max_misses = 3
        self.track_remove_misses = 8
        self.track_smooth_alpha = smooth_alpha

    def _update_tracks(self, groups: List[dict]) -> Tuple[List[dict], List[dict]]:
        """组级持续性跟踪: 新组须连续命中 confirm_hits 帧才确认发布。

        当帧组与跟踪组按圆心距最近邻关联; 未关联的当帧组新建 candidate;
        未被关联的跟踪组计丢失, 连丢超容忍帧数命中史清零, 超删除帧数移除。
        返回 (confirmed 组, candidate 组), 只有 confirmed 进入 groups 话题。
        """
        matched = set()
        for group in groups:
            gx, gy = group['x'], group['y']
            best, best_dist = None, self.track_match_dist
            for track in self.tracked:
                dist = math.hypot(gx - track['x'], gy - track['y'])
                if dist <= best_dist:
                    best, best_dist = track, dist
            if best is None:
                self.tracked.append({
                    'x': gx, 'y': gy, 'r_inner': group['r_inner'],
                    'coverage': group['coverage'], 'n': group['n'],
                    'hits': 1, 'misses': 0,
                })
                matched.add(id(self.tracked[-1]))
                continue
            best['hits'] += 1
            best['misses'] = 0
            alpha = self.track_smooth_alpha
            best['x'] += alpha * (gx - best['x'])
            best['y'] += alpha * (gy - best['y'])
            best['r_inner'] += alpha * (group['r_inner'] - best['r_inner'])
            best['coverage'] = group['coverage']
            best['n'] = group['n']
            matched.add(id(best))

        confirmed, candidates, survivors = [], [], []
        for track in self.tracked:
            if id(track) not in matched:
                track['misses'] += 1
                if track['misses'] > self.track_max_misses:
                    track['hits'] = 0  # 连丢超容忍: 命中史作废, 防间歇假组攒数
            if track['misses'] > self.track_remove_misses:
                continue  # 删除: 防旧位置残留
            if track['hits'] >= self.track_confirm_hits and track['misses'] == 0:
                track['confirmed'] = True
                confirmed.append(track)
            else:
                candidates.append(track)
            survivors.append(track)
        self.tracked = survivors

        confirmed.sort(key=lambda t: t['x'])
        out = [{'x': round(t['x'], 1), 'y': round(t['y'], 1),
                'r_inner': round(t['r_inner'], 1),
                'coverage': round(t['coverage'], 3), 'n': t['n']}
               for t in confirmed]
        cand = [{'x': t['x'], 'y': t['y'], 'r_inner': t['r_inner'],
                 'hits': t['hits']} for t in candidates]
        return out, cand


class RingValidation:
    def __init__(self, smooth_alpha=0.8):
        self.expected_spacing_px = 150.0 * 1.014
        self.spacing_tolerance = 0.2
        self.spacing_ratio_min = 0.85
        self.max_collinearity = 0.08
        self.smooth_alpha = smooth_alpha
        self.snap_dist_px = 15.0
        self.smoothed_sites = self.smoothed_yaw = None
        self.digit_conf_min = 0.6
        self.digit_available = False
        self.latest_gray = None

    def load_digit_model(self, model_path, logger):
        model = Path(model_path).expanduser()
        try:
            if not model.is_file():
                raise ValueError('数字模型不存在: %s' % model)
            # Load lazily: absent model/HOG support must not disable full triples.
            from . import digit_features
            self.digit_features = digit_features
            self.svm = cv2.ml.SVM_load(str(model))
            if self.svm.empty():
                raise ValueError('数字模型为空')
            self.hog = digit_features.create_hog()
            self.digit_available = True
        except (ImportError, AttributeError, ValueError, cv2.error) as error:
            logger.warning('缺环编号识别不可用，未知编号停车等待: %s' % error)


    def _select(self, groups: List[dict]):
        """返回 (达标三组, 关系量, 被淘汰组)。不足三组或无达标组合时返回空。"""
        return select_three(groups, self.expected_spacing_px, self.spacing_tolerance,
                            self.spacing_ratio_min, self.max_collinearity)


    def _smooth(self, chosen: List[dict], relation) -> Tuple[List[dict], float]:
        """EMA 平滑 + 跳变重置。返回 (平滑后 sites, 平滑后站位角)。

        组按 x 排序, 平滑状态按左中右位次维护; 站位角已归一到 (-90, 90),
        帧间无 180° 跳变, 直接 EMA。
        """
        chosen = sorted(chosen, key=lambda g: g['x'])
        new_yaw = relation[3]
        if self.smoothed_sites is None or len(self.smoothed_sites) != len(chosen):
            self.smoothed_sites = [(g['x'], g['y'], g['r_inner']) for g in chosen]
            self.smoothed_yaw = new_yaw
        else:
            alpha = self.smooth_alpha
            updated = []
            for index, group in enumerate(chosen):
                sx, sy, sr = self.smoothed_sites[index]
                if math.hypot(group['x'] - sx, group['y'] - sy) > self.snap_dist_px:
                    # 跳变重置: 车被挪动或严重误检, 直接采用新值不拖尾
                    updated.append((group['x'], group['y'], group['r_inner']))
                else:
                    updated.append((
                        sx + alpha * (group['x'] - sx),
                        sy + alpha * (group['y'] - sy),
                        sr + alpha * (group['r_inner'] - sr),
                    ))
            self.smoothed_sites = updated
            self.smoothed_yaw += alpha * (new_yaw - self.smoothed_yaw)

        sites = []
        for index, group in enumerate(chosen):
            sx, sy, sr = self.smoothed_sites[index]
            item = dict(group)
            item['x'] = round(sx, 2)
            item['y'] = round(sy, 2)
            item['r_inner'] = round(sr, 2)
            sites.append(item)
        # 间距从平滑后坐标现算, 保证输出自洽
        for index in range(len(sites) - 1):
            sites[index]['spacing_to_next_px'] = round(math.hypot(
                sites[index + 1]['x'] - sites[index]['x'],
                sites[index + 1]['y'] - sites[index]['y']), 1)
        return sites, round(self.smoothed_yaw, 2)


    def _recognize_identity(self, group: dict) -> Tuple[int, float]:
        if not self.digit_available or self.latest_gray is None:
            return 0, 0.0
        try:
            gray = self.latest_gray
            cx, cy = int(round(group['x'])), int(round(group['y']))
            radius = int(round(group['r_inner'] * 0.85))
            if radius < 8:
                return 0, 0.0
            height, width = gray.shape[:2]
            x0, y0 = max(0, cx - radius), max(0, cy - radius)
            x1, y1 = min(width, cx + radius), min(height, cy + radius)
            if x1 - x0 < 16 or y1 - y0 < 16:
                return 0, 0.0
            features = self.digit_features
            binary = features.binarize_digit(gray[y0:y1, x0:x1])
            disk = np.zeros(binary.shape, dtype=np.uint8)
            cv2.circle(disk, (cx - x0, cy - y0), radius, 255, -1)
            mask = features.normalize_mask(cv2.bitwise_and(binary, disk))
            if mask is None:
                return 0, 0.0
            scores = features.svm_class_scores(self.svm, features.hog_features(self.hog, mask))
            index = int(np.argmax(scores))
            confidence = float(scores[index])
            return ((index + 1, confidence) if math.isfinite(confidence) and
                    confidence >= self.digit_conf_min else (0, confidence))
        except (AttributeError, ValueError, cv2.error):
            return 0, 0.0


    def _partial_identities(self, sites: List[dict]) -> None:
        for site in sites:
            identity, confidence = self._recognize_identity(site)
            site['id'] = identity
            site['digit_confidence'] = round(confidence, 3) if math.isfinite(confidence) else 0.0
        identities = [s['id'] for s in sites if s['id']]
        # Conflicting/reversed digits are not sufficient evidence to move.
        if len(identities) != len(set(identities)) or identities != sorted(identities):
            for site in sites:
                site['id'] = 0


    def evaluate(self, groups):
        chosen, relation, rejected = self._select(groups)
        if chosen is not None:
            sites, yaw = self._smooth(chosen, relation)
            for identity, site in enumerate(sites, 1):
                site['id'] = identity
            yaw = ring_yaw(sites)
            status = 'full'
        elif 1 <= len(groups) <= 2:
            # Posterior-confirmed centers remain useful without a digit model.
            # Partial indices are image order, not physical station identities.
            self.smoothed_sites = None
            self.smoothed_yaw = None
            sites = [dict(g) for g in groups]
            self._partial_identities(sites)
            yaw = ring_yaw(sites)
            status, rejected = 'partial', []
        else:
            # N=0 / 无可信三环组合: 如实报告 none, 不发送旧坐标
            status, sites, yaw = 'none', [], None

        return dict(status=status, count=len(sites), sites=sites, missing=[],
                    station_yaw_deg=yaw, yaw_valid=yaw is not None), rejected


def annotate_ring(gray, status, sites, yaw, rejected):
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for g in rejected:
        cv2.circle(annotated, (int(g['x']), int(g['y'])),
                   int(g['r_inner']), (0, 0, 160), 1)
    for index, g in enumerate(sites, start=1):
        point = (int(round(g['x'])), int(round(g['y'])))
        cv2.circle(annotated, point, int(round(g['r_inner'])), (0, 255, 0), 2)
        cv2.drawMarker(annotated, point, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(annotated, 'ID%d (%.0f,%.0f)' % (g.get('id', 0), g['x'], g['y']),
                    (point[0] + 8, point[1] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, 'RING %s yaw=%s' % (status.upper(), yaw if yaw is not None else '--'),
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if status == 'full' else (0, 165, 255), 2, cv2.LINE_AA)
    return annotated
