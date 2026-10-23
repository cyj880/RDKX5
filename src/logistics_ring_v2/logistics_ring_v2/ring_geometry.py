"""Shared geometric checks used by ordinary and stacked ring detection."""

import math
from itertools import combinations

import numpy as np


def circle_groups(circles, edges, cluster_dist=20., sectors=12, samples=30,
                  min_coverage=.7, radial_tolerance=0):
    clusters = []
    for circle in circles:
        for cluster in clusters:
            center = np.mean(np.asarray(cluster)[:, :2], axis=0)
            if np.hypot(circle[0] - center[0], circle[1] - center[1]) <= cluster_dist:
                cluster.append(circle)
                break
        else:
            clusters.append([circle])
    angles = np.arange(sectors * samples) * (2 * np.pi / (sectors * samples))
    accepted, rejected = [], []
    for cluster in clusters:
        scored = []
        for x, y, r in cluster:
            hits = np.zeros(len(angles), dtype=bool)
            for shift in range(-radial_tolerance, radial_tolerance + 1):
                px = np.rint(x + (r + shift) * np.cos(angles)).astype(int)
                py = np.rint(y + (r + shift) * np.sin(angles)).astype(int)
                valid = (px >= 0) & (px < edges.shape[1]) & (py >= 0) & (py < edges.shape[0])
                hits[valid] |= edges[py[valid], px[valid]] != 0
            coverage = float(hits.reshape(sectors, samples).any(axis=1).mean())
            (scored if coverage >= min_coverage else rejected).append((x, y, r, coverage))
        if scored:
            weights = [item[3] for item in scored]
            accepted.append({'x': round(float(np.average([item[0] for item in scored], weights=weights)), 1),
                             'y': round(float(np.average([item[1] for item in scored], weights=weights)), 1),
                             'r_inner': round(min(item[2] for item in scored), 1),
                             'r_outer': round(max(item[2] for item in scored), 1),
                             'coverage': round(float(np.mean(weights)), 3), 'n': len(scored)})
    return accepted, rejected


def select_three(groups, expected_spacing=152.1, tolerance=.2, ratio_min=.85,
                 max_collinearity=.08):
    best = None
    for triple in combinations(groups, 3):
        ordered = sorted(triple, key=lambda g: g['x'])
        a, b, c = ordered
        first = math.hypot(b['x'] - a['x'], b['y'] - a['y'])
        second = math.hypot(c['x'] - b['x'], c['y'] - b['y'])
        span = math.hypot(c['x'] - a['x'], c['y'] - a['y'])
        if min(first, second) <= 0 or span <= 0:
            continue
        col = abs((b['x'] - a['x']) * (c['y'] - a['y']) -
                  (b['y'] - a['y']) * (c['x'] - a['x'])) / (span * span)
        avg = (first + second) / 2
        if (min(first, second) / max(first, second) < ratio_min or col > max_collinearity
                or abs(avg - expected_spacing) > expected_spacing * tolerance):
            continue
        score = abs(math.log(first / second)) + col + abs(avg - expected_spacing) / expected_spacing
        if best is None or score < best[0]:
            best = score, ordered, (first, second, col, math.degrees(math.atan2(c['y']-a['y'], c['x']-a['x'])))
    if best is None:
        return None, None, groups
    return best[1], best[2], [g for g in groups if g not in best[1]]
