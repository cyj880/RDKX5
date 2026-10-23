"""Half-size acquisition, original-pixel edge fitting and Hough-free local tracking.

Candidates never become measurements without fresh original-image edge support.
Only this ring process uses the thread limit configured by its ROS adapter.
"""
import math

import cv2
import numpy as np

from .ring_geometry import circle_groups
from .ring_processing import detect_circles


class FastRingDetector:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    @staticmethod
    def _fit(points):
        # Center the system before solving to keep original-pixel fits stable.
        mean = points.mean(axis=0)
        p = points-mean
        solution, _, _, _ = np.linalg.lstsq(
            np.column_stack((2*p[:, 0], 2*p[:, 1], np.ones(len(p)))),
            np.sum(p*p, axis=1), rcond=None)
        radius_squared = solution[2]+solution[0]**2+solution[1]**2
        if radius_squared <= 0:
            return None
        center = mean+solution[:2]
        return float(center[0]), float(center[1]), math.sqrt(float(radius_squared))

    @staticmethod
    def _support(points, circle):
        x,y,r = circle
        radial = np.hypot(points[:, 0]-x, points[:, 1]-y)
        inliers = points[np.abs(radial-r) <= 1.5]
        if len(inliers) < 24:
            return None
        angles = np.arctan2(inliers[:, 1]-y, inliers[:, 0]-x)
        sectors = np.floor((angles+np.pi)*12/(2*np.pi)).astype(int) % 12
        if len(np.unique(sectors)) < 9: # Same >=0.7 angular coverage gate.
            return None
        return inliers

    def _refine(self, gray, seed, edge_map, cache):
        cx,cy,radius = seed
        margin = 26
        half = radius+margin
        bounds = (max(0, int(cx-half)), max(0, int(cy-half)),
                  min(640, int(cx+half)+1), min(480, int(cy+half)+1))
        x0,y0,x1,y1 = bounds
        if bounds not in cache:
            patch = cv2.GaussianBlur(gray[y0:y1, x0:x1], (3,3), 1.0)
            edges = cv2.Canny(patch, 50, 150)
            cache[bounds] = edges
            edge_map[y0:y1,x0:x1] |= edges
        edges = cache[bounds]
        rows,cols = np.nonzero(edges)
        if len(cols) < 24:
            return None
        points = np.column_stack((cols+x0,rows+y0)).astype(float)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        candidates = []
        for contour in contours:
            if len(contour) < 24:
                continue
            _,_,width,height = cv2.boundingRect(contour)
            if min(width,height) < 32 or min(width,height)/max(width,height) < .75:
                continue
            # OpenCV fits whole contour arcs in native code; reserve the more
            # costly robust least-squares pass for the final supported circle.
            (x,y),(a,b),_ = cv2.fitEllipse(contour)
            if min(a,b)/max(a,b) >= .85:
                candidates.append((x+x0,y+y0,(a+b)/4))
        # Broken/dashed edges may not form one contour. Fit the radial band of
        # current edges around the coarse seed, then trim twice; no old center
        # is emitted if this fit fails the coverage and displacement gates.
        distance = np.hypot(points[:,0]-cx,points[:,1]-cy)
        band = points[np.abs(distance-radius) <= 5]
        if not candidates and len(band) >= 24:
            fit = self._fit(band)
            if fit is not None:
                for _ in range(2):
                    residual = np.abs(np.hypot(band[:,0]-fit[0],band[:,1]-fit[1])-fit[2])
                    band = band[residual <= 2]
                    if len(band) < 24:
                        fit = None
                        break
                    fit = self._fit(band)
                    if fit is None:
                        break
                if fit is not None:
                    candidates.append(fit)
        candidates.sort(key=lambda fit: math.hypot(fit[0]-cx,fit[1]-cy)+abs(fit[2]-radius)*.25)
        for fit in candidates:
            x,y,r = fit
            if (not 20 <= r <= 60 or math.hypot(x-cx,y-cy) > 25 or
                    abs(r-radius) > 8 or not (0 <= x < 640 and 0 <= y < 480)):
                continue
            supported = self._support(points, fit)
            if supported is None:
                continue
            fit = self._fit(supported)
            if fit is None or self._support(points, fit) is None:
                continue
            # Recheck after least-squares refinement as well.
            x,y,r=fit
            if (not 20 <= r <= 60 or math.hypot(x-cx,y-cy)>25 or abs(r-radius)>8 or
                    not (0 <= x < 640 and 0 <= y < 480)):
                continue
            return fit
        return None

    def detect(self, gray, sites=None):
        # The Hough search is always 320x240, with radii/distances scaled too.
        # Tracking instead refits three original-pixel local edge patches.
        coarse_pixels=0
        if sites is None:
            small=cv2.resize(gray,(320,240),interpolation=cv2.INTER_AREA)
            small=self.clahe.apply(cv2.GaussianBlur(small,(3,3),1.0))
            seeds=[(x*2,y*2,r*2) for x,y,r in detect_circles(
                small,min_dist=30,min_radius=10,max_radius=30)]
            coarse_pixels=320*240
        else:
            seeds=[(s['x'],s['y'],s['r_inner']) for s in sites]
        edges=np.zeros(gray.shape,np.uint8)
        cache={}
        circles=[]
        for seed in seeds:
            fitted=self._refine(gray,seed,edges,cache)
            if fitted is not None:
                circles.append(fitted)
        # One-pixel radial tolerance accounts for rasterization of edge fits;
        # center clustering and 12-sector >=0.7 coverage still gate every ring.
        groups,_=circle_groups(circles,edges,radial_tolerance=1)
        for g in groups:
            g.pop('r_outer')
        return groups, coarse_pixels, sum(e.size for e in cache.values())
