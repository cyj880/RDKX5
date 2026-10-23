"""Same-frame 640x480 ring processing and a bounded latest-frame worker."""
from collections import deque
import math
from threading import Condition, Thread
import time

import cv2

from .ring_geometry import circle_groups
from .ring_processing import RingTracker, RingValidation, detect_circles
from .ring_fast_detector import FastRingDetector
from .reference_ring_detector import ReferenceRingDetector


class RingPipeline:
    def __init__(self, validator=None, roi_enabled=True, roi_margin_px=32,
                 roi_full_scan_interval=15, fast_enabled=True, detector_backend='reference'):
        if detector_backend not in ('reference', 'fast_v3'):
            raise ValueError('detector_backend must be reference or fast_v3')
        if roi_margin_px < 0 or roi_full_scan_interval < 1:
            raise ValueError('ROI margin must be >=0; full scan interval must be >=1')
        self.tracker = RingTracker()
        self.validator = validator or RingValidation()
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.roi_enabled = roi_enabled
        self.roi_margin_px = roi_margin_px
        self.roi_full_scan_interval = roi_full_scan_interval
        self.roi_sites = None
        self.frames_since_full = 0
        self.fast_enabled = fast_enabled
        self.fast_detector = FastRingDetector() if fast_enabled else None
        self.detector_backend = detector_backend
        self.reference_detector = (ReferenceRingDetector(
            min_work_width=640, min_work_height=480, min_radius=20, max_radius=60,
            center_merge_dist=20, radius_merge_tol=3, min_rings_per_cluster=2,
            smooth_alpha=0.0) if detector_backend == 'reference' else None)

    def reset(self):
        self.tracker.tracked.clear()
        self.validator.latest_gray = None
        self.validator.smoothed_sites = self.validator.smoothed_yaw = None
        self.roi_sites = None
        self.frames_since_full = 0
        if self.reference_detector is not None:
            self.reference_detector._prev_rings.clear()

    def _bounds(self, margin):
        # Include the largest permitted circle, not just the detected inner ring.
        # Keep original pixels and a motion margin on all four sides.
        radius = max(60.0, max(g['r_inner'] for g in self.roi_sites)) + margin
        return (max(0, math.floor(min(g['x'] for g in self.roi_sites)-radius)),
                max(0, math.floor(min(g['y'] for g in self.roi_sites)-radius)),
                min(640, math.ceil(max(g['x'] for g in self.roi_sites)+radius)+1),
                min(480, math.ceil(max(g['y'] for g in self.roi_sites)+radius)+1))

    def _detect_groups(self, enhanced, edges, bounds):
        x0, y0, x1, y1 = bounds
        # Preserve the original rounding after converting crop coordinates back
        # to the global 640x480 frame. No scaling or threshold changes.
        circles = [(round(x+x0, 1), round(y+y0, 1), round(r, 1))
                   for x, y, r in detect_circles(enhanced[y0:y1, x0:x1])]
        groups, _ = circle_groups(circles, edges)
        for group in groups:
            group.pop('r_outer')
        return groups

    def _search(self, enhanced, edges):
        full = (0, 0, 640, 480)
        attempts = []
        if (self.roi_enabled and self.roi_sites is not None and
                self.frames_since_full < self.roi_full_scan_interval-1):
            attempts = [(self._bounds(self.roi_margin_px), 'roi'),
                        (self._bounds(self.roi_margin_px*2), 'roi_expanded')]
        seen, pixels = set(), 0
        for bounds, mode in attempts + [(full, 'full')]:
            if bounds in seen:
                continue
            seen.add(bounds)
            x0, y0, x1, y1 = bounds
            groups = self._detect_groups(enhanced, edges, bounds)
            pixels += (x1-x0)*(y1-y0)
            # Search attempts must not advance temporal confirmation. Only the
            # chosen final detection is passed to the tracker once per frame.
            chosen, _, _ = self.validator._select(groups)
            if bounds == full or chosen is not None:
                if bounds == full:
                    self.frames_since_full = 0
                    mode = 'full_recovery' if attempts else 'full'
                else:
                    self.frames_since_full += 1
                return groups, mode, bounds, pixels

    def process(self, frame):
        begin=time.monotonic_ns()
        if frame.shape != (480, 640, 3):
            raise ValueError('Ring input must be 640x480 BGR')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_done=time.monotonic_ns()
        if self.reference_detector is not None:
            rings = self.reference_detector.detect(frame)
            groups = [dict(x=float(r.center_x), y=float(r.center_y), r_inner=r.radius,
                           coverage=r.confidence/100.0, n=1)
                      for r in rings if (0 <= r.center_x < 640 and 0 <= r.center_y < 480)]
            groups.sort(key=lambda g: (g['x'],g['y']))
            detection_done=time.monotonic_ns()
            # Like the reference: every current measured ring is eligible now.
            # No distance association or 8-hit gate; keep fixed IDs/Yaw geometry.
            self.validator.latest_gray=gray
            result,rejected=self.validator.evaluate(groups)
            result.update(search_mode='reference', detection_region=[0,0,640,480],
                          hough_pixels=640*480, refine_pixels=0,
                          detector_version='reference-v4',
                          stage_ms=dict(gray=round((gray_done-begin)/1e6,3),
                                        detection=round((detection_done-gray_done)/1e6,3),
                                        validation=round((time.monotonic_ns()-detection_done)/1e6,3)))
            return result,gray,rejected
        full=(0,0,640,480)
        refine_pixels=0
        if self.fast_enabled:
            enhanced=gray
            local=(self.roi_enabled and self.roi_sites is not None and
                   self.frames_since_full < self.roi_full_scan_interval-1)
            if local:
                groups,pixels,refine_pixels=self.fast_detector.detect(gray,self.roi_sites)
                chosen,_,_=self.validator._select(groups)
                if chosen is not None:
                    mode,bounds='edge_track',self._bounds(0)
                    self.frames_since_full+=1
                else:
                    groups,pixels,more_pixels=self.fast_detector.detect(gray)
                    refine_pixels+=more_pixels
                    mode,bounds='fast_recovery',full
                    self.frames_since_full=0
            else:
                groups,pixels,refine_pixels=self.fast_detector.detect(gray)
                mode,bounds='fast_search',full
                self.frames_since_full=0
        else:
            blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)
            enhanced = self.clahe.apply(blurred)
            edges = cv2.Canny(enhanced, 50, 150)
            groups, mode, bounds, pixels = self._search(enhanced, edges)
        detection_done=time.monotonic_ns()
        confirmed, _ = self.tracker._update_tracks(groups)
        self.validator.latest_gray = enhanced
        result, rejected = self.validator.evaluate(confirmed)
        # Build next ROI from current measured centers, never the smoothed pose.
        chosen, _, _ = self.validator._select(confirmed)
        self.roi_sites = [dict(g) for g in chosen] if result['status'] == 'full' else None
        result.update(search_mode=mode, detection_region=list(bounds), hough_pixels=pixels,
                      refine_pixels=refine_pixels, detector_version='fast-v3' if self.fast_enabled else 'roi-v2')
        result['stage_ms']=dict(gray=round((gray_done-begin)/1e6,3),
                                detection=round((detection_done-gray_done)/1e6,3),
                                validation=round((time.monotonic_ns()-detection_done)/1e6,3))
        return result, enhanced, rejected


class LatestFrameWorker:
    """One processing thread; one overwriteable pending frame; session fencing.

    Processing runs outside the condition. Delivery runs inside it, so stop()
    cannot return while an old session is still publishing. The next session
    resets the processor on its own thread before any new frame is processed.
    """
    def __init__(self, process, deliver, reset, on_error):
        self.process, self.deliver = process, deliver
        self.reset, self.on_error = reset, on_error
        self.condition = Condition()
        self.pending = None
        self.generation = 0
        self.active = self.closed = False
        self.sequence = 0
        self.received = self.processed = self.overwritten = self.discarded = 0
        self.durations = deque(maxlen=512)
        self.thread = Thread(target=self._run, name='ring-latest-frame', daemon=True)
        self.thread.start()

    def start(self):
        with self.condition:
            self.generation += 1
            self.active = True
            self.pending = None
            self.condition.notify()

    def stop(self):
        with self.condition:
            self.generation += 1
            self.active = False
            self.pending = None
            self.condition.notify()

    def submit(self, frame, ingress_ns=None):
        with self.condition:
            if not self.active or self.closed:
                return
            self.received += 1
            self.sequence += 1
            if self.pending is not None:
                self.overwritten += 1
            self.pending = (self.generation, self.sequence,
                            time.monotonic_ns() if ingress_ns is None else ingress_ns, frame)
            self.condition.notify()

    def snapshot(self, reset_window=False):
        with self.condition:
            durations = sorted(self.durations)
            result = dict(received=self.received, processed=self.processed,
                          overwritten=self.overwritten, discarded=self.discarded,
                          process_p95_ms=durations[int((len(durations)-1)*0.95)] if durations else 0.0)
            if reset_window:
                self.durations.clear()
            return result

    def close(self):
        with self.condition:
            self.closed = True
            self.active = False
            self.pending = None
            self.condition.notify()
        # Finish the sole in-flight OpenCV call before ROS resources are destroyed.
        self.thread.join()

    def _run(self):
        current_generation = None
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None or
                                        current_generation != self.generation)
                if self.closed:
                    return
                generation = self.generation
                item, self.pending = self.pending, None
            if current_generation != generation:
                self.reset()
                current_generation = generation
            if item is None:
                continue
            session, sequence, ingress_ns, frame = item
            begin = time.monotonic_ns()
            try:
                output = self.process(frame)
                duration_ms = (time.monotonic_ns()-begin)/1e6
                with self.condition:
                    self.processed += 1
                    self.durations.append(duration_ms)
                    if self.closed or not self.active or session != self.generation:
                        self.discarded += 1
                        continue
                    self.deliver(frame, output, sequence, ingress_ns, duration_ms)
            except Exception as error:
                self.reset()
                with self.condition:
                    if not self.closed and self.active and session == self.generation:
                        self.on_error(error)
