"""Latency, frame ownership and protocol tests using the actual worker and pipeline."""
from collections import deque
import json
from threading import Event, Lock
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from std_msgs.msg import String

from logistics_ring_v2.ring_pipeline import LatestFrameWorker, RingPipeline
from logistics_ring_v2.ring_pipeline_node import RingPipelineNode
from logistics_ring_v2.ring_processing import RingValidation, RingTracker
from logistics_ring_v2.ring_serial_bridge_node import RingSerialBridgeNode
from logistics_ring_v2.ring_serial_protocol import FrameParser, encode_identified_ring_result


def scene(shift=0, slope=0, count=3):
    frame = np.full((480, 640, 3), 220, np.uint8)
    for index in range(count):
        point = (170 + index*152 + shift, 200 + int(index*152*slope))
        cv2.circle(frame, point, 30, (20, 20, 20), -1, cv2.LINE_AA)
    return frame


class WorkerTest(unittest.TestCase):
    def test_latest_only_and_single_processing_thread(self):
        entered, release, done = Event(), Event(), Event()
        processed, delivered, resets, errors = [], [], [], []
        lock = Lock()

        def process(frame):
            self.assertTrue(lock.acquire(blocking=False))
            processed.append(frame)
            if frame == 1:
                entered.set()
                self.assertTrue(release.wait(3))
            lock.release()
            return frame

        def deliver(frame, output, sequence, ingress, duration):
            delivered.append((frame, sequence, ingress))
            if frame == 30:
                done.set()

        worker = LatestFrameWorker(process, deliver, lambda: resets.append(1), errors.append)
        self.addCleanup(worker.close)
        worker.start()
        worker.submit(1, 100)
        self.assertTrue(entered.wait(3))
        for value in range(2, 31):
            worker.submit(value, value*100)
        release.set()
        self.assertTrue(done.wait(3))
        self.assertEqual(processed, [1, 30])
        self.assertEqual([d[0] for d in delivered], [1, 30])
        self.assertEqual(delivered[-1], (30, 30, 3000))
        self.assertEqual(worker.snapshot()['overwritten'], 28)
        self.assertEqual(errors, [])

    def test_stop_restart_discards_inflight_and_resets_before_new_frame(self):
        entered, release, done = Event(), Event(), Event()
        state, delivered, errors = [], [], []

        def process(value):
            if value == 'old':
                entered.set()
                self.assertTrue(release.wait(3))
            state.append(value)
            return list(state)

        def deliver(frame, output, *args):
            delivered.append(output)
            done.set()

        worker = LatestFrameWorker(process, deliver, state.clear, errors.append)
        self.addCleanup(worker.close)
        worker.start()
        worker.submit('old')
        self.assertTrue(entered.wait(3))
        worker.submit('pending-old')
        worker.stop()
        worker.submit('ignored')
        worker.start()
        worker.submit('new')
        release.set()
        self.assertTrue(done.wait(3))
        self.assertEqual(delivered, [['new']])
        self.assertEqual(worker.snapshot()['discarded'], 1)
        self.assertEqual(errors, [])

    def test_processing_error_does_not_kill_worker(self):
        done = Event()
        errors, delivered = [], []

        def process(frame):
            if frame == 1:
                raise ValueError('bad frame')
            return frame

        worker = LatestFrameWorker(process, lambda *args: (delivered.append(args[0]), done.set()),
                                   lambda: None, lambda err: (errors.append(err), done.set()))
        self.addCleanup(worker.close)
        worker.start()
        worker.submit(1)
        self.assertTrue(done.wait(3))
        done.clear()
        worker.submit(2)
        self.assertTrue(done.wait(3))
        self.assertEqual(delivered, [2])
        self.assertEqual(len(errors), 1)


class FrameOwnershipTest(unittest.TestCase):
    def node(self):
        node = RingPipelineNode.__new__(RingPipelineNode)
        node.publisher = Mock()
        node.annotated_publisher = Mock()
        node.annotated_publisher.get_subscription_count.return_value = 1
        node.bridge = Mock()
        node.bridge.cv2_to_imgmsg.side_effect = lambda frame, **kwargs: SimpleNamespace(image=frame)
        node.display = node.display_debug = False
        return node

    def test_annotation_owns_source_header_pixels_and_metadata(self):
        node = self.node()
        source = SimpleNamespace(header=object())
        gray = np.full((480, 640), 57, np.uint8)
        result = dict(status='full', count=3, sites=[dict(id=i+1, x=100+i*152, y=200, r_inner=30)
                                                  for i in range(3)], station_yaw_deg=0)
        node._deliver(source, (result, gray, []), 12, 345, 4.5)
        image = node.annotated_publisher.publish.call_args.args[0]
        self.assertIs(image.header, source.header)
        self.assertEqual(tuple(image.image[450, 620]), (57, 57, 57))
        self.assertTrue(np.any(image.image[200, 100] != 57))
        data = json.loads(node.publisher.publish.call_args.args[0].data)
        self.assertEqual((data['frame_sequence'], data['ingress_monotonic_ns'], data['processing_ms']),
                         (12, 345, 4.5))

    def test_no_subscriber_skips_annotation_and_headless_debug_stays_off(self):
        node = self.node()
        node.annotated_publisher.get_subscription_count.return_value = 0
        node._deliver(SimpleNamespace(header=None),
                      (dict(status='none', count=0, sites=[], station_yaw_deg=None), None, []), 1, 10, 2)
        node.bridge.cv2_to_imgmsg.assert_not_called()
        node.annotated_publisher.publish.assert_not_called()


class FreshnessTest(unittest.TestCase):
    def bridge(self):
        node = RingSerialBridgeNode.__new__(RingSerialBridgeNode)
        node.active_mode = 'ring'
        node.latest_ring_time = None
        node.latest_ring_frame = b''
        node.ring_pending = False
        node.last_ring_source_ns = node.ring_mode_since_ns = 0
        node.ring_tx_ages = deque(maxlen=512)
        node.ring_stats_time = float('inf')
        node._write = Mock()
        return node

    def message(self, stamp=10_000_000_000, sequence=1):
        return String(data=json.dumps(dict(count=1, sites=[dict(id=1, x=123, y=234)],
                                           frame_sequence=sequence, ingress_monotonic_ns=stamp)))

    def empty(self, node):
        self.assertEqual(FrameParser().feed(node._write.call_args.args[0])[0], (0x13, bytes(4)))

    def test_source_age_includes_processing_and_timer_wait(self):
        node = self.bridge()
        with patch('time.monotonic', return_value=10.25):
            node._ring_result_callback(self.message())
            node._publish_latest()
        self.assertEqual(node._write.call_args.args[0], encode_identified_ring_result([dict(id=1,x=123,y=234)]))
        with patch('time.monotonic', return_value=10.301):
            node._publish_latest()
        self.empty(node)

    def test_expired_missing_future_and_wrong_session_are_empty(self):
        for stamp, seq in [(9_699_999_999, 1), (10_000_000_001, 1), (True, 1),
                           (0, 1), (10_000_000_000, 0), ('1000', 1)]:
            node = self.bridge()
            with patch('time.monotonic', return_value=10.):
                node._ring_result_callback(self.message(stamp, seq))
                node._publish_latest()
            self.empty(node)
        node = self.bridge()
        node.ring_mode_since_ns = 9_950_000_000
        with patch('time.monotonic', return_value=10.):
            node._ring_result_callback(self.message(9_900_000_000))
            node._publish_latest()
        self.empty(node)
        data = json.loads(self.message().data)
        del data['ingress_monotonic_ns']
        node._ring_result_callback(String(data=json.dumps(data)))
        node._publish_latest()
        self.empty(node)

    def test_duplicate_and_out_of_order_do_not_rearm_pending(self):
        node = self.bridge()
        with patch('time.monotonic', return_value=10.1):
            node._ring_result_callback(self.message())
            node._publish_latest()
            node._ring_result_callback(self.message())
            node._publish_latest()
            node._ring_result_callback(self.message(9_999_999_999, 2))
            node._publish_latest()
        self.assertEqual(node._write.call_count, 1)
        with patch('time.monotonic', return_value=10.31):
            node._ring_result_callback(self.message())
            node._publish_latest()
        self.empty(node)


class PipelineTest(unittest.TestCase):
    def test_confirmation_threshold_and_instant_missing_target(self):
        node = RingPipeline(detector_backend='fast_v3')
        frame = scene()
        for _ in range(7):
            result, _, _ = node.process(frame)
            self.assertEqual(result['count'], 0)
        result, enhanced, _ = node.process(frame)
        self.assertEqual([s['id'] for s in result['sites']], [1, 2, 3])
        self.assertEqual(enhanced.shape, (480, 640))
        result, _, _ = node.process(np.full_like(frame, 220))
        self.assertEqual(result['count'], 0)
        node.reset()
        self.assertEqual(node.tracker.tracked, [])
        self.assertIsNone(node.validator.latest_gray)
        self.assertIsNone(node.validator.smoothed_sites)

    def test_static_and_moving_sequence_matches_legacy_geometry_with_less_lag(self):
        # Same real input frames and algorithms, legacy tracker/EMA coefficients.
        optimized = RingPipeline(detector_backend='fast_v3')
        legacy = RingPipeline(detector_backend='fast_v3', validator=RingValidation(smooth_alpha=0.3), roi_enabled=False)
        legacy.tracker.track_smooth_alpha = 0.5
        old_errors, new_errors, count_pairs = [], [], []
        for shift in [0]*9 + list(range(2, 20, 2)) + [18]*4:
            frame = scene(shift, slope=0.05)
            before, _, _ = legacy.process(frame)
            after, _, _ = optimized.process(frame)
            count_pairs.append((before['count'], after['count']))
            if after['count'] == 3 and before['count'] == 3:
                expected = 322+shift
                old_errors.append(abs(np.mean([s['x'] for s in before['sites']])-expected))
                new_errors.append(abs(np.mean([s['x'] for s in after['sites']])-expected))
                self.assertLess(abs(after['station_yaw_deg']-np.degrees(np.arctan(0.05))), 1)
        self.assertTrue(all(a == b for a,b in count_pairs))
        self.assertLess(np.mean(new_errors), np.mean(old_errors))

    def test_tracker_has_no_coordinate_smoothing_and_partial_model_is_optional(self):
        tracker = RingTracker()
        g = dict(x=100,y=200,r_inner=30,coverage=1,n=1)
        for _ in range(8):
            tracker._update_tracks([g])
        groups, _ = tracker._update_tracks([dict(g, x=110)])
        self.assertEqual(groups[0]['x'], 110)
        validation = RingValidation()
        result, _ = validation.evaluate(groups)
        self.assertEqual((result['count'], result['sites'][0]['id']), (1,0))
        self.assertFalse(result['yaw_valid'])

    def test_partial_counts_missing_model_and_positive_negative_yaw(self):
        for count in range(4):
            pipeline = RingPipeline(detector_backend='fast_v3')
            for _ in range(8):
                result, _, _ = pipeline.process(scene(count=count))
            self.assertEqual(result['count'], count)
            self.assertEqual(result['yaw_valid'], count >= 2)
            if count in (1, 2):
                self.assertEqual([s['id'] for s in result['sites']], [0]*count)
        for slope in (-0.08, 0.08):
            pipeline = RingPipeline(detector_backend='fast_v3')
            for _ in range(8):
                result, _, _ = pipeline.process(scene(slope=slope))
            self.assertEqual(result['count'], 3)
            self.assertLess(abs(result['station_yaw_deg']-np.degrees(np.arctan(slope))), 1.0)


class NodeInitializationTest(unittest.TestCase):
    def test_new_and_legacy_nodes_initialize_and_shutdown(self):
        import rclpy
        from logistics_ring_v2.posterior_node import PosteriorNode
        from logistics_ring_v2.validate_node import ValidateNode
        rclpy.init()
        try:
            for cls in (PosteriorNode, ValidateNode, RingPipelineNode):
                node = cls()
                try:
                    self.assertTrue(node.running)
                    if cls is RingPipelineNode:
                        self.assertEqual(node.pipeline.validator.smooth_alpha, 0.8)
                        self.assertEqual(node.pipeline.tracker.track_smooth_alpha, 1.0)
                        self.assertEqual(node.pipeline.detector_backend, 'hough')
                    elif cls is ValidateNode:
                        self.assertEqual(node.smooth_alpha, 0.3)
                    else:
                        self.assertEqual(node.track_smooth_alpha, 0.5)
                finally:
                    node.destroy_node()
        finally:
            rclpy.shutdown()


class RoiSearchTest(unittest.TestCase):
    def acquire(self, pipeline=None):
        pipeline = pipeline or RingPipeline(detector_backend='fast_v3', fast_enabled=False)
        for _ in range(8):
            result, _, _ = pipeline.process(scene())
        self.assertEqual(result['count'], 3)
        return pipeline

    def test_search_full_until_confirmed_then_crop_without_resizing(self):
        pipeline = self.acquire()
        full_pipeline = self.acquire(RingPipeline(detector_backend='fast_v3', roi_enabled=False, fast_enabled=False))
        frame = scene(shift=6)
        with patch('logistics_ring_v2.ring_pipeline.detect_circles', wraps=__import__(
                'logistics_ring_v2.ring_processing', fromlist=['detect_circles']).detect_circles) as detect:
            result, gray, _ = pipeline.process(frame)
        before, _, _ = full_pipeline.process(frame)
        self.assertEqual(result['search_mode'], 'roi')
        self.assertEqual(result['count'], 3)
        self.assertLess(result['hough_pixels'], 640*480/2)
        x0,y0,x1,y1 = result['detection_region']
        self.assertEqual(detect.call_args.args[0].shape, (y1-y0, x1-x0))
        self.assertEqual(gray.shape, (480,640))
        for a,b in zip(result['sites'],before['sites']):
            self.assertEqual(a['id'], b['id'])
            self.assertLess(abs(a['x']-b['x']), 2)
            self.assertLess(abs(a['y']-b['y']), 2)
        self.assertLess(abs(result['station_yaw_deg']-before['station_yaw_deg']), .5)
        # The next search follows measured centers, not the lagging EMA output.
        self.assertGreater(pipeline.roi_sites[0]['x'],result['sites'][0]['x'])

    def test_expand_and_full_recovery_attempts_only_update_tracks_once(self):
        for successful_call, expected in [(2,'roi_expanded'), (3,'full_recovery')]:
            pipeline = self.acquire()
            valid = [dict(g) for g in pipeline.roi_sites]
            attempts = [[]]*(successful_call-1)+[valid]
            with patch.object(pipeline,'_detect_groups',side_effect=attempts) as detect, \
                    patch.object(pipeline.tracker,'_update_tracks',wraps=pipeline.tracker._update_tracks) as update:
                result,_,_=pipeline.process(scene())
            self.assertEqual(detect.call_count, successful_call)
            self.assertEqual(update.call_count, 1)
            self.assertEqual(result['search_mode'],expected)
            self.assertEqual(result['count'],3)

    def test_loss_returns_actual_partial_or_empty_and_clears_roi(self):
        for count in (0,1,2):
            pipeline = self.acquire()
            result,_,_=pipeline.process(scene(count=count))
            self.assertEqual(result['search_mode'],'full_recovery')
            self.assertEqual(result['count'], count)
            self.assertIsNone(pipeline.roi_sites)
            result,_,_=pipeline.process(scene(count=count))
            self.assertEqual(result['search_mode'],'full')
            self.assertEqual(result['count'],count)

    def test_large_motion_recovers_globally_without_outputting_old_centers(self):
        pipeline = self.acquire()
        moved = np.roll(scene(),160,axis=0)
        result,_,_=pipeline.process(moved)
        self.assertEqual(result['search_mode'],'full_recovery')
        self.assertEqual(result['count'],0) # new tracks still require 8 confirmations
        for _ in range(7):
            result,_,_=pipeline.process(moved)
        self.assertEqual(result['count'],3)
        self.assertTrue(all(abs(s['y']-360)<2 for s in result['sites']))

    def test_periodic_full_scan_and_reset(self):
        pipeline = self.acquire(RingPipeline(detector_backend='fast_v3', roi_full_scan_interval=3, fast_enabled=False))
        modes=[pipeline.process(scene())[0]['search_mode'] for _ in range(3)]
        self.assertEqual(modes,['roi','roi','full'])
        pipeline.reset()
        self.assertIsNone(pipeline.roi_sites)
        result,_,_=pipeline.process(scene())
        self.assertEqual((result['search_mode'],result['count']),('full',0))

    def test_roi_clipping_and_validation(self):
        pipeline=RingPipeline(detector_backend='fast_v3', fast_enabled=False)
        pipeline.roi_sites=[dict(x=x,y=5,r_inner=30) for x in (5,157,309)]
        self.assertEqual(pipeline._bounds(32),(0,0,402,98))
        pipeline.roi_sites=[dict(x=x,y=475,r_inner=30) for x in (330,482,634)]
        self.assertEqual(pipeline._bounds(32),(238,383,640,480))
        with self.assertRaises(ValueError):
            RingPipeline(detector_backend='fast_v3', roi_margin_px=-1)
        with self.assertRaises(ValueError):
            RingPipeline(detector_backend='fast_v3', roi_full_scan_interval=0)


if __name__ == '__main__':
    unittest.main()
