import unittest
from unittest.mock import patch

import numpy as np
import cv2
import rclpy
from std_msgs.msg import String

from logistics_ring_v2.material_pipeline import MaterialPipeline, BlobTrack, PipelineFrame
from logistics_ring_v2.material_track_detect_node import MaterialTrackDetectNode
from logistics_ring_v2.mode_gate_node import ModeGateNode


class MaterialSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_speed_band_and_missing(self):
        p = MaterialPipeline.__new__(MaterialPipeline)
        p.moving_speed_px_s = 40.
        p.stillness_speed_px_s = 10.
        p.motion_confirm_frames = 2
        p.stillness_frames = 6
        p.initial_still_frames = 60
        p.reset()
        for speed in [50., 50.] + [25.] * 60:
            p.motion_speed = speed
            p._update_stillness()
        self.assertFalse(p.stationary)
        for _ in range(6):
            p.motion_speed = 0.
            p._update_stillness()
        self.assertTrue(p.stationary)
        p.motion_speed = None
        p._update_stillness()
        self.assertFalse(p.stationary)
        p.reset()
        for _ in range(60):
            p.motion_speed = 0.
            p._update_stillness()
        self.assertTrue(p.stationary)

    def test_invalid_mode_does_not_change_gate(self):
        node = ModeGateNode()
        try:
            node.ready = True
            with patch.object(node, '_select') as select:
                node._command(String(data='material_track 99'))
                select.assert_not_called()
                node._command(String(data='material_track 2'))
                select.assert_called_once_with('material_track', 'track 2')
        finally:
            node.destroy_node()

    def test_cache_reset_expiry_and_no_target(self):
        node = MaterialTrackDetectNode()
        try:
            node.turntable_cache = (320, 240, 180.)
            node._command_callback(String(data='track 2'))
            self.assertIsNone(node.turntable_cache)
            node.pipeline.stationary = True
            node._frame_count = 2
            with patch('logistics_ring_v2.material_track_detect_node.time.monotonic', return_value=10.), \
                    patch.object(node, '_detect_turntable_circle', return_value=(320, 240, 180.)):
                node._maybe_update_turntable(np.zeros((480, 640, 3), np.uint8))
            self.assertIsNotNone(node.turntable_cache)
            with patch('logistics_ring_v2.material_track_detect_node.time.monotonic', return_value=12.), \
                    patch.object(node, '_detect_turntable_circle', return_value=None):
                node._maybe_update_turntable(np.zeros((480, 640, 3), np.uint8))
            self.assertIsNone(node.turntable_cache)
            frame = PipelineFrame([], True, 0., None, None)
            track = BlobTrack(1, (320, 240), 1000, (300, 220, 40, 40))
            result = node._fill(frame, [(track, 2, 1.)])
            self.assertFalse(result['grasp_ready'])
            self.assertIsNone(result['target'])
            self.assertEqual(result['turntable_state'], 'lost')
            self.assertFalse(node._fill(frame, [])['grasp_ready'])
        finally:
            node.destroy_node()

    def test_optical_distance_and_coordinate_output(self):
        node = MaterialTrackDetectNode()
        try:
            node.measurement_matrix = np.array([[400., 0, 320], [0, 400., 240], [0, 0, 1]])
            node.track_color = 2
            track = BlobTrack(1, (320, 240), 1000, (300, 220, 40, 40))
            frame = PipelineFrame([], True, 0., None, None)
            for expected in (200., 300., 400.):
                node.turntable_cache = (320, 240, 400. * 150. / expected)
                result = node._fill(frame, [(track, 2, 1.)])
                self.assertTrue(result['distance_valid'])
                self.assertAlmostEqual(node.pipeline.camera_to_tray_mm, expected)
                for item in (result, result['materials'][0], result['target']):
                    self.assertEqual(item['camera_to_turntable_distance_mm'], expected)
                self.assertTrue(result['grasp_ready'])
            node.turntable_cache = None
            result = node._fill(frame, [(track, 2, 1.)])
            self.assertIsNone(result['camera_to_turntable_distance_mm'])
            self.assertIsNone(result['materials'][0]['center_u'])
            self.assertFalse(result['distance_valid'])
            self.assertIsNone(result['target'])
            node.turntable_cache = (320, 240, float('nan'))
            self.assertIsNone(node._optical_distance())
            node.turntable_cache = (320, 240, 200.)
            node.measurement_matrix = None
            self.assertIsNone(node._optical_distance())
        finally:
            node.destroy_node()

    def test_synthetic_disk_distance(self):
        node = MaterialTrackDetectNode()
        try:
            node.measurement_matrix = np.array([[400., 0, 320], [0, 400., 240], [0, 0, 1]])
            for expected in (250., 300., 375.):
                image = np.full((640, 640, 3), 255, np.uint8)
                cv2.circle(image, (320, 320), round(400. * 150. / expected),
                           (0, 0, 0), 3, cv2.LINE_AA)
                node.turntable_cache = node._detect_turntable_circle(image)
                distance = node._optical_distance()
                self.assertIsNotNone(distance)
                self.assertAlmostEqual(distance, expected, delta=expected * .05)
        finally:
            node.destroy_node()


if __name__ == '__main__':
    unittest.main()
