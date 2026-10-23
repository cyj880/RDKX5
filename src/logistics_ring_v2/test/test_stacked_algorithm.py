import unittest

import cv2
import numpy as np

from logistics_ring_v2.ring_geometry import select_three
from logistics_ring_v2.stacked_ring_algorithm import StackedTracker


def scene(circles=True, top=True, bottom=True):
    frame = np.full((480, 640, 3), 255, np.uint8)
    if circles:
        for x in (130, 280, 430):
            for radius in (40, 22):
                cv2.circle(frame, (x, 240), radius, (0, 0, 0), 2)
    if top:
        cv2.line(frame, (20, 177), (620, 177), (0, 0, 0), 3)
    if bottom:
        cv2.line(frame, (20, 303), (620, 303), (0, 0, 0), 3)
    return frame


class StackedAlgorithmTest(unittest.TestCase):
    def test_three_required_and_reinitialize(self):
        tracker = StackedTracker(confirm_frames=3)
        for _ in range(4):
            result, _ = tracker.process(scene(circles=False))
            self.assertFalse(result['detected'])
        for _ in range(7):
            result, _ = tracker.process(scene())
        self.assertTrue(result['stable'])
        self.assertAlmostEqual(result['middle_center_px']['x'], 280, delta=2)
        self.assertEqual(result['side_centers_inferred'][0]['source'], 'inferred')
        result, _ = tracker.process(scene(circles=False))
        self.assertFalse(result['detected'])
        self.assertFalse(result['stable'])
        self.assertIsNone(result['middle_center_px'])
        result, _ = tracker.process(scene())
        self.assertFalse(result['detected'])

    def test_one_line_and_missing_lines(self):
        tracker = StackedTracker(confirm_frames=2)
        for _ in range(3):
            tracker.process(scene())
        result, _ = tracker.process(scene(top=False))
        self.assertTrue(result['detected'])
        self.assertIsNone(result['top_line_angle_deg'])
        self.assertIsNotNone(result['bottom_line_angle_deg'])
        result, _ = tracker.process(scene(top=False, bottom=False))
        self.assertFalse(result['detected'])
        self.assertFalse(result['stable'])
        self.assertIsNone(result['station_yaw_deg'])

    def test_conflicting_lines_and_geometry(self):
        tracker = StackedTracker(confirm_frames=2)
        for _ in range(3):
            tracker.process(scene())
        frame = scene(bottom=False)
        cv2.line(frame, (20, 277), (620, 329), (0, 0, 0), 3)
        result, _ = tracker.process(frame)
        self.assertFalse(result['stable'])
        self.assertFalse(result['detected'])
        self.assertIn('冲突', result['failure_reason'])
        groups = [{'x': 130., 'y': 240.}, {'x': 280., 'y': 240.},
                  {'x': 430., 'y': 240.}]
        triple, _, _ = select_three(groups)
        self.assertEqual(triple[1]['x'], 280.)
        self.assertIsNone(select_three(groups[:2])[0])


if __name__ == '__main__':
    unittest.main()
