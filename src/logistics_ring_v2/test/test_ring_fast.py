"""Actual half-size acquisition and current-frame edge tracking regression tests."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from logistics_ring_v2.ring_pipeline import RingPipeline
from logistics_ring_v2.ring_processing import detect_circles
from logistics_ring_v2.ring_fast_detector import FastRingDetector
from logistics_ring_v2.ring_serial_bridge_node import RingSerialBridgeNode


def board(shift=0, dy=0, slope=0, count=3, style='thin'):
    frame=np.full((480,640,3),225,np.uint8)
    for index in range(count):
        center=(160+152*index+shift,200+dy+round(slope*152*index))
        if style=='disk':
            cv2.circle(frame,center,30,(20,20,20),-1,cv2.LINE_AA)
        elif style=='dashed':
            for start in range(0,360,30):
                cv2.ellipse(frame,center,(30,30),0,start,start+20,(20,20,20),2,cv2.LINE_AA)
        else:
            for radius in (30,50):
                cv2.circle(frame,center,radius,(20,20,20),1 if style=='thin' else 2,cv2.LINE_AA)
    return frame


class FastDetectorTest(unittest.TestCase):
    def acquire(self, style='disk', **kwargs):
        pipeline=RingPipeline(**kwargs)
        for _ in range(8):
            result,_,_=pipeline.process(board(style=style))
        self.assertEqual(result['count'],3)
        return pipeline

    def test_half_size_hough_parameters_and_original_refinement(self):
        detector=FastRingDetector()
        gray=cv2.cvtColor(board(style='thin'),cv2.COLOR_BGR2GRAY)
        with patch('logistics_ring_v2.ring_fast_detector.detect_circles',wraps=detect_circles) as detect:
            groups,pixels,_=detector.detect(gray)
        self.assertEqual(detect.call_args.args[0].shape,(240,320))
        self.assertEqual(detect.call_args.kwargs,dict(min_dist=30,min_radius=10,max_radius=30))
        self.assertEqual(pixels,76800)
        self.assertEqual(len(groups),3)
        for i,g in enumerate(sorted(groups,key=lambda g:g['x'])):
            self.assertLess(abs(g['x']-(160+152*i)),1.5)
            self.assertLess(abs(g['y']-200),1.5)

    def test_stable_tracking_never_calls_hough_and_follows_motion(self):
        pipeline=self.acquire(roi_full_scan_interval=50)
        for shift in (4,10,20,30):
            with patch('logistics_ring_v2.ring_fast_detector.detect_circles',side_effect=AssertionError('Hough during tracking')):
                result,_,_=pipeline.process(board(shift=shift,style='disk'))
            self.assertEqual((result['search_mode'],result['count'],result['hough_pixels']),('edge_track',3,0))
            self.assertLess(abs(np.mean([s['x'] for s in result['sites']])-(312+shift)),3)
            self.assertEqual([s['id'] for s in result['sites']],[1,2,3])

    def test_thin_thick_and_dashed_ring_acquisition(self):
        for style in ('thin','thick','dashed'):
            pipeline=self.acquire(style=style)
            result,_,_=pipeline.process(board(shift=5,style=style))
            self.assertEqual(result['count'],3,style)
            self.assertLess(abs(np.mean([s['x'] for s in result['sites']])-317),3,style)

    def test_current_empty_or_partial_never_reuses_missing_centers(self):
        for count in (0,1,2):
            pipeline=self.acquire()
            result,_,_=pipeline.process(board(count=count,style='disk'))
            self.assertEqual((result['search_mode'],result['count']),('fast_recovery',count))
            self.assertIsNone(pipeline.roi_sites)
            if count:
                self.assertTrue(all(s['id']==0 for s in result['sites']))

    def test_recovery_once_per_frame_periodic_search_and_mode_reset(self):
        pipeline=self.acquire(roi_full_scan_interval=3)
        for expected in ('edge_track','edge_track','fast_search'):
            self.assertEqual(pipeline.process(board(style='disk'))[0]['search_mode'],expected)
        with patch.object(pipeline.tracker,'_update_tracks',wraps=pipeline.tracker._update_tracks) as update:
            result,_,_=pipeline.process(board(dy=160,style='disk'))
        self.assertEqual(update.call_count,1)
        self.assertEqual((result['search_mode'],result['count']),('fast_recovery',0))
        for _ in range(7):
            result,_,_=pipeline.process(board(dy=160,style='disk'))
        self.assertEqual(result['count'],3)
        self.assertTrue(all(abs(s['y']-360)<2 for s in result['sites']))
        pipeline.reset()
        self.assertEqual(pipeline.process(board(style='disk'))[0]['count'],0)

    def test_negative_positive_yaw_original_coordinates(self):
        for slope in (-.08,.08):
            pipeline=RingPipeline()
            for _ in range(8):
                result,_,_=pipeline.process(board(slope=slope,style='thin'))
            self.assertEqual(result['count'],3)
            self.assertLess(abs(result['station_yaw_deg']-np.degrees(np.arctan(slope))),1)

    def test_geometry_rejection_and_no_edges_reject_old_seed(self):
        detector=FastRingDetector()
        pipeline=self.acquire()
        groups,_,_=detector.detect(np.full((480,640),225,np.uint8),pipeline.roi_sites)
        self.assertEqual(groups,[])
        frame=board(style='disk')
        # Move only the middle circle off the shared line.
        cv2.circle(frame,(312,200),35,(225,225,225),-1)
        cv2.circle(frame,(312,300),30,(20,20,20),-1)
        result,_,_=pipeline.process(frame)
        self.assertLess(result['count'],3)
        self.assertTrue(all(s['id']==0 for s in result['sites']))
        self.assertFalse(any(abs(s['x']-312)<10 and abs(s['y']-200)<10 for s in result['sites']))

    def test_ring_timer_speedup_is_separate_from_turntable_timer(self):
        from unittest.mock import Mock
        node=RingSerialBridgeNode.__new__(RingSerialBridgeNode)
        node._publish_latest=Mock()
        for mode in ('ring','turntable','raw'):
            node.active_mode=mode
            node._publish_latest.reset_mock()
            node._publish_ring()
            self.assertEqual(node._publish_latest.call_count, int(mode=='ring'))
            node._publish_latest.reset_mock()
            node._publish_other_modes()
            self.assertEqual(node._publish_latest.call_count, int(mode!='ring'))

    def test_default_bridge_has_30hz_ring_and_10hz_other_modes(self):
        import rclpy
        rclpy.init()
        node=RingSerialBridgeNode()
        try:
            self.assertAlmostEqual(node.ring_publish_period,1/30)
            self.assertAlmostEqual(node.publish_period,.1)
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__=='__main__':
    unittest.main()
