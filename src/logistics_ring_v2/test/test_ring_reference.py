"""Reference algorithm integration, immediate measurements and wire compatibility."""
import json
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from logistics_ring_v2.ring_pipeline import RingPipeline
from logistics_ring_v2.reference_ring_detector import ReferenceRingDetector, Ring
from logistics_ring_v2.ring_pipeline_node import RingPipelineNode
from logistics_ring_v2.ring_serial_protocol import encode_identified_ring_result, FrameParser


def rings(count=3, shift=0, slope=0, arc=360, hollow=True):
    image=np.full((480,640,3),240,np.uint8)
    for i in range(count):
        center=(160+i*152+shift,210+round(slope*i*152))
        for radius in (30,50):
            if not hollow:
                cv2.circle(image,center,radius,(25,25,25),-1)
            else:
                cv2.ellipse(image,center,(radius,radius),0,0,arc,(25,25,25),2,cv2.LINE_AA)
    return image


class ReferenceRingTest(unittest.TestCase):
    def test_default_native_size_and_first_frame_full_result(self):
        pipeline=RingPipeline(detector_backend='reference')
        frame=rings()
        work,scale=pipeline.reference_detector._prepare(frame)
        self.assertIs(work,frame)
        self.assertEqual(scale,1)
        result,gray,_=pipeline.process(frame)
        self.assertEqual(result['count'],3)
        self.assertEqual([g['id'] for g in result['sites']],[1,2,3])
        self.assertEqual(result['detector_version'],'reference-v4')
        self.assertEqual(gray.shape,(480,640))
        self.assertTrue(result['yaw_valid'])
        self.assertLess(abs(result['station_yaw_deg']),1)
        self.assertEqual(pipeline.tracker.tracked,[]) # never enters old association path

    def test_large_move_does_not_restart_confirmation(self):
        pipeline=RingPipeline(detector_backend='reference')
        a,_,_=pipeline.process(rings())
        b,_,_=pipeline.process(rings(shift=40))
        self.assertEqual((a['count'],b['count']),(3,3))
        self.assertGreater(b['sites'][0]['x']-a['sites'][0]['x'],30)
        self.assertEqual([g['id'] for g in b['sites']],[1,2,3])

    def test_partial_ids_missing_model_and_disappearance(self):
        pipeline=RingPipeline(detector_backend='reference')
        for count in (3,2,1,0):
            result,_,_=pipeline.process(rings(count=count))
            self.assertEqual(result['count'],count)
            self.assertEqual(result['yaw_valid'],count>=2)
            if count in (1,2):
                self.assertEqual([g['id'] for g in result['sites']],[0]*count)

    def test_yaw_sign_and_wire_format_preserved(self):
        for slope in (-.08,.08):
            pipeline=RingPipeline(detector_backend='reference')
            result,_,_=pipeline.process(rings(slope=slope))
            self.assertEqual(result['count'],3)
            self.assertLess(abs(result['station_yaw_deg']-np.degrees(np.arctan(slope))),1)
            command,payload=FrameParser().feed(encode_identified_ring_result(result['sites']))[0]
            self.assertEqual((command,payload[0],len(payload)),(0x13,3,19))

    def test_hough_failure_uses_contours_and_hollow_rejects_disk(self):
        detector=ReferenceRingDetector(min_work_width=640,min_work_height=480,
                                      min_radius=20,max_radius=60,center_merge_dist=20,
                                      radius_merge_tol=3,min_rings_per_cluster=2)
        with patch.object(detector,'_detect_hough',return_value=[]), \
                patch.object(detector,'_detect_contours',wraps=detector._detect_contours) as fallback:
            output=detector.detect(rings())
        self.assertEqual(fallback.call_count,1)
        self.assertEqual(len(output),3)
        self.assertEqual(detector.detect(rings(hollow=False)),[])

    def test_same_reference_defaults_for_hough_and_perspective_gate(self):
        detector=RingPipeline(detector_backend='reference').reference_detector
        self.assertEqual((detector.hough_dp,detector.hough_param2,detector.hollow_threshold),(1.5,.8,15))
        self.assertEqual(detector.smooth_alpha,0)
        self.assertTrue(detector.refine_center)
        # Known ellipse points: short/long = .5, rejected by old fast-v3 gate.
        gray=np.full((480,640),240,np.uint8)
        cv2.ellipse(gray,(300,200),(50,25),0,0,360,20,2,cv2.LINE_AA)
        self.assertIsNotNone(detector._fit_ellipse_center(gray,300,200,42))

    def test_geometry_and_partial_digit_rules_remain(self):
        pipeline=RingPipeline(detector_backend='reference')
        # Reference's ring_no=1 is never used as a fixed station identity.
        measured=[Ring(1,100,200,90,30),Ring(1,252,200,90,30)]
        pipeline.validator._recognize_identity=Mock(side_effect=[(2,.9),(3,.9)])
        with patch.object(pipeline.reference_detector,'detect',return_value=measured):
            result,_,_=pipeline.process(rings())
        self.assertEqual([s['id'] for s in result['sites']],[2,3])
        with patch.object(pipeline.reference_detector,'detect',return_value=[
                Ring(1,50,50,90,30),Ring(1,80,400,90,30),Ring(1,550,420,90,30)]):
            result,_,_=pipeline.process(rings())
        self.assertEqual(result['count'],0)

    def test_reference_reset_discards_smoothing(self):
        pipeline=RingPipeline(detector_backend='reference')
        pipeline.process(rings())
        self.assertIsNotNone(pipeline.validator.smoothed_sites)
        pipeline.reset()
        self.assertIsNone(pipeline.validator.smoothed_sites)
        self.assertEqual(pipeline.reference_detector._prev_rings,[])

    def test_default_restores_full_resolution_hough(self):
        from test_ring_latency import scene
        pipeline=RingPipeline()
        self.assertEqual(pipeline.detector_backend,'hough')
        self.assertFalse(pipeline.fast_enabled)
        self.assertFalse(pipeline.roi_enabled)
        self.assertIsNone(pipeline.reference_detector)
        with patch('logistics_ring_v2.ring_pipeline.detect_circles', wraps=__import__(
                'logistics_ring_v2.ring_processing',fromlist=['detect_circles']).detect_circles) as detect:
            for _ in range(9):
                result,_,_=pipeline.process(scene())
        self.assertEqual(result['count'],3)
        self.assertEqual(result['detector_version'],'hough-restored')
        self.assertEqual(result['search_mode'],'full')
        self.assertEqual(result['hough_pixels'],640*480)
        self.assertTrue(all(c.args[0].shape==(480,640) for c in detect.call_args_list))

    def test_invalid_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            RingPipeline(detector_backend='unknown')


if __name__=='__main__':
    unittest.main()
