"""Actual detector/bridge callbacks; no camera or serial hardware required."""
import json
import struct
from collections import deque
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from std_msgs.msg import String

from logistics_ring_v2.ring_serial_protocol import FrameParser, encode_mode
from logistics_ring_v2.ring_serial_bridge_node import RingSerialBridgeNode
from logistics_ring_v2.validate_node import ValidateNode


def group(x, y):
    return dict(x=x, y=y, r_inner=30, coverage=0.95, n=3)


class RingResultsTest(unittest.TestCase):
    def validator(self):
        node = ValidateNode.__new__(ValidateNode)
        node.running = True
        node.expected_spacing_px = 150.0
        node.spacing_tolerance = 0.2
        node.spacing_ratio_min = 0.85
        node.max_collinearity = 0.08
        node.smooth_alpha = 0.3
        node.snap_dist_px = 15.0
        node.smoothed_sites = node.smoothed_yaw = None
        node.digit_available = False
        node.latest_gray = None
        node.publisher = Mock()
        node.get_logger = Mock(return_value=Mock())
        node._publish_annotated = Mock()
        return node

    def bridge(self):
        node = RingSerialBridgeNode.__new__(RingSerialBridgeNode)
        node.active_mode = 'ring'
        node.latest_ring_time = None
        node.last_ring_source_ns = 0
        node.ring_mode_since_ns = 0
        node.ring_tx_ages = deque(maxlen=512)
        node.ring_stats_time = float('inf')
        node.latest_ring_frame = b''
        node.ring_pending = False
        node.latest_turntable_center = None
        node._write = Mock()
        return node

    def result(self, node, groups):
        node._groups_callback(String(data=json.dumps({'groups': groups})))
        return json.loads(node.publisher.publish.call_args.args[0].data)

    def test_partial_without_digits_full_and_disappearance(self):
        node = self.validator()  # No digit model loaded or digit recognizer set.
        for count in range(4):
            result = self.result(node, [group(100 + 150 * i, 200 + 15 * i)
                                        for i in reversed(range(count))])
            self.assertEqual(result['count'], count)
            self.assertEqual(result['yaw_valid'], count >= 2)
            self.assertEqual(result['status'], ('none', 'partial', 'partial', 'full')[count])
            self.assertEqual(result['sites'], sorted(result['sites'], key=lambda g: (g['x'], g['y'])))
        result = self.result(node, [])
        self.assertEqual((result['count'], result['station_yaw_deg']), (0, None))

    def test_many_candidates_require_a_valid_triple(self):
        node = self.validator()
        result = self.result(node, [group(50, 200), group(200, 200),
                                    group(350, 200), group(500, 400)])
        self.assertEqual(result['count'], 3)
        result = self.result(node, [group(50, 50), group(70, 60),
                                    group(80, 400), group(550, 420)])
        self.assertEqual(result['count'], 0)

    def test_actual_validator_to_bridge_and_stale_result(self):
        detector, bridge = self.validator(), self.bridge()
        result = self.result(detector, [group(100, 240), group(250, 225)])
        result.update(frame_sequence=1, ingress_monotonic_ns=10_000_000_000)
        with patch('logistics_ring_v2.ring_serial_bridge_node.time.monotonic', return_value=10.0):
            bridge._ring_result_callback(String(data=json.dumps(result)))
            bridge._publish_latest()
        command, payload = FrameParser().feed(bridge._write.call_args.args[0])[0]
        self.assertEqual(command, 0x13)
        self.assertEqual(struct.unpack('<BBhBHHBHH', payload), (2, 1, -571, 0, 100, 240, 0, 250, 225))
        with patch('logistics_ring_v2.ring_serial_bridge_node.time.monotonic', return_value=10.301):
            bridge._publish_latest()
        self.assertEqual(FrameParser().feed(bridge._write.call_args.args[0])[0][1], bytes(4))

    def test_zero_invalid_and_wrong_mode_never_reuse_old_coordinates(self):
        bridge = self.bridge()
        for text in ['{"count":0,"sites":[]}', '{bad', '{"count":1,"sites":[]}',
                     '{"count":1,"sites":[{"x":640,"y":1}]}']:
            bridge._ring_result_callback(String(data=text))
            bridge._publish_latest()
            self.assertEqual(FrameParser().feed(bridge._write.call_args.args[0])[0][1], bytes(4))
        bridge.active_mode = 'raw'
        bridge._write.reset_mock()
        bridge._publish_latest()
        bridge._write.assert_not_called()

    def test_fixed_ids_full_partial_unknown_and_conflicts(self):
        node = self.validator()
        result = self.result(node, [group(100, 240), group(250, 240), group(400, 240)])
        self.assertEqual([s['id'] for s in result['sites']], [1, 2, 3])
        result = self.result(node, [group(100, 240)])
        self.assertEqual(result['sites'][0]['id'], 0)
        for identities in ([1], [3], [1, 2], [2, 3]):
            node._recognize_identity = Mock(side_effect=[(i, 0.9) for i in identities])
            result = self.result(node, [group(100 + 150 * i, 240) for i in range(len(identities))])
            self.assertEqual([s['id'] for s in result['sites']], identities)
        for identities in ([1, 1], [3, 1]):
            node._recognize_identity = Mock(side_effect=[(i, 0.9) for i in identities])
            result = self.result(node, [group(100, 240), group(250, 240)])
            self.assertEqual([s['id'] for s in result['sites']], [0, 0])

    def test_cached_result_is_never_repeated_as_a_fresh_observation(self):
        bridge = self.bridge()
        with patch('logistics_ring_v2.ring_serial_bridge_node.time.monotonic', return_value=10.0):
            bridge._ring_result_callback(String(data=json.dumps(dict(
                count=1, sites=[dict(id=3, x=100, y=240)], frame_sequence=1,
                ingress_monotonic_ns=10_000_000_000))))
            bridge._publish_latest()
            bridge._publish_latest()
            bridge._publish_latest()
        self.assertEqual(bridge._write.call_count, 1)
        with patch('logistics_ring_v2.ring_serial_bridge_node.time.monotonic', return_value=10.301):
            bridge._publish_latest()
        self.assertEqual(FrameParser().feed(bridge._write.call_args.args[0])[0], (0x13, bytes(4)))

    def test_actual_digit_crop_confidence_and_missing_model_fallback(self):
        import numpy as np
        node = self.validator()
        self.assertEqual(node._recognize_identity(group(100, 240))[0], 0)
        node.digit_available = True
        node.digit_conf_min = 0.6
        node.latest_gray = np.zeros((480, 640), dtype=np.uint8)
        node.hog = node.svm = Mock()
        features = SimpleNamespace(
            binarize_digit=lambda gray: gray.copy(), normalize_mask=lambda gray: gray,
            hog_features=Mock(return_value=np.zeros(10)),
            svm_class_scores=Mock(return_value=np.array([0.1, 0.2, 0.9])))
        node.digit_features = features
        self.assertEqual(node._recognize_identity(group(100, 240)), (3, 0.9))
        features.svm_class_scores.return_value = np.array([0.3, 0.5, 0.2])
        self.assertEqual(node._recognize_identity(group(100, 240))[0], 0)
        features.normalize_mask = lambda gray: None
        self.assertEqual(node._recognize_identity(group(100, 240))[0], 0)

    def test_bytewise_mode_input_resets_cached_results_and_raw_stops_tx(self):
        bridge = self.bridge()
        bridge.parser = FrameParser()
        bridge.last_serial_rx = None
        bridge._open_serial = Mock()
        bridge.serial_handle = Mock()
        bridge.serial_handle.in_waiting = 1
        bridge.get_logger = Mock(return_value=Mock())
        bridge.mode_publisher = Mock()
        for mode in (1, 0):
            for byte in encode_mode(mode):
                bridge.serial_handle.read.return_value = bytes((byte,))
                bridge._poll_serial()
            self.assertEqual(bridge.active_mode, 'ring' if mode else 'raw')
            self.assertIsNone(bridge.latest_ring_time)
        self.assertEqual([c.args[0].data for c in bridge.mode_publisher.publish.call_args_list],
                         ['ring', 'raw'])

    def test_turntable_wire_format_unchanged(self):
        bridge = self.bridge()
        bridge.active_mode = 'turntable'
        bridge._turntable_result_callback(String(data='{"detected":true,"center_u":320,"center_v":240}'))
        bridge._publish_latest()
        bridge._write.assert_called_once_with(bytes.fromhex('66 01 01 40 00 F0 77'))

    def test_half_frame_timeout_does_not_poison_next_command(self):
        bridge = self.bridge()
        bridge.parser = FrameParser()
        bridge.last_serial_rx = None
        bridge._open_serial = Mock()
        bridge.serial_handle = Mock()
        bridge.serial_handle.in_waiting = 20
        bridge.mode_publisher = Mock()
        bridge.get_logger = Mock(return_value=Mock())
        with patch('logistics_ring_v2.ring_serial_bridge_node.time.monotonic', return_value=1.0):
            bridge.serial_handle.read.return_value = encode_mode(1)[:4]
            bridge._poll_serial()
        with patch('logistics_ring_v2.ring_serial_bridge_node.time.monotonic', return_value=1.1):
            bridge.serial_handle.read.return_value = encode_mode(1)
            bridge._poll_serial()
        bridge.mode_publisher.publish.assert_called_once()
        self.assertEqual(bridge.active_mode, 'ring')

    def test_partial_write_closes_port_and_reconnects(self):
        bridge = self.bridge()
        bridge._write = RingSerialBridgeNode._write.__get__(bridge)
        bridge.serial_port, bridge.baudrate = 'mock', 115200
        bridge.serial_handle = None
        bridge.last_open_error = ''
        bridge.tx_reported = False
        bridge.get_logger = Mock(return_value=Mock())
        failed, recovered = Mock(), Mock()
        failed.write.return_value = 2
        recovered.write.side_effect = len
        serial_api = SimpleNamespace(SerialException=OSError, SerialTimeoutException=OSError,
                                     Serial=Mock(side_effect=[failed, recovered]))
        with patch('logistics_ring_v2.ring_serial_bridge_node.serial', serial_api):
            bridge._write(b'1234567')
            failed.close.assert_called_once()
            self.assertIsNone(bridge.serial_handle)
            bridge._write(b'1234567')
            self.assertIs(bridge.serial_handle, recovered)
            self.assertTrue(bridge.tx_reported)
        self.assertEqual(serial_api.Serial.call_count, 2)


if __name__ == '__main__':
    unittest.main()
