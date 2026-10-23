"""Exercise actual UART parser -> mode gate startup -> ring worker activation."""
import unittest
import json
from unittest.mock import Mock, patch

import rclpy
from std_msgs.msg import String

from logistics_ring_v2.mode_gate_node import ModeGateNode
from logistics_ring_v2.ring_pipeline_node import RingPipelineNode
from logistics_ring_v2.ring_serial_bridge_node import RingSerialBridgeNode
from logistics_ring_v2.ring_serial_protocol import encode_mode, FrameParser
from logistics_ring_v2.debug_display_node import DebugDisplayNode


class ModeStartupTest(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.gate = ModeGateNode()
        self.gate.gate_publishers = {mode: Mock() for mode in self.gate.commands}
        for pub in self.gate.gate_publishers.values():
            pub.get_subscription_count.return_value = 1
        self.gate.state_pub = Mock()
        self.gate.timer = Mock()
        self.ring = RingPipelineNode.__new__(RingPipelineNode)
        self.ring.running = False
        self.ring.worker = Mock()
        self.ring.preview = None
        self.ring.get_logger = Mock(return_value=Mock())
        self.gate.gate_publishers['ring'].publish.side_effect = self.ring._command

    def tearDown(self):
        self.gate.destroy_node()
        rclpy.shutdown()

    def serial(self):
        bridge = RingSerialBridgeNode.__new__(RingSerialBridgeNode)
        bridge.parser = FrameParser()
        bridge.last_serial_rx = None
        bridge._open_serial = Mock()
        bridge.serial_handle = Mock()
        bridge.serial_handle.in_waiting = 1
        bridge.get_logger = Mock(return_value=Mock())
        bridge.mode_publisher = Mock()
        bridge.mode_publisher.publish.side_effect = self.gate._command
        return bridge

    def feed(self, bridge, mode):
        for byte in encode_mode(mode):
            bridge.serial_handle.read.return_value = bytes([byte])
            bridge._poll_serial()

    def test_single_ring_subscriber_allows_startup_raw(self):
        self.gate._startup()
        self.assertTrue(self.gate.ready)
        self.assertEqual(self.gate.active, 'raw')
        self.gate.timer.cancel.assert_not_called()
        self.assertFalse(self.ring.running)

    def test_early_serial_ring_is_replayed_after_one_subscriber_ready(self):
        bridge = self.serial()
        self.feed(bridge, 1)
        self.assertEqual(self.gate.pending_command, 'ring')
        self.assertFalse(self.ring.running)
        self.gate._startup()
        self.assertTrue(self.gate.ready)
        self.assertEqual(self.gate.active, 'ring')
        self.assertTrue(self.ring.running)
        self.ring.worker.start.assert_called_once()
        self.feed(bridge, 0)
        self.assertEqual(self.gate.active, 'raw')
        self.assertFalse(self.ring.running)
        self.ring.worker.stop.assert_called()

    def test_command_after_startup_works_and_other_gates_close(self):
        self.gate._startup()
        bridge = self.serial()
        self.feed(bridge, 1)
        self.assertTrue(self.ring.running)
        for mode,pub in self.gate.gate_publishers.items():
            self.assertEqual(pub.publish.call_args.args[0].data, 'run' if mode=='ring' else 'stop')
        self.assertEqual(self.gate.state_pub.publish.call_args.args[0].data, 'ring')

    def test_missing_other_detectors_do_not_block_ring(self):
        for missing in ('turntable','material_all','material_track','stacked'):
            with self.subTest(missing=missing):
                self.gate.ready = False
                for pub in self.gate.gate_publishers.values():
                    pub.get_subscription_count.return_value = 1
                self.gate.gate_publishers[missing].get_subscription_count.return_value = 0
                self.gate._command(String(data='ring'))
                self.gate._startup()
                self.assertTrue(self.gate.ready)
                self.assertEqual(self.gate.active, 'ring')
                self.assertTrue(self.ring.running)

    def test_missing_target_waits_then_replays_and_raw_cancels(self):
        self.gate.gate_publishers['ring'].get_subscription_count.return_value = 0
        self.gate._command(String(data='ring'))
        self.gate._startup()
        self.assertTrue(self.gate.ready)
        self.assertEqual(self.gate.active, 'raw')
        self.assertEqual(self.gate.pending_selection, ('ring','run'))
        self.assertFalse(self.ring.running)
        self.gate._startup()
        self.assertFalse(self.ring.running)
        self.gate.gate_publishers['ring'].get_subscription_count.return_value = 1
        self.gate._startup()
        self.assertTrue(self.ring.running)
        self.assertIsNone(self.gate.pending_selection)
        self.gate.gate_publishers['ring'].get_subscription_count.return_value = 0
        self.gate._command(String(data='ring'))
        self.gate._command(String(data='raw'))
        self.gate.gate_publishers['ring'].get_subscription_count.return_value = 1
        self.gate._startup()
        self.assertFalse(self.ring.running)

    def test_raw_startup_needs_no_detectors(self):
        for pub in self.gate.gate_publishers.values():
            pub.get_subscription_count.return_value = 0
        self.gate._startup()
        self.assertTrue(self.gate.ready)
        self.assertEqual(self.gate.active, 'raw')

    def test_keyboard_r_uses_same_gate_and_display_switches_topic(self):
        display = DebugDisplayNode.__new__(DebugDisplayNode)
        display.active='raw'
        display.track_color=5
        display.enabled=True
        display.frames={}
        display.results={}
        display.image_subscription=None
        display.topics=dict(raw='/vision/camera/image_undistorted', ring='/vision/ring2/annotated')
        display.get_logger=Mock(return_value=Mock())
        display.create_subscription=Mock()
        display.destroy_subscription=Mock()
        display.mode_command_pub=Mock()
        display.mode_command_pub.publish.side_effect=self.gate._command
        self.gate.state_pub.publish.side_effect=display._mode
        self.gate.gate_publishers['material_all'].get_subscription_count.return_value=0
        display._handle_key(ord('r'))
        self.gate._startup()
        self.assertTrue(self.ring.running)
        self.assertEqual(display.active,'ring')
        self.assertEqual(display.create_subscription.call_args.args[1],'/vision/ring2/annotated')
        self.assertIsNone(display.pending_command)
        display._handle_key(ord('i'))
        self.assertEqual(display.active,'raw')
        self.assertFalse(self.ring.running)

    def test_uppercase_keyboard_and_wait_indicator_preserve_request(self):
        display=DebugDisplayNode.__new__(DebugDisplayNode)
        display.active='raw'
        display.track_color=5
        display.topics={'raw':'/vision/camera/image_undistorted','ring':'/vision/ring2/annotated'}
        display.image_subscription=None
        display.enabled=False
        display.frames={}
        display.results={}
        display.get_logger=Mock(return_value=Mock())
        display.mode_command_pub=Mock()
        display._handle_key(ord('R'))
        self.assertEqual(display.pending_command,'ring')
        display._mode(String(data='raw'))
        self.assertEqual(display.pending_command,'ring')
        display._handle_key(ord('i'))
        self.assertEqual(display.mode_command_pub.publish.call_args.args[0].data,'raw')
        display._handle_key(ord('R'))
        display._mode(String(data='ring'))
        self.assertIsNone(display.pending_command)

    def test_keyboard_executed_mode_enables_bridge_and_raw_stops_return(self):
        bridge = self.serial()
        bridge.active_mode = 'raw'
        bridge._write = Mock()
        self.gate.state_pub.publish.side_effect = bridge._active_mode_callback
        # Only the requested ring node is running; others may have failed.
        self.gate.gate_publishers['material_all'].get_subscription_count.return_value = 0
        self.gate._startup()
        with patch('time.monotonic', return_value=10.0):
            self.gate._command(String(data='ring'))
        self.assertEqual(bridge.active_mode, 'ring')
        self.assertTrue(self.ring.running)
        result=dict(count=1,sites=[dict(id=1,x=100,y=200)], frame_sequence=1,
                    ingress_monotonic_ns=10_010_000_000)
        bridge.ring_tx_ages=[]
        bridge.ring_stats_time=float('inf')
        with patch('time.monotonic',return_value=10.02):
            bridge._ring_result_callback(String(data=json.dumps(result)))
            bridge._publish_ring()
        command,payload=FrameParser().feed(bridge._write.call_args.args[0])[0]
        self.assertEqual((command,payload[0]),(0x13,1))
        self.gate._command(String(data='raw'))
        self.assertEqual(bridge.active_mode,'raw')
        self.assertIsNone(bridge.latest_ring_time)
        bridge._write.reset_mock()
        bridge._publish_ring()
        bridge._write.assert_not_called()

    def test_gate_state_repetition_does_not_clear_fresh_result(self):
        bridge=self.serial()
        bridge.active_mode='raw'
        bridge._active_mode_callback(String(data='ring'))
        bridge.latest_ring_time=123
        bridge.ring_pending=True
        bridge._active_mode_callback(String(data='ring'))
        self.assertEqual(bridge.latest_ring_time,123)
        self.assertTrue(bridge.ring_pending)


if __name__ == '__main__':
    unittest.main()
