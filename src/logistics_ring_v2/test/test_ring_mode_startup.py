"""Exercise actual UART parser -> mode gate startup -> ring worker activation."""
import unittest
from unittest.mock import Mock

import rclpy
from std_msgs.msg import String

from logistics_ring_v2.mode_gate_node import ModeGateNode
from logistics_ring_v2.ring_pipeline_node import RingPipelineNode
from logistics_ring_v2.ring_serial_bridge_node import RingSerialBridgeNode
from logistics_ring_v2.ring_serial_protocol import encode_mode, FrameParser


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
        self.gate.timer.cancel.assert_called_once()
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

    def test_missing_ring_or_other_detector_keeps_pending_command(self):
        for missing in ('ring','turntable','material_all','material_track','stacked'):
            with self.subTest(missing=missing):
                self.gate.ready = False
                for pub in self.gate.gate_publishers.values():
                    pub.get_subscription_count.return_value = 1
                self.gate.gate_publishers[missing].get_subscription_count.return_value = 0
                self.gate._command(String(data='ring'))
                self.gate._startup()
                self.assertFalse(self.gate.ready)
                self.assertEqual(self.gate.pending_command, 'ring')
        self.gate.gate_publishers['stacked'].get_subscription_count.return_value = 1
        self.gate._startup()
        self.assertTrue(self.ring.running)


if __name__ == '__main__':
    unittest.main()
