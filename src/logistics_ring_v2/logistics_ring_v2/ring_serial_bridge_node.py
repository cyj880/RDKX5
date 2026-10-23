"""Own the STM32 serial port and bridge mode/control frames to ROS."""

from __future__ import annotations

from collections import deque
import json
import struct
import time
from typing import Optional, Tuple

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String

try:
    import serial
except ImportError:  # pragma: no cover - reported on RDK at runtime
    serial = None

from .ring_serial_protocol import (
    COMMAND,
    FrameParser,
    decode_mode,
    encode_identified_ring_result,
    mode_command,
)


LEGACY_HEAD = 0x66
LEGACY_CMD_CENTER = 0x01
LEGACY_TAIL = 0x77


class RingSerialBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("ring_serial_bridge")
        self.declare_parameter("serial_port", "/dev/ttyS1")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("mode_topic", "/vision/mode_command")
        self.declare_parameter("ring_result_topic", "/vision/ring2/sites")
        self.declare_parameter("turntable_result_topic", "/vision/turntable/results")
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("ring_publish_hz", 30.0)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)

        self.serial_port = str(self.get_parameter("serial_port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.publish_period = 1.0 / max(1.0, float(self.get_parameter("publish_hz").value))
        self.ring_publish_period = 1.0 / max(1.0, min(30.0, float(self.get_parameter("ring_publish_hz").value)))
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.serial_handle = None
        self.last_open_error = ""
        self.parser = FrameParser()
        self.active_mode = "raw"
        self.tx_reported = False
        self.latest_ring_frame = encode_identified_ring_result([])
        self.ring_pending = False
        self.latest_ring_time = None
        self.last_ring_source_ns = 0
        self.ring_mode_since_ns = 0
        self.ring_tx_ages = deque(maxlen=512)
        self.ring_stats_time = time.monotonic()
        self.last_serial_rx = None
        self.latest_turntable_center: Optional[Tuple[int, int]] = None

        mode_topic = str(self.get_parameter("mode_topic").value)
        self.mode_publisher = self.create_publisher(String, mode_topic, 10)
        retained = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/vision/active_mode', self._active_mode_callback, retained)
        self.create_subscription(
            String, str(self.get_parameter("ring_result_topic").value),
            self._ring_result_callback, 1)
        self.create_subscription(
            String, str(self.get_parameter("turntable_result_topic").value),
            self._turntable_result_callback, 10)
        self.poll_timer = self.create_timer(0.02, self._poll_serial)
        self.publish_timer = self.create_timer(self.publish_period, self._publish_other_modes)
        self.ring_publish_timer = self.create_timer(self.ring_publish_period, self._publish_ring)
        self.get_logger().info(
            "串口桥接已启动: %s @ %d，模式话题 %s" %
            (self.serial_port, self.baudrate, mode_topic))

    def _open_serial(self) -> None:
        if self.serial_handle is not None:
            return
        if serial is None:
            if not self.last_open_error:
                self.last_open_error = "python3-serial 未安装，串口桥接已禁用"
                self.get_logger().error(self.last_open_error)
            return
        try:
            self.serial_handle = serial.Serial(
                self.serial_port, self.baudrate, timeout=0, write_timeout=0)
            self.last_open_error = ""
            self.get_logger().info("已连接 STM32 串口 %s。" % self.serial_port)
        except serial.SerialException as error:
            text = str(error)
            if text != self.last_open_error:
                self.get_logger().warn("打开 STM32 串口失败: %s" % text)
                self.last_open_error = text

    def _close_serial(self) -> None:
        if self.serial_handle is not None:
            try:
                self.serial_handle.close()
            except (serial.SerialException, OSError):
                pass
        self.serial_handle = None
        self.parser = FrameParser()
        self.last_serial_rx = None

    def _write(self, frame: bytes) -> None:
        self._open_serial()
        if self.serial_handle is None:
            return
        try:
            written = self.serial_handle.write(frame)
            if written != len(frame):
                raise serial.SerialTimeoutException(
                    "partial write: %d/%d" % (written, len(frame)))
            if not self.tx_reported:
                self.get_logger().info("首个坐标帧已写入串口 %s: %s" %
                                       (self.serial_port, frame.hex(" ").upper()))
                self.tx_reported = True
        except (serial.SerialException, OSError) as error:
            self.get_logger().warn("STM32 串口发送失败: %s" % error)
            self._close_serial()

    def _poll_serial(self) -> None:
        self._open_serial()
        if self.serial_handle is None:
            return
        try:
            waiting = self.serial_handle.in_waiting
            data = self.serial_handle.read(waiting or 1)
        except (serial.SerialException, OSError) as error:
            self.get_logger().warn("STM32 串口读取失败: %s" % error)
            self._close_serial()
            return
        if data:
            now = time.monotonic()
            if self.last_serial_rx is not None and now - self.last_serial_rx > 0.05:
                self.parser = FrameParser()
            self.last_serial_rx = now
        for command, payload in self.parser.feed(data):
            if command != COMMAND:
                continue
            decoded = decode_mode(payload)
            if decoded is None:
                continue
            mode, color = decoded
            self.active_mode = mode_command(mode, color)
            self.tx_reported = False
            self.latest_ring_frame = encode_identified_ring_result([])
            self.ring_pending = False
            self.latest_ring_time = None
            self.last_ring_source_ns = 0
            self.ring_mode_since_ns = int(time.monotonic() * 1e9)
            self.latest_turntable_center = None
            message = String(data=self.active_mode)
            self.mode_publisher.publish(message)
            self.get_logger().info("收到 STM32 模式命令: %s" % self.active_mode)

    def _ring_result_callback(self, message: String) -> None:
        if self.active_mode != "ring":
            return
        try:
            result = json.loads(message.data)
            sites = result["sites"]
            if not isinstance(sites, list) or result["count"] != len(sites):
                raise ValueError("invalid ring result count")
            source_ns = result["ingress_monotonic_ns"]
            sequence = result["frame_sequence"]
            now_ns = int(time.monotonic() * 1e9)
            if (type(source_ns) is not int or source_ns <= 0 or
                    type(sequence) is not int or sequence <= 0 or
                    source_ns > now_ns or now_ns - source_ns > 300_000_000 or
                    source_ns < self.ring_mode_since_ns):
                raise ValueError("invalid or expired source frame time")
            if source_ns <= self.last_ring_source_ns:
                return  # Never re-arm a cached/duplicate/out-of-order observation.
            self.latest_ring_frame = encode_identified_ring_result(sites)
            self.latest_ring_time = source_ns / 1e9
            self.last_ring_source_ns = source_ns
            self.ring_pending = True
        except (json.JSONDecodeError, AttributeError, TypeError, KeyError, ValueError, OverflowError):
            self.latest_ring_frame = encode_identified_ring_result([])
            self.latest_ring_time = None
            self.ring_pending = False

    def _active_mode_callback(self, message: String) -> None:
        mode = message.data.strip()
        if mode not in ('raw', 'ring', 'turntable', 'stacked', 'material_all', 'material_track'):
            return
        if self.active_mode == mode:
            return
        # Follow the gate's executed mode, including keyboard requests. A
        # received serial request alone is not proof a detector is running.
        self.active_mode = mode
        self.tx_reported = False
        self.latest_ring_frame = encode_identified_ring_result([])
        self.latest_ring_time = None
        self.ring_pending = False
        self.last_ring_source_ns = 0
        self.ring_mode_since_ns = int(time.monotonic() * 1e9)
        self.latest_turntable_center = None
        self.get_logger().info('门控确认执行模式: %s' % mode)

    def _turntable_result_callback(self, message: String) -> None:
        if self.active_mode != "turntable":
            return
        try:
            result = json.loads(message.data)
            if result.get("detected") and result.get("center_u") is not None:
                self.latest_turntable_center = (
                    int(result["center_u"]), int(result["center_v"]))
            else:
                self.latest_turntable_center = None
        except (json.JSONDecodeError, AttributeError, TypeError, KeyError, ValueError):
            self.latest_turntable_center = None

    @staticmethod
    def _legacy_center_frame(center: Tuple[int, int]) -> bytes:
        x, y = center
        payload = struct.pack(">hh", int(x), int(y))
        return bytes((LEGACY_HEAD, LEGACY_CMD_CENTER)) + payload + bytes((LEGACY_TAIL,))

    def _publish_latest(self) -> None:
        if self.active_mode == "ring":
            fresh = (self.latest_ring_time is not None and
                     time.monotonic() - self.latest_ring_time <= 0.3)
            if not fresh:
                self._write(encode_identified_ring_result([]))
            elif self.ring_pending:
                self._write(self.latest_ring_frame)
                self.ring_tx_ages.append((time.monotonic() - self.latest_ring_time) * 1000.0)
                self.ring_pending = False
            now = time.monotonic()
            if now - self.ring_stats_time >= 5.0:
                ages = sorted(self.ring_tx_ages)
                if ages:
                    self.get_logger().info("RING TX source_age_p95=%.1fms samples=%d" %
                                           (ages[int((len(ages)-1)*0.95)], len(ages)))
                self.ring_tx_ages.clear()
                self.ring_stats_time = now
        elif self.active_mode == "turntable" and self.latest_turntable_center is not None:
            self._write(self._legacy_center_frame(self.latest_turntable_center))

    def _publish_ring(self) -> None:
        if self.active_mode == "ring":
            self._publish_latest()

    def _publish_other_modes(self) -> None:
        if self.active_mode != "ring":
            self._publish_latest()

    def destroy_node(self) -> bool:
        self._close_serial()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RingSerialBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
