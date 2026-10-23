"""AA55 serial framing used by the STM32/RDK vision bridge."""

from __future__ import annotations

import math
import struct
from typing import Iterable, List, Optional, Tuple


HEAD = b"\xAA\x55"
TAIL = b"\x0D\x0A"
COMMAND = 0x10
COORDINATE = 0x11
RING_RESULT = 0x12
RING_IDENTIFIED_RESULT = 0x13

RAW = 0
RING = 1
MATERIAL_ALL = 2
MATERIAL_TRACK = 3
TURNTABLE = 4
STACKED = 5

# Compatibility names used by the original ring-only bridge.
START = RING
STOP = RAW

MODE_NAMES = {
    RAW: "raw",
    RING: "ring",
    MATERIAL_ALL: "material_all",
    MATERIAL_TRACK: "material_track",
    TURNTABLE: "turntable",
    STACKED: "stacked",
}


def _checksum(command: int, payload: bytes) -> int:
    return (command + len(payload) + sum(payload)) & 0xFF


def encode_frame(command: int, payload: bytes = b"") -> bytes:
    if not 0 <= command <= 0xFF:
        raise ValueError("command must fit in one byte")
    if len(payload) > 0xFF:
        raise ValueError("payload is too long")
    return (HEAD + bytes((command, len(payload))) + payload
            + bytes((_checksum(command, payload),)) + TAIL)


def encode_mode(mode: int, color: Optional[int] = None) -> bytes:
    """Encode one mode-selection frame from STM32 to RDK."""
    if mode not in MODE_NAMES:
        raise ValueError("unknown mode: %s" % mode)
    if mode == MATERIAL_TRACK:
        if color is None or not 1 <= int(color) <= 6:
            raise ValueError("material_track requires color 1..6")
        payload = bytes((mode, int(color)))
    elif color is not None:
        raise ValueError("color is only valid for material_track")
    else:
        payload = bytes((mode,))
    return encode_frame(COMMAND, payload)


def decode_mode(payload: bytes) -> Optional[Tuple[int, Optional[int]]]:
    if not payload or payload[0] not in MODE_NAMES:
        return None
    mode = payload[0]
    if mode == MATERIAL_TRACK:
        if len(payload) != 2 or not 1 <= payload[1] <= 6:
            return None
        return mode, payload[1]
    if len(payload) != 1:
        return None
    return mode, None


def mode_command(mode: int, color: Optional[int] = None) -> str:
    decoded = decode_mode(bytes((mode,)) if color is None else bytes((mode, color)))
    if decoded is None:
        raise ValueError("invalid mode payload")
    selected, selected_color = decoded
    command = MODE_NAMES[selected]
    if selected == MATERIAL_TRACK:
        command += " %d" % selected_color
    return command


def encode_coordinate(x: int, y: int) -> bytes:
    """Encode signed image-center offsets sent from RDK to STM32."""
    payload = struct.pack("<hh", int(x), int(y))
    return encode_frame(COORDINATE, payload)


def decode_coordinate_payload(payload: bytes) -> Tuple[int, int]:
    if len(payload) != 4:
        raise ValueError("coordinate payload must contain four bytes")
    return struct.unpack("<hh", payload)


def ring_yaw(sites: Iterable[dict]) -> Optional[float]:
    """Image-line angle: right/down is positive; no heading from one point."""
    points = sorted((float(s["x"]), float(s["y"])) for s in sites)
    if len(points) < 2 or points[0] == points[-1]:
        return None
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    return round(math.degrees(math.atan2(dy, dx)), 2)


def encode_ring_result(sites: Iterable[dict]) -> bytes:
    """One coherent 0..3-ring snapshot, absolute 640x480 pixels, little endian."""
    points = []
    for site in sites:
        x, y = float(site["x"]), float(site["y"])
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < 640 and 0 <= y < 480):
            raise ValueError("ring coordinate outside 640x480")
        points.append((min(639, int(round(x))), min(479, int(round(y)))))
    if len(points) > 3 or len(set(points)) != len(points):
        raise ValueError("expected at most three distinct ring centers")
    points.sort()
    yaw = ring_yaw([{"x": x, "y": y} for x, y in points])
    payload = struct.pack("<BBh", len(points), int(yaw is not None),
                          0 if yaw is None else int(round(yaw * 100)))
    payload += b"".join(struct.pack("<HH", x, y) for x, y in points)
    return encode_frame(RING_RESULT, payload)


def encode_identified_ring_result(sites: Iterable[dict]) -> bytes:
    """Fixed board IDs (0=unknown); never renumber a partial observation."""
    sites = list(sites)
    legacy = encode_ring_result(sites)
    ordered = sorted(sites, key=lambda s: (min(639, int(round(float(s['x'])))),
                                           min(479, int(round(float(s['y']))))))
    payload = legacy[4:8]
    seen = set()
    for index, site in enumerate(ordered):
        identity = site.get('id', 0)
        if type(identity) is not int or not 0 <= identity <= 3:
            raise ValueError('ring id must be 0..3')
        if identity and identity in seen:
            raise ValueError('duplicate fixed ring id')
        seen.add(identity)
        payload += bytes((identity,)) + legacy[8 + 4 * index:12 + 4 * index]
    return encode_frame(RING_IDENTIFIED_RESULT, payload)


def select_center_offset(
    groups: Iterable[dict], image_width: int = 640, image_height: int = 480
) -> Optional[Tuple[int, int]]:
    """Return the leftmost valid group as an image-center offset."""
    valid = []
    for group in groups:
        try:
            x = float(group["x"])
            y = float(group["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= x < image_width and 0 <= y < image_height:
            valid.append((x, y))
    if not valid:
        return None
    x, y = min(valid, key=lambda point: point[0])
    return int(round(x - image_width / 2.0)), int(round(y - image_height / 2.0))


class FrameParser:
    """Incremental parser that tolerates noise, splits and sticky frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> List[Tuple[int, bytes]]:
        self._buffer.extend(data)
        frames: List[Tuple[int, bytes]] = []
        while True:
            start = self._buffer.find(HEAD)
            if start < 0:
                trailing_head = self._buffer[-1:] == HEAD[:1]
                self._buffer.clear()
                if trailing_head:
                    self._buffer.extend(HEAD[:1])
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 4:
                break
            length = self._buffer[3]
            if length > 64:
                del self._buffer[0]
                continue
            total = 2 + 1 + 1 + length + 1 + len(TAIL)
            if len(self._buffer) < total:
                break
            frame = bytes(self._buffer[:total])
            command = frame[2]
            payload = frame[4:4 + length]
            if frame[-2:] != TAIL:
                del self._buffer[0]
                continue
            if frame[4 + length] != _checksum(command, payload):
                del self._buffer[0]
                continue
            del self._buffer[:total]
            frames.append((command, payload))
        return frames

