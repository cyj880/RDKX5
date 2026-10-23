import unittest
import struct

from logistics_ring_v2.ring_serial_protocol import (
    COMMAND,
    FrameParser,
    MATERIAL_TRACK,
    decode_mode,
    encode_coordinate,
    encode_mode,
    mode_command,
    encode_ring_result,
    ring_yaw,
    encode_identified_ring_result,
)


class RingSerialProtocolTest(unittest.TestCase):
    def test_mode_frames_and_checksum(self):
        self.assertEqual(encode_mode(0), bytes.fromhex("AA 55 10 01 00 11 0D 0A"))
        self.assertEqual(encode_mode(1), bytes.fromhex("AA 55 10 01 01 12 0D 0A"))
        self.assertEqual(encode_mode(MATERIAL_TRACK, 2),
                         bytes.fromhex("AA 55 10 02 03 02 17 0D 0A"))

    def test_modes_decode_to_public_commands(self):
        expected = [(0, None, "raw"), (1, None, "ring"),
                    (2, None, "material_all"), (3, 6, "material_track 6"),
                    (4, None, "turntable"), (5, None, "stacked")]
        for mode, color, command in expected:
            payload = bytes((mode,)) if color is None else bytes((mode, color))
            self.assertEqual(mode_command(mode, color), command)
            self.assertEqual(decode_mode(payload), (mode, color))

    def test_parser_handles_noise_split_sticky_and_bad_frames(self):
        raw = encode_mode(1) + encode_mode(MATERIAL_TRACK, 3)
        parser = FrameParser()
        self.assertEqual(parser.feed(b"noise" + raw[:3]), [])
        self.assertEqual(parser.feed(raw[3:]),
                         [(COMMAND, bytes((1,))), (COMMAND, bytes((3, 3)))])

        bad = bytearray(encode_mode(2))
        bad[-3] ^= 1
        self.assertEqual(parser.feed(bytes(bad) + encode_mode(4)),
                         [(COMMAND, bytes((4,)))])

    def test_coordinate_frame_is_signed_and_incremental(self):
        frame = encode_coordinate(-320, 239)
        parser = FrameParser()
        self.assertEqual(parser.feed(frame[:4]), [])
        command, payload = parser.feed(frame[4:])[0]
        self.assertEqual(command, 0x11)
        self.assertEqual(payload, bytes.fromhex("C0 FE EF 00"))

    def test_bytewise_mode_frame_including_split_header(self):
        parser = FrameParser()
        frames = []
        for byte in b"noise" + encode_mode(1):
            frames.extend(parser.feed(bytes((byte,))))
        self.assertEqual(frames, [(COMMAND, b"\x01")])

    def test_ring_counts_sorting_yaw_and_frame_checksum(self):
        for count in range(4):
            sites = [{"x": 100 + 150 * i, "y": 180 + 15 * i} for i in range(count)]
            frame = encode_ring_result(reversed(sites))
            self.assertEqual(len(frame), 11 + 4 * count)
            command, payload = FrameParser().feed(frame)[0]
            self.assertEqual(command, 0x12)
            n, flags, yaw = struct.unpack("<BBh", payload[:4])
            self.assertEqual((n, flags), (count, int(count >= 2)))
            self.assertEqual(yaw, 571 if count >= 2 else 0)
            for i, site in enumerate(sites):
                self.assertEqual(struct.unpack_from("<HH", payload, 4 + 4 * i),
                                 (site['x'], site['y']))

    def test_ring_negative_horizontal_and_vertical_yaw(self):
        for points, expected in [([(100, 240), (200, 230)], -571),
                                 ([(100, 240), (200, 240)], 0),
                                 ([(100, 240), (100, 340)], 9000)]:
            frame = encode_ring_result([dict(x=x, y=y) for x, y in points])
            self.assertEqual(struct.unpack_from('<h', frame, 6)[0], expected)
        self.assertIsNone(ring_yaw([dict(x=1, y=2)]))

    def test_ring_rejects_bad_coordinates_count_and_duplicates(self):
        invalid = [[dict(x=640, y=0)], [dict(x=0, y=480)], [dict(x=-1, y=0)],
                   [dict(x=float('nan'), y=0)], [dict(x=float('inf'), y=0)],
                   [dict(x=1, y=2)] * 2,
                   [dict(x=i, y=2) for i in range(4)]]
        for sites in invalid:
            with self.assertRaises(ValueError):
                encode_ring_result(sites)

    def test_identified_ring_wire_order_and_unknown_ids(self):
        sites = [dict(id=3, x=400, y=240), dict(id=2, x=250, y=240), dict(id=1, x=100, y=240)]
        frame = encode_identified_ring_result(sites)
        command, payload = FrameParser().feed(frame)[0]
        self.assertEqual(command, 0x13)
        self.assertEqual(len(payload), 19)
        self.assertEqual([payload[4 + 5 * i] for i in range(3)], [1, 2, 3])
        single = FrameParser().feed(encode_identified_ring_result([dict(id=3, x=100, y=240)]))[0][1]
        self.assertEqual(single[4], 3)
        unknown = FrameParser().feed(encode_identified_ring_result([dict(x=100, y=240)]))[0][1]
        self.assertEqual(unknown[4], 0)
        for invalid in [[dict(id=4,x=1,y=2)], [dict(id=1,x=1,y=2),dict(id=1,x=3,y=4)]]:
            with self.assertRaises(ValueError):
                encode_identified_ring_result(invalid)
        # Rounding ties must keep IDs attached to their quantized centers.
        tied = [dict(id=1,x=100.1,y=250),dict(id=2,x=100.2,y=200)]
        payload = FrameParser().feed(encode_identified_ring_result(tied))[0][1]
        self.assertEqual(struct.unpack_from('<BHH', payload, 4), (2,100,200))


if __name__ == "__main__":
    unittest.main()
