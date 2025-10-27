#!/usr/bin/env python3
import unittest
from scripts.encoding import clip_vector, encode_vector_to_int, decode_int_to_float, l2_norm


class TestEncoding(unittest.TestCase):
    def test_clip_vector(self):
        v = [3.0, 4.0]  # norm 5
        C = 2.5
        vc = clip_vector(v, C)
        self.assertAlmostEqual(l2_norm(vc), C, places=6)

    def test_encode_decode_roundtrip(self):
        v = [0.001, -0.0025, 0.5, -0.25]
        S = 2**16
        q = 2**61 - 1
        enc = encode_vector_to_int(v, S=S, q=q)
        dec = decode_int_to_float(enc, S=S, q=q)
        for a, b in zip(v, dec):
            self.assertLessEqual(abs(a - b), 0.5 / S)

    def test_no_wrap_for_small_sum(self):
        # Sum of vectors should not wrap if magnitudes are small
        S = 2**16
        q = 2**61 - 1
        v1 = [0.1] * 10
        v2 = [-0.05] * 10
        e1 = encode_vector_to_int(v1, S=S, q=q)
        e2 = encode_vector_to_int(v2, S=S, q=q)
        summed = [(a + b) % q for a, b in zip(e1, e2)]
        dec = decode_int_to_float(summed, S=S, q=q)
        for x in dec:
            self.assertLessEqual(abs(x - 0.05), 1.0 / S)


if __name__ == '__main__':
    unittest.main()
