#!/usr/bin/env python3
import unittest
import random

class TestMaskCancellationToy(unittest.TestCase):
    def test_three_client_mask_cancel(self):
        rng = random.Random(123)
        q = 2**61 - 1
        dim = 50
        # Original integer vectors (encoded gradients)
        v1 = [rng.randrange(0, 1000) for _ in range(dim)]
        v2 = [rng.randrange(0, 1000) for _ in range(dim)]
        v3 = [rng.randrange(0, 1000) for _ in range(dim)]
        # Pairwise masks r_ij where r_ji = -r_ij (mod q)
        r12 = [rng.randrange(0, q) for _ in range(dim)]
        r23 = [rng.randrange(0, q) for _ in range(dim)]
        r13 = [rng.randrange(0, q) for _ in range(dim)]
        def neg(u):
            return [(q - x) % q for x in u]
        r21 = neg(r12)
        r32 = neg(r23)
        r31 = neg(r13)
        # Masked uploads
        m1 = [(v1[i] + r12[i] + r13[i]) % q for i in range(dim)]
        m2 = [(v2[i] + r21[i] + r23[i]) % q for i in range(dim)]
        m3 = [(v3[i] + r31[i] + r32[i]) % q for i in range(dim)]
        # Server sum
        summed = [(m1[i] + m2[i] + m3[i]) % q for i in range(dim)]
        # Should equal (v1+v2+v3) mod q
        expected = [(v1[i] + v2[i] + v3[i]) % q for i in range(dim)]
        self.assertEqual(summed, expected)

if __name__ == '__main__':
    unittest.main()
