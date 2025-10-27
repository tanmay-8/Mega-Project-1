#!/usr/bin/env python3
import unittest

from server.secure_aggregation.ecdh import generate_keypair
from server.secure_aggregation.bonawitz import make_mask_vector

class TestECDHMask(unittest.TestCase):
    def test_masks_cancel(self):
        q = 2**61 - 1
        dim = 32
        round_id = 1
        # 3 clients
        kp = {
            'A': generate_keypair(),
            'B': generate_keypair(),
            'C': generate_keypair(),
        }
        pub = {k: v.serialize_public() for k, v in kp.items()}
        # Peers map includes all public keys
        peers = pub
        mA = make_mask_vector(kp['A'], peers, round_id, dim, q, 'A')
        mB = make_mask_vector(kp['B'], peers, round_id, dim, q, 'B')
        mC = make_mask_vector(kp['C'], peers, round_id, dim, q, 'C')
        summed = [(mA[i] + mB[i] + mC[i]) % q for i in range(dim)]
        self.assertTrue(all(x == 0 for x in summed))

if __name__ == '__main__':
    unittest.main()
