#!/usr/bin/env python3
"""
Simplified dropout simulation: demonstrate that if the server can reconstruct
pairwise masks involving a dropped client using backup shares from remaining
clients, it can remove the dropped client's masks from the aggregate.

This is a prototype-only simplification for Sem-1.
"""
import unittest

from server.secure_aggregation.ecdh import generate_keypair
from server.secure_aggregation.bonawitz import make_mask_vector, derive_pairwise_seed
from server.secure_aggregation.hkdf_prg import prg_mask_vector

class TestDropoutSim(unittest.TestCase):
    def test_single_dropout_recovery(self):
        q = 2**61 - 1
        dim = 16
        round_id = 7
        # Clients A, B, C; C drops
        kpA = generate_keypair()
        kpB = generate_keypair()
        kpC = generate_keypair()
        pubA, pubB, pubC = kpA.serialize_public(), kpB.serialize_public(), kpC.serialize_public()
        peers = {'A': pubA, 'B': pubB, 'C': pubC}

        mA = make_mask_vector(kpA, peers, round_id, dim, q, 'A')
        mB = make_mask_vector(kpB, peers, round_id, dim, q, 'B')
        mC = make_mask_vector(kpC, peers, round_id, dim, q, 'C')

        # Suppose C drops after masking. Server receives mA and mB only, but their sum includes
        # +/- masks involving C. With backup seeds, server reconstructs masks involving C and cancels them.
        summed = [(mA[i] + mB[i]) % q for i in range(dim)]

        # Reconstruct A-C and B-C masks using peers' backups of seeds with C
        seed_AC = derive_pairwise_seed(kpA.sk, pubC, round_id, 'A', 'C')
        r_AC = prg_mask_vector(seed_AC, dim, q)  # A added +r_AC (since 'A'<'C')
        seed_BC = derive_pairwise_seed(kpB.sk, pubC, round_id, 'B', 'C')
        r_BC = prg_mask_vector(seed_BC, dim, q)  # B added +r_BC (since 'B'<'C')

        # Server subtracts these from the sum to remove C-related masks
        recovered = [(summed[i] - r_AC[i] - r_BC[i]) % q for i in range(dim)]

        # If only A and B remain, expected mask sum equals A-B pair only, which cancels in pair
        # when aggregating mA and mB without the C-related components.
        # Therefore recovered should be zero vector.
        self.assertTrue(all(x % q == 0 for x in recovered))

if __name__ == '__main__':
    unittest.main()
