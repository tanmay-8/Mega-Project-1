#!/usr/bin/env python3
"""
Bonawitz-style mask orchestration helpers (simplified for prototype):
- Key agreement: peers exchange ephemeral X25519 public keys
- Derive pairwise seeds via HKDF with (round_id, i, j) context
- Expand seeds into mask vectors; apply sign convention to ensure cancellation

This module does not implement full dropout recovery. For prototype,
we assume all declared clients submit in a round. Basic extensions can
add seed sharing to the server for reconstruction.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from .ecdh import KeyPair, generate_keypair, shared_secret
from .hkdf_prg import hkdf_expand, prg_mask_vector


def derive_pairwise_seed(my_sk, peer_pub_bytes: bytes, round_id: int, i: str, j: str) -> bytes:
    ss = shared_secret(my_sk, peer_pub_bytes)
    info = f"round:{round_id}|i:{i}|j:{j}".encode()
    salt = b"secagg-hkdf"
    return hkdf_expand(ss, salt=salt, info=info, length=32)


def make_mask_vector(my_kp: KeyPair, peers: Dict[str, bytes], round_id: int, dim: int, q: int, my_id: str) -> List[int]:
    """
    Compute total mask for this client: sum over j!=i of sign(i,j) * r_ij
    where r_ij = PRG(seed_ij) and sign(i,j) = +1 if i<j else -1 (string order).
    """
    acc = [0] * dim
    for pid, ppub in peers.items():
        if pid == my_id:
            continue
        if my_id < pid:
            seed = derive_pairwise_seed(my_kp.sk, ppub, round_id, my_id, pid)
            r = prg_mask_vector(seed, dim, q)
            for k in range(dim):
                acc[k] = (acc[k] + r[k]) % q
        else:
            seed = derive_pairwise_seed(my_kp.sk, ppub, round_id, pid, my_id)
            r = prg_mask_vector(seed, dim, q)
            for k in range(dim):
                acc[k] = (acc[k] - r[k]) % q
    return acc
