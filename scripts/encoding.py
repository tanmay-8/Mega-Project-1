#!/usr/bin/env python3
"""
Encoding utilities for Secure Aggregation compatibility.
All operations are pure Python; vectors are Python lists of floats/ints.

- clip_vector(vec, C): L2-norm clip to radius C
- encode_vector_to_int(vec, S, q): fixed-point encoding with scale S into modulo q ring
- decode_int_to_float(int_vec, S, q): inverse decoding to floats in [-0.5, 0.5] wrapping assumptions

Notes:
- Choose q as a large prime (e.g., 2**61 - 1) and S as a power of 2 (e.g., 2**16).
- Ensure that the sum of magnitudes stays below q/(2*S) to avoid wrap-around.
"""
from __future__ import annotations
import math
from typing import List


def l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def clip_vector(vec: List[float], C: float) -> List[float]:
    nrm = l2_norm(vec)
    if nrm <= C or nrm == 0:
        return list(vec)
    scale = C / nrm
    return [x * scale for x in vec]


def mod_q(x: int, q: int) -> int:
    r = x % q
    if r < 0:
        r += q
    return r


def encode_vector_to_int(vec: List[float], S: int = 2**16, q: int = 2**61 - 1) -> List[int]:
    # fixed-point: round(S * x) mod q
    return [mod_q(int(round(S * v)), q) for v in vec]


def center_lift(x: int, q: int) -> int:
    # Map from [0, q-1] to centered range (-(q//2), +(q//2)]
    if x > q // 2:
        return x - q
    return x


def decode_int_to_float(int_vec: List[int], S: int = 2**16, q: int = 2**61 - 1) -> List[float]:
    return [center_lift(v, q) / float(S) for v in int_vec]
