"""Fast Walsh-Hadamard transform (FWHT) helpers.

Provides fwht, is_power_of_two, decompose_into_power_of_two_blocks and the
key-derived application kernels used by the KHT Hadamard diffuser."""


import hashlib
from typing import List, Tuple

import numpy as np

def fwht(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform on the last axis.

    In-place on a copy.  Supports batch dimensions.
    `d` must be a power of 2.  Complexity: O(d log d) per vector.

    Uses the standard in-place butterfly with stride-based pairing
    (paired elements are distance h apart in the original array).
    """
    y = np.asarray(x, dtype=np.float64).copy()
    n = y.shape[-1]

    if n & (n - 1):
        raise ValueError(f"fwht requires power-of-2 dimension, got {n}")

    h = 1
    while h < n:
        y = y.reshape(y.shape[:-1] + (-1, 2, h))
        a = y[..., 0, :].copy()
        b = y[..., 1, :].copy()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.reshape(x.shape[:-1] + (n,))
        h *= 2

    return y / np.sqrt(n)

def _derive_seed(secret_key: bytes, seed_offset: int) -> int:
    """Derive a 32-bit integer seed from secret_key and offset."""
    material = secret_key + f"hadamard_{seed_offset}".encode()
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:4], "big")

def generate_hadamard_signs(secret_key: bytes, seed_offset: int, dim: int) -> np.ndarray:
    """Generate (dim,) array of ±1.0 floats deterministically from key."""
    seed = _derive_seed(secret_key, seed_offset)
    rng = np.random.RandomState(seed)
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=dim)

def generate_hadamard_permutation(secret_key: bytes, seed_offset: int, dim: int) -> np.ndarray:
    """Generate random permutation of arange(dim) deterministically from key."""
    seed = _derive_seed(secret_key, seed_offset)
    rng = np.random.RandomState(seed)
    perm = np.arange(dim)
    rng.shuffle(perm)
    return perm

def apply_keyed_hadamard_rotation(
    x: np.ndarray,
    signs1: np.ndarray,
    signs2: np.ndarray,
    signs3: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """Apply R_K(v) = D3 H P D2 H D1 (v).

    Args:
        x: input vector (d,) or (batch, d)
        signs1, signs2, signs3: (d,) ±1 arrays
        permutation: (d,) int array

    Returns:
        transformed vector, same shape as x
    """
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)

    out = x * signs1.reshape(1, -1)

    out = fwht(out)

    out = out[:, permutation]

    out = out * signs2.reshape(1, -1)

    out = fwht(out)

    out = out * signs3.reshape(1, -1)

    if single:
        out = out.reshape(-1)
    return out.astype(np.float64)

def decompose_into_power_of_two_blocks(dim: int) -> List[Tuple[int, int]]:
    """Decompose dim into power-of-2 blocks.

    Example: 768 → [(0, 512), (512, 256)]
    """
    blocks = []
    remaining = dim
    offset = 0
    while remaining > 0:
        block_size = 1
        while block_size * 2 <= remaining:
            block_size *= 2
        blocks.append((offset, block_size))
        offset += block_size
        remaining -= block_size
    return blocks

def apply_block_hadamard_rotation(
    x: np.ndarray,
    block_param_sets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    blocks: List[Tuple[int, int]],
    global_permutation: np.ndarray | None = None,
) -> np.ndarray:
    """Apply block-wise Hadamard rotation for non-power-of-2 dimensions.

    Args:
        x: input vector (d,) or (batch, d)
        block_param_sets: list of (signs1, signs2, signs3, perm) per block
        blocks: list of (start, length) defining each block
        global_permutation: optional global permutation applied after all blocks

    Returns:
        transformed vector
    """
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)

    out = np.zeros_like(x, dtype=np.float64)

    for (signs1, signs2, signs3, perm), (start, length) in zip(block_param_sets, blocks):
        block_slice = slice(start, start + length)
        block_x = x[:, block_slice]

        block_out = block_x * signs1.reshape(1, -1)
        block_out = fwht(block_out)
        block_out = block_out[:, perm]
        block_out = block_out * signs2.reshape(1, -1)
        block_out = fwht(block_out)
        block_out = block_out * signs3.reshape(1, -1)

        out[:, block_slice] = block_out

    if global_permutation is not None:
        out = out[:, global_permutation]

    if single:
        out = out.reshape(-1)
    return out.astype(np.float64)

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
