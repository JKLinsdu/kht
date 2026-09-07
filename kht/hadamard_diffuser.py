"""The KHT Hadamard diffuser: B_K = D3 P2 H D2 P1 H D1.

A keyed cascade of FWHT stages interleaved with key-derived sign diagonals (D)
and permutations (P), giving O(d log d) structured orthogonal mixing.
Provides HadamardDiffuserParams, derive_hadamard_diffuser_params, apply_hadamard_diffuser."""


import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np

from .hadamard import fwht, decompose_into_power_of_two_blocks, is_power_of_two

def derive_seed(
    secret_key: bytes | str,
    *,
    namespace: str = "ksom",
    stage: int = 0,
    purpose: str = "",
    block_id: int = 0,
) -> int:
    """Deterministic 32-bit seed with domain-separated purpose labels.

    Uses SHA-256; all arguments are embedded in a fixed-format material string
    to prevent cross-purpose seed collisions.
    """
    if isinstance(secret_key, str):
        try:
            secret_key = bytes.fromhex(secret_key)
        except ValueError:
            secret_key = secret_key.encode("utf-8")

    material = (
        f"{namespace}|s={stage}|b={block_id}|{purpose}|".encode("utf-8")
        + secret_key
    )
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:4], "big")

def make_rng(
    secret_key: bytes | str,
    *,
    namespace: str = "ksom",
    stage: int = 0,
    purpose: str = "",
    block_id: int = 0,
) -> np.random.Generator:
    """Create a seeded numpy Generator."""
    return np.random.default_rng(
        derive_seed(secret_key, namespace=namespace, stage=stage,
                    purpose=purpose, block_id=block_id)
    )

@dataclass
class HadamardDiffuserParams:
    """Parameters for one Hadamard diffuser transform instance."""
    dim: int
    variant: str  # dhpdhd | block_dhpdhd
    stage: int = 0

    signs1: np.ndarray | None = None
    signs2: np.ndarray | None = None
    signs3: np.ndarray | None = None

    perm1: np.ndarray | None = None
    perm2: np.ndarray | None = None

    blocks: List[Tuple[int, int]] = field(default_factory=list)
    block_params: List[Dict] = field(default_factory=list)
    global_perm: np.ndarray | None = None

def derive_hadamard_diffuser_params(
    secret_key: bytes | str,
    dim: int,
    stage: int = 0,
    variant: str = "dhpdhd",
    block_size: int | None = None,
) -> HadamardDiffuserParams:
    """Derive Hadamard diffuser parameters from key material."""
    params = HadamardDiffuserParams(dim=dim, variant=variant, stage=stage)

    if variant == "block_dhpdhd" or not is_power_of_two(dim):
        base_v = variant.replace("block_", "")
        params.variant = "block_" + base_v
        blocks = decompose_into_power_of_two_blocks(dim)
        params.blocks = blocks
        has_l2 = True  # dhpdhd: two Hadamard layers
        for blk_idx, (start, blk_dim) in enumerate(blocks):
            bp = {
                "signs1": _gen_signs(secret_key, blk_dim, stage, "sign1", blk_idx),
                "signs2": _gen_signs(secret_key, blk_dim, stage, "sign2", blk_idx) if has_l2 else None,
                "signs3": _gen_signs(secret_key, blk_dim, stage, "sign3", blk_idx) if base_v == "dhpdhd" else None,
                "perm1": _gen_perm(secret_key, blk_dim, stage, "perm1", blk_idx),
                "perm2": _gen_perm(secret_key, blk_dim, stage, "perm2", blk_idx) if has_l2 else None,
            }
            params.block_params.append(bp)
        params.global_perm = _gen_perm(secret_key, dim, stage, "global_perm")
    else:
        params.signs1 = _gen_signs(secret_key, dim, stage, "sign1")
        params.perm1 = _gen_perm(secret_key, dim, stage, "perm1")
        params.signs2 = _gen_signs(secret_key, dim, stage, "sign2")
        params.signs3 = _gen_signs(secret_key, dim, stage, "sign3")
        params.perm2 = _gen_perm(secret_key, dim, stage, "perm2")

    return params

def apply_hadamard_diffuser(x: np.ndarray, params: HadamardDiffuserParams) -> np.ndarray:
    """Apply Hadamard diffuser forward transform. Supports (d,) and (batch, d)."""
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)

    out = x.astype(np.float64)
    v = params.variant

    if v.startswith("block_"):
        result = np.zeros_like(out)
        for bp, (start, blk_dim) in zip(params.block_params, params.blocks):
            sl = slice(start, start + blk_dim)
            blk = out[:, sl]
            blk = blk * bp["signs1"].reshape(1, -1)
            blk = fwht(blk)
            blk = blk[:, bp["perm1"]]
            if bp.get("signs2") is not None and bp.get("perm2") is not None:
                blk = blk * bp["signs2"].reshape(1, -1)
                blk = fwht(blk)
                blk = blk[:, bp["perm2"]]
            if bp.get("signs3") is not None:
                blk = blk * bp["signs3"].reshape(1, -1)
            result[:, sl] = blk
        out = result
        if params.global_perm is not None:
            out = out[:, params.global_perm]
    else:
        if params.signs1 is not None:
            out = out * params.signs1.reshape(1, -1)
        out = fwht(out)
        if params.perm1 is not None:
            out = out[:, params.perm1]
        if v == "dhpdhd":
            if params.signs2 is not None:
                out = out * params.signs2.reshape(1, -1)
            out = fwht(out)
            if params.perm2 is not None:
                out = out[:, params.perm2]
        if v == "dhpdhd" and params.signs3 is not None:
            out = out * params.signs3.reshape(1, -1)

    if single:
        out = out.reshape(-1)
    return out

def inverse_hadamard_diffuser(y: np.ndarray, params: HadamardDiffuserParams) -> np.ndarray:
    """Exact inverse of apply_hadamard_diffuser."""
    single = y.ndim == 1
    if single:
        y = y.reshape(1, -1)

    out = y.astype(np.float64)
    v = params.variant

    if v.startswith("block_"):
        if params.global_perm is not None:
            out = out[:, np.argsort(params.global_perm)]
        result = np.zeros_like(out)
        for bp, (start, blk_dim) in zip(params.block_params, params.blocks):
            sl = slice(start, start + blk_dim)
            blk = out[:, sl]
            if bp.get("signs3") is not None:
                blk = blk * bp["signs3"].reshape(1, -1)
            if bp.get("perm2") is not None:
                blk = blk[:, np.argsort(bp["perm2"])]
            if bp.get("signs2") is not None:
                blk = fwht(blk)
                blk = blk * bp["signs2"].reshape(1, -1)
            blk = blk[:, np.argsort(bp["perm1"])]
            blk = fwht(blk)
            blk = blk * bp["signs1"].reshape(1, -1)
            result[:, sl] = blk
        out = result
    else:
        if v == "dhpdhd" and params.signs3 is not None:
            out = out * params.signs3.reshape(1, -1)
        if v == "dhpdhd":
            if params.perm2 is not None:
                out = out[:, np.argsort(params.perm2)]
            out = fwht(out)
            if params.signs2 is not None:
                out = out * params.signs2.reshape(1, -1)
        if params.perm1 is not None:
            out = out[:, np.argsort(params.perm1)]
        out = fwht(out)
        if params.signs1 is not None:
            out = out * params.signs1.reshape(1, -1)

    if single:
        out = out.reshape(-1)
    return out

def hadamard_diffuser_explicit_matrix(params: HadamardDiffuserParams, dim: int = 0) -> np.ndarray:
    """Build explicit d×d matrix for diagnostics (d ≤ 1024 only)."""
    d = dim or params.dim
    if d > 1024:
        raise ValueError("Explicit matrix only supported for d ≤ 1024")
    S = np.eye(d)
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        S[:, i] = apply_hadamard_diffuser(e, params)
    return S

def _gen_signs(key, dim, stage, purpose, block_id=0):
    rng = make_rng(key, stage=stage, purpose=f"ksom.{purpose}", block_id=block_id)
    return rng.choice([-1.0, 1.0], size=dim).astype(np.float64)

def _gen_perm(key, dim, stage, purpose, block_id=0):
    rng = make_rng(key, stage=stage, purpose=f"ksom.{purpose}", block_id=block_id)
    perm = np.arange(dim)
    rng.shuffle(perm)
    return perm
