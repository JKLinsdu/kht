"""Bounded nonlinearity used by KHT (the arctan/tanh elementwise map).

Provides scaled_arctan, scaled_tanh and get_nonlinearity."""

import numpy as np

def scaled_tanh(x: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """ϕ(x) = tanh(βx) / β — scaled tanh nonlinearity."""
    return np.tanh(beta * x) / beta

def scaled_softsign(x: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """ϕ(x) = x / (1 + β|x|) — cheaper saturation than tanh."""
    return x / (1.0 + beta * np.abs(x))

def scaled_arctan(x: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """ϕ(x) = β · arctan(x / τ) — bounded nonlinearity."""
    return np.arctan(beta * x) / beta

def hard_clip_identity(x: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """ϕ(x) = clip(x, -C, C) with C = 1/β — piecewise linear."""
    C = 1.0 / max(beta, 1e-8)
    return np.clip(x, -C, C)

def radial_scaled_tanh(x: np.ndarray, beta: float = 0.1, eps: float = 1e-12) -> np.ndarray:
    """ϕ(x) = x * tanh(β‖x‖) / (β‖x‖) — radial basis variant.

    Applies the same scalar gain to all coordinates based on vector norm.
    """
    r = np.linalg.norm(x, axis=-1, keepdims=True)
    gate = np.tanh(beta * r) / (beta * r + eps)
    return x * gate

NONLINEARITY_MAP = {
    "scaled_tanh": scaled_tanh,
    "scaled_softsign": scaled_softsign,
    "scaled_arctan": scaled_arctan,
    "hard_clip_identity": hard_clip_identity,
    "radial_scaled_tanh": radial_scaled_tanh,
}

def get_nonlinearity(name: str = "scaled_tanh"):
    """Look up a bounded nonlinearity by name."""
    fn = NONLINEARITY_MAP.get(name)
    if fn is None:
        raise ValueError(f"Unknown nonlinearity: {name}. Choices: {list(NONLINEARITY_MAP.keys())}")
    return fn
