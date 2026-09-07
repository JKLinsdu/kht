# KHT

**KHT** is a fast keyed Hadamard transform for secure, similarity-preserving vector
transformation in cross-organizational and multimodal retrieval. It replaces the dense
$O(d^2)$ orthogonal rotation of prior keyed transforms with a keyed cascade of fast
Walsh–Hadamard transforms (FWHT), reducing per-vector cost to $O(d \log d)$ and key
storage to $O(d)$, while keeping representations under different keys nearly orthogonal.

The transform is a 3-stage keyed pipeline:

```
x → permutation → blinding → offset → nonlinearity (arctan) → Hadamard diffuser → normalization
```

Each stage applies a key-derived permutation, input-independent blinding, an offset, a
bounded arctan nonlinearity, and the Hadamard diffuser

```
B_K = D_3 P_2 H D_2 P_1 H D_1,
```

where `D` are key-derived sign diagonals, `P` key-derived permutations, and `H` the
fixed Hadamard matrix evaluated via FWHT — all in $O(d \log d)$.

## Install

Requires Python 3.10+ and numpy/scipy.

```
pip install -e .
```

## Usage

```python
import numpy as np
from kht import KHTTransformation, KHTConfig

d = 128
config = KHTConfig(dim=d, stages=3, rotation_backend="ksom", nonlinearity="scaled_arctan")
key = bytes(range(32))            # 256-bit secret key
transform = KHTTransformation(config, secret_key=key)

x = np.random.randn(d); x /= np.linalg.norm(x)
z = transform.transform(x)        # transformed vector, L2-normalized, same dimension
```

Same-key retrieval preserves similarity; independent keys give nearly orthogonal spaces:

```python
x2 = np.random.randn(d); x2 /= np.linalg.norm(x2)
print(x @ x2)                                   # original similarity
print(transform.transform(x) @ transform.transform(x2))   # ≈ original (preserved)
other = KHTTransformation(config, secret_key=bytes(range(32, 64)))
print(transform.transform(x) @ other.transform(x))        # ≈ 0 (isolated)
```

## License

Released under the MIT License. Please cite the accompanying paper if you use this
code.

## Notes

This package contains the **core transform** (`kht/kht_transform.py`, `hadamard_diffuser.py`,
`nonlinearity.py`, `hadamard.py`). The full experimental reproduction (retrieval, key
isolation, efficiency, end-to-end RAG) and the Trans-RAG baseline are separate and are
not packaged here.
