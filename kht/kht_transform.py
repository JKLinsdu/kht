"""The KHT keyed vector transform (KHTTransformation).

A 3-stage pipeline: permutation -> blinding -> offset -> nonlinearity (arctan)
-> Hadamard diffuser -> normalization, key-derived from a 256-bit secret key.
Provides KHTConfig, KHTTransformation and transform_vectors."""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np
try:  # torch is optional: only used to detect torch-tensor inputs
    import torch
except ImportError:
    torch = None
from scipy.linalg import qr

KEY_VERSION = "1"

@dataclass(frozen=True)
class KHTConfig:
    dim: int
    stages: int = 3
    beta: float = 0.1
    alpha: float = 0.05
    use_permutation: bool = True
    use_blinding: bool = True
    use_sign: bool = True
    rotation_backend: str = "dense_qr"
    hadamard_block_size: int | None = None
    ksom_variant: str = "dhpdhd"
    ksom_block_size: int | None = None
    nonlinearity: str = "scaled_tanh"
    offset_scale: float = 0.1
    norm_scale: float = 0.05

KHTStage = Dict[str, Any]

class _KHTBase:
    """
    Original Trans-RAG vector transformation implementation.

    This is the original local implementation style: parameters and
    permutation patterns are generated during key initialization and serialized
    into the key object.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dim_in = int(config.get("dim_in", 768))
        self.dim_out = int(config.get("dim_out", 768))
        self.name = self.__class__.__name__

        self.stages = int(config.get("stages", 3))
        raw_beta = config.get("nonlinearity_beta", 0.1)
        self.nonlinearity_beta = 0.1 if raw_beta is None else float(raw_beta)

        self.use_permutation = config.get("use_permutation", True)
        self.use_blinding = config.get("use_blinding", True)
        self.use_sign = config.get("use_sign", True)
        raw_blinding_scale = config.get("blinding_scale", 0.05)
        self.blinding_scale = 0.05 if raw_blinding_scale is None else float(raw_blinding_scale)
        self.offset_scale = float(config.get("offset_scale", 0.1))
        self.norm_scale = float(config.get("norm_scale", 0.05))

        self.nonlinearity_name = config.get("nonlinearity", "scaled_tanh")

        self.rotation_backend = config.get("rotation_backend", "dense_qr")
        self.hadamard_block_size = config.get("hadamard_block_size", None)
        self.ksom_variant = config.get("ksom_variant", "dhpdhd")
        self.ksom_block_size = config.get("ksom_block_size", None)

        self.secret_key = config.get("secret_key")
        if self.secret_key is None:
            self.secret_key = secrets.token_bytes(32)
        else:
            self.secret_key = _secret_to_bytes(self.secret_key)

        self.parameters = None
        self.permutation_patterns = None
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if self.dim_in <= 0 or self.dim_out <= 0:
            raise ValueError("dim_in and dim_out must be positive")
        if self.dim_in != self.dim_out:
            raise ValueError("KHT requires dim_in == dim_out")
        if self.stages <= 0:
            raise ValueError("stages must be positive")
        if self.nonlinearity_beta <= 0:
            raise ValueError("nonlinearity_beta must be positive")
        if self.blinding_scale < 0:
            raise ValueError("blinding_scale must be non-negative")
        if len(self.secret_key) < 32:
            raise ValueError("secret_key must contain at least 32 bytes")

    def initialize_key(self) -> Dict[str, Any]:
        self.parameters = self._initialize_parameters()

        if self.use_permutation:
            self.permutation_patterns = self._initialize_permutation_patterns()

        serializable_parameters = []
        for stage_params in self.parameters:
            serializable_stage = {}
            for key, value in stage_params.items():
                if key == "block_param_sets":
                    serializable_stage[key] = [
                        [s1.tolist(), s2.tolist(), s3.tolist(), perm.tolist()]
                        for (s1, s2, s3, perm) in value
                    ]
                elif isinstance(value, np.ndarray):
                    serializable_stage[key] = value.tolist()
                else:
                    serializable_stage[key] = value
            serializable_parameters.append(serializable_stage)

        serializable_permutations = None
        if self.use_permutation and self.permutation_patterns:
            serializable_permutations = []
            for pattern in self.permutation_patterns:
                serializable_permutations.append(pattern.tolist())

        key = {
            "parameters": serializable_parameters,
            "secret_key": self.secret_key.hex() if isinstance(self.secret_key, bytes) else self.secret_key,
            "stages": self.stages,
            "nonlinearity_beta": self.nonlinearity_beta,
            "dim_in": self.dim_in,
            "dim_out": self.dim_out,
            "transformation_type": "kht",
            "use_permutation": self.use_permutation,
            "use_blinding": self.use_blinding,
            "blinding_scale": self.blinding_scale,
            "offset_scale": self.offset_scale,
            "norm_scale": self.norm_scale,
            "rotation_backend": self.rotation_backend,
            "ksom_variant": self.ksom_variant,
            "nonlinearity": self.nonlinearity_name,
        }

        if self.use_permutation and serializable_permutations:
            key["permutation_patterns"] = serializable_permutations

        return key

    def _initialize_hadamard_parameters(self, stage: int) -> Dict[str, Any]:
        """Generate keyed Hadamard rotation parameters for a stage.

        Does NOT store dense matrices — only sign vectors and permutations.
        """
        from .hadamard import (
            generate_hadamard_signs, generate_hadamard_permutation,
            decompose_into_power_of_two_blocks, apply_keyed_hadamard_rotation,
            apply_block_hadamard_rotation,
        )

        dim = self.dim_in
        if self.rotation_backend == "block_hadamard":
            blocks = decompose_into_power_of_two_blocks(dim)
            block_param_sets = []
            for blk_idx, (start, length) in enumerate(blocks):
                s1 = generate_hadamard_signs(self.secret_key, stage * 100 + blk_idx * 3, length)
                s2 = generate_hadamard_signs(self.secret_key, stage * 100 + blk_idx * 3 + 1, length)
                s3 = generate_hadamard_signs(self.secret_key, stage * 100 + blk_idx * 3 + 2, length)
                blk_perm = generate_hadamard_permutation(self.secret_key, stage * 100 + blk_idx, length)
                block_param_sets.append((s1, s2, s3, blk_perm))
            global_perm = generate_hadamard_permutation(self.secret_key, stage * 1000, dim)
            return {
                "rotation_backend": "block_hadamard",
                "hadamard_blocks": blocks,
                "block_param_sets": block_param_sets,
                "global_permutation": global_perm,
            }
        else:
            signs1 = generate_hadamard_signs(self.secret_key, stage * 3, dim)
            signs2 = generate_hadamard_signs(self.secret_key, stage * 3 + 1, dim)
            signs3 = generate_hadamard_signs(self.secret_key, stage * 3 + 2, dim)
            permutation = generate_hadamard_permutation(self.secret_key, stage, dim)
            return {
                "rotation_backend": "hadamard",
                "signs1": signs1,
                "signs2": signs2,
                "signs3": signs3,
                "hadamard_permutation": permutation,
            }

    def _initialize_parameters(self):
        parameters = []

        for stage in range(self.stages):
            stage_seed = hashlib.sha256(self.secret_key + f"stage{stage}".encode()).digest()

            np.random.seed(int.from_bytes(stage_seed[:4], byteorder="big"))

            if self.rotation_backend == "ksom":
                from .hadamard_diffuser import derive_hadamard_diffuser_params
                variant = self.ksom_variant
                if self.ksom_variant == "block_dhpdhd" or not (self.dim_in & (self.dim_in - 1)):
                    pass  # use as-is
                elif self.dim_in != (self.dim_in & -self.dim_in):
                    variant = "block_" + self.ksom_variant
                ksom_params = derive_hadamard_diffuser_params(self.secret_key, self.dim_in, stage=stage, variant=variant)
                if not self.use_sign:
                    _neutralize_signs(ksom_params)
                hadamard_params = {"rotation_backend": "ksom", "ksom_variant": variant, "ksom_params": _serialize_ksom(ksom_params)}
            elif self.rotation_backend in ("hadamard", "block_hadamard"):
                hadamard_params = self._initialize_hadamard_parameters(stage)
            else:
                random_matrix = np.random.randn(self.dim_in, self.dim_out)
                q, r = qr(random_matrix)

                diagonal_signs = np.sign(np.diag(r))
                diagonal_signs[diagonal_signs == 0] = 1
                d = np.diag(diagonal_signs)
                orthogonal_matrix = q @ d

            offset_scale = self.offset_scale * (stage + 1)
            offset_seed = int.from_bytes(stage_seed[4:8], byteorder="big")
            np.random.seed(offset_seed)
            offset_vector = (offset_scale / np.sqrt(self.dim_out)) * np.random.randn(self.dim_out)

            norm_seed = int.from_bytes(stage_seed[8:12], byteorder="big")
            np.random.seed(norm_seed)
            norm_vector = (self.norm_scale / np.sqrt(self.dim_out)) * np.random.randn(self.dim_out)

            beta_seed = int.from_bytes(stage_seed[12:16], byteorder="big")
            np.random.seed(beta_seed)
            beta = self.nonlinearity_beta * (1 + 0.2 * np.random.randn())

            stage_params = {
                "offset_vector": offset_vector,
                "norm_vector": norm_vector,
                "beta": beta,
            }
            if self.rotation_backend in ("hadamard", "block_hadamard", "ksom"):
                stage_params.update(hadamard_params)
            else:
                stage_params["orthogonal_matrix"] = orthogonal_matrix
            parameters.append(stage_params)

        return parameters

    def _initialize_permutation_patterns(self) -> List[np.ndarray]:
        permutation_patterns = []

        for stage in range(self.stages):
            stage_perm_seed = hashlib.sha256(
                self.secret_key + f"perm_stage{stage}".encode()
            ).digest()

            np.random.seed(int.from_bytes(stage_perm_seed[:4], byteorder="big"))

            permutation = np.arange(self.dim_in)
            np.random.shuffle(permutation)

            permutation_patterns.append(permutation)

        return permutation_patterns

    def _generate_blinding_factors(self, vector: np.ndarray, stage_id: int) -> np.ndarray:
        vector_fingerprint = hashlib.sha256(vector.tobytes()).digest()[:8]
        combined_seed = hashlib.sha256(
            self.secret_key
            + f"blind_stage{stage_id}".encode()
            + vector_fingerprint
        ).digest()

        seed = int.from_bytes(combined_seed[:4], byteorder="big")
        random_generator = np.random.RandomState(seed)

        dim = vector.shape[0]
        blinding_factors = random_generator.randn(dim)

        norm = np.linalg.norm(blinding_factors)
        if norm <= 1e-12:
            return np.zeros(dim, dtype=vector.dtype)
        blinding_factors = self.blinding_scale * blinding_factors / norm

        return blinding_factors

    def _apply_permutation(self, vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        return vector[permutation]

    def _apply_inverse_permutation(self, vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        inverse_perm = np.zeros_like(permutation)
        inverse_perm[permutation] = np.arange(len(permutation))
        return vector[inverse_perm.astype(int)]

    def transform(self, vectors: np.ndarray, key: Dict[str, Any]) -> np.ndarray:
        if self.parameters is None or key.get("secret_key") != getattr(self, "secret_key", None):
            self.secret_key = _secret_to_bytes(key["secret_key"])
            self.stages = int(key.get("stages", 3))
            raw_beta = key.get("nonlinearity_beta", 0.1)
            self.nonlinearity_beta = 0.1 if raw_beta is None else float(raw_beta)
            self.dim_in = int(_first_present(key, "dim_in", "dim", default=self.dim_in))
            self.dim_out = int(_first_present(key, "dim_out", "dim", default=self.dim_out))

            self.use_permutation = key.get("use_permutation", True)
            self.use_blinding = key.get("use_blinding", True)
            raw_blinding_scale = key.get("blinding_scale", 0.05)
            self.blinding_scale = 0.05 if raw_blinding_scale is None else float(raw_blinding_scale)
            self.offset_scale = float(key.get("offset_scale", 0.1))
            self.norm_scale = float(key.get("norm_scale", 0.05))
            self._validate_configuration()

            if "parameters" in key:
                parameters = key["parameters"]
                self.parameters = []
                for stage_params in parameters:
                    restored_stage = {}
                    for param_name, value in stage_params.items():
                        if param_name == "block_param_sets":
                            restored_stage[param_name] = [
                                (np.array(s1), np.array(s2), np.array(s3), np.array(perm))
                                for (s1, s2, s3, perm) in value
                            ]
                        elif isinstance(value, list):
                            restored_stage[param_name] = np.array(value)
                        else:
                            restored_stage[param_name] = value
                    self.parameters.append(restored_stage)
            else:
                self.parameters = self._initialize_parameters()

            if self.use_permutation and "permutation_patterns" in key:
                self.permutation_patterns = []
                for pattern in key["permutation_patterns"]:
                    self.permutation_patterns.append(np.array(pattern))
            elif self.use_permutation:
                self.permutation_patterns = self._initialize_permutation_patterns()

        if torch is not None and isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy()
        else:
            vectors = np.asarray(vectors)
            if not np.issubdtype(vectors.dtype, np.floating):
                vectors = vectors.astype(np.float32)

        single_vector = False
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            single_vector = True
        elif vectors.ndim != 2:
            raise ValueError(f"Expected a 1-D or 2-D vector array; got shape {vectors.shape}")

        if vectors.shape[1] != self.dim_in:
            raise ValueError(f"Expected vectors with dimension {self.dim_in}; got {vectors.shape[1]}")

        transformed_vectors = np.zeros((vectors.shape[0], self.dim_out), dtype=vectors.dtype)

        for index in range(vectors.shape[0]):
            vector = vectors[index]
            transformed = vector
            for stage in range(len(self.parameters)):
                if self.use_permutation and self.permutation_patterns:
                    transformed = self._apply_permutation(transformed, self.permutation_patterns[stage])

                if self.use_blinding:
                    pre_blinding = self._generate_blinding_factors(transformed, stage)
                    transformed = transformed + pre_blinding - np.mean(pre_blinding)

                transformed = self._apply_stage_transformation(transformed, self.parameters[stage])

                if self.use_blinding:
                    post_blinding = self._generate_blinding_factors(transformed, stage + self.stages)
                    transformed = transformed + post_blinding - np.mean(post_blinding)

                if self.use_permutation and self.permutation_patterns:
                    transformed = self._apply_inverse_permutation(transformed, self.permutation_patterns[stage])

            transformed_vectors[index] = transformed

        normalized = _l2_normalize(transformed_vectors)

        if single_vector:
            return normalized[0]
        return normalized

    def _nonlinear_function(self, x, beta):
        if abs(beta) <= 1e-12:
            return x
        from .nonlinearity import get_nonlinearity
        fn = get_nonlinearity(self.nonlinearity_name)
        return fn(x, beta)

    def _apply_stage_transformation(self, vector, stage_params):
        b = stage_params["offset_vector"]
        c = stage_params["norm_vector"]
        beta = stage_params["beta"]

        vector_with_offset = vector + b
        nonlinear_output = self._nonlinear_function(vector_with_offset, beta)

        backend = stage_params.get("rotation_backend", "dense_qr")
        if backend == "ksom":
            from .hadamard_diffuser import apply_hadamard_diffuser
            ksom_params = _deserialize_ksom(stage_params["ksom_params"])
            transformed = apply_hadamard_diffuser(nonlinear_output, ksom_params)
        elif backend == "hadamard":
            from .hadamard import apply_keyed_hadamard_rotation
            transformed = apply_keyed_hadamard_rotation(
                nonlinear_output,
                stage_params["signs1"],
                stage_params["signs2"],
                stage_params["signs3"],
                stage_params["hadamard_permutation"],
            )
        elif backend == "block_hadamard":
            from .hadamard import apply_block_hadamard_rotation
            transformed = apply_block_hadamard_rotation(
                nonlinear_output,
                stage_params["block_param_sets"],
                stage_params["hadamard_blocks"],
                stage_params.get("global_permutation"),
            )
        else:
            w = stage_params["orthogonal_matrix"]
            transformed = nonlinear_output @ w

        transformed = transformed + c

        return transformed

class KHTTransformation(_KHTBase):
    """The KHT keyed vector transform."""

    def __init__(self, config: KHTConfig | Mapping[str, Any], secret_key: bytes | str | None = None):
        legacy_config = _config_to_legacy_dict(config)
        if secret_key is not None:
            legacy_config["secret_key"] = _secret_to_bytes(secret_key)
        super().__init__(legacy_config)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "KHTTransformation":
        return cls(config=config, secret_key=config.get("secret_key"))

    @classmethod
    def from_key(cls, key: Mapping[str, Any]) -> "KHTTransformation":
        return cls(config=key, secret_key=key["secret_key"])

    def initialize_key(self, org_id: str | None = None) -> Dict[str, Any]:
        key = super().initialize_key()
        key["key_version"] = KEY_VERSION
        key["dim"] = key["dim_in"]
        if org_id is not None:
            key["org_id"] = org_id
        return key

    def transform(self, vectors: np.ndarray, key: Dict[str, Any] | None = None) -> np.ndarray:
        if key is None:
            key = self.initialize_key()
        return super().transform(vectors, dict(key))

def _neutralize_signs(params) -> None:
    """Zero out the key-derived sign flips in KSOM params (used by h2_no_sign)."""
    for attr in ("signs1", "signs2", "signs3"):
        v = getattr(params, attr, None)
        if v is not None:
            setattr(params, attr, np.ones_like(v))
    for bp in getattr(params, "block_params", []) or []:
        for k in ("signs1", "signs2", "signs3"):
            if bp.get(k) is not None:
                bp[k] = np.ones_like(bp[k])

def _serialize_ksom(params):
    """Serialize KSOMParams to JSON-safe dict."""
    d = {"variant": params.variant, "dim": params.dim, "stage": params.stage}
    for attr in ["signs1", "signs2", "signs3", "perm1", "perm2", "global_perm"]:
        v = getattr(params, attr, None)
        if v is not None:
            d[attr] = v.tolist()
    if params.blocks:
        d["blocks"] = params.blocks
        d["block_params"] = []
        for bp in params.block_params:
            bp_d = {}
            for k in ["signs1", "signs2", "signs3", "perm1", "perm2"]:
                if bp.get(k) is not None:
                    bp_d[k] = bp[k].tolist()
            d["block_params"].append(bp_d)
    return d

def _deserialize_ksom(d):
    """Deserialize JSON dict back to HadamardDiffuserParams."""
    from .hadamard_diffuser import HadamardDiffuserParams
    params = HadamardDiffuserParams(dim=d["dim"], variant=d["variant"], stage=d.get("stage", 0))
    for attr in ["signs1", "signs2", "signs3", "perm1", "perm2", "global_perm"]:
        if attr in d:
            setattr(params, attr, np.array(d[attr]))
    if "blocks" in d:
        params.blocks = d["blocks"]
        params.block_params = []
        for bp_d in d.get("block_params", []):
            bp = {}
            for k in ["signs1", "signs2", "signs3", "perm1", "perm2"]:
                if k in bp_d:
                    bp[k] = np.array(bp_d[k])
            params.block_params.append(bp)
    return params

def _config_to_legacy_dict(config: KHTConfig | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(config, KHTConfig):
        return {
            "dim_in": config.dim,
            "dim_out": config.dim,
            "stages": config.stages,
            "nonlinearity_beta": config.beta,
            "use_permutation": config.use_permutation,
            "use_blinding": config.use_blinding,
            "use_sign": config.use_sign,
            "blinding_scale": config.alpha,
            "offset_scale": config.offset_scale,
            "norm_scale": config.norm_scale,
            "rotation_backend": config.rotation_backend,
            "hadamard_block_size": config.hadamard_block_size,
            "ksom_variant": config.ksom_variant,
            "nonlinearity": config.nonlinearity,
            "ksom_block_size": config.ksom_block_size,
        }

    dim = _first_present(config, "dim", "dim_in", "dim_out", default=768)
    return {
        "dim_in": int(_first_present(config, "dim_in", "dim", default=dim)),
        "dim_out": int(_first_present(config, "dim_out", "dim", default=dim)),
        "stages": int(_first_present(config, "stages", default=3)),
        "nonlinearity_beta": float(_first_present(config, "nonlinearity_beta", "beta", default=0.1)),
        "use_permutation": bool(_first_present(config, "use_permutation", default=True)),
        "use_blinding": bool(_first_present(config, "use_blinding", default=True)),
        "use_sign": bool(_first_present(config, "use_sign", default=True)),
        "blinding_scale": float(_first_present(config, "blinding_scale", "alpha", default=0.05)),
        "offset_scale": float(_first_present(config, "offset_scale", default=0.1)),
        "norm_scale": float(_first_present(config, "norm_scale", default=0.05)),
        "rotation_backend": config.get("rotation_backend", "dense_qr"),
        "hadamard_block_size": config.get("hadamard_block_size", None),
        "ksom_variant": config.get("ksom_variant", "dhpdhd"),
        "nonlinearity": config.get("nonlinearity", "scaled_tanh"),
        "ksom_block_size": config.get("ksom_block_size", None),
        "secret_key": _secret_to_bytes(config["secret_key"]) if "secret_key" in config else None,
    }

def _first_present(values: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in values and values[name] is not None:
            return values[name]
    return default

def _secret_to_bytes(secret_key: bytes | str) -> bytes:
    if isinstance(secret_key, bytes):
        return secret_key
    try:
        return bytes.fromhex(secret_key)
    except ValueError:
        return secret_key.encode("utf-8")

def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1e-10)
    return vectors / norms

