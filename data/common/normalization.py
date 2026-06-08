from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SUPPORTED_NORM_TYPES = {"identity", "bounds_99_woclip", "meanstd", "minmax"}


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.float()
        return tensor.cpu().numpy()
    return np.asarray(value)


def _from_numpy(value: np.ndarray, reference: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return value.astype(np.float32, copy=False)


def load_norm_stats(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    stats_path = Path(path)
    with stats_path.open("r") as f:
        payload = json.load(f)
    raw_stats = payload.get("norm_stats", payload)
    if not isinstance(raw_stats, Mapping):
        raise TypeError(f"Invalid norm stats file: {stats_path}")

    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, value in raw_stats.items():
        if isinstance(value, Mapping):
            stats[key] = {sub_key: _as_array(sub_value) for sub_key, sub_value in value.items()}
    return stats


class FeatureNormalizer:
    """Small normalizer shared by real-robot training and inference."""

    def __init__(
        self,
        stats: Mapping[str, Mapping[str, Any]] | str | Path | None,
        norm_type: str = "bounds_99_woclip",
    ) -> None:
        if norm_type not in SUPPORTED_NORM_TYPES:
            raise ValueError(f"Unsupported norm_type={norm_type!r}; expected one of {sorted(SUPPORTED_NORM_TYPES)}")
        if stats is None:
            self.stats: dict[str, dict[str, np.ndarray]] = {}
        elif isinstance(stats, (str, Path)):
            self.stats = load_norm_stats(stats)
        else:
            self.stats = {
                key: {sub_key: _as_array(sub_value) for sub_key, sub_value in value.items()}
                for key, value in stats.items()
            }
        self.norm_type = norm_type

    def _normalize_np(self, value: np.ndarray, key: str) -> np.ndarray:
        value = value.astype(np.float32, copy=False)
        if key not in self.stats or self.norm_type == "identity":
            return value
        stats = self.stats[key]
        if self.norm_type == "bounds_99_woclip":
            low = stats["q01"]
            high = stats["q99"]
            return (value - low) / (high - low + 1e-6) * 2.0 - 1.0
        if self.norm_type == "meanstd":
            return (value - stats["mean"]) / (stats["std"] + 1e-6)
        if self.norm_type == "minmax":
            return (value - stats["min"]) / (stats["max"] - stats["min"] + 1e-6) * 2.0 - 1.0
        return value

    def _denormalize_np(self, value: np.ndarray, key: str) -> np.ndarray:
        value = value.astype(np.float32, copy=False)
        if key not in self.stats or self.norm_type == "identity":
            return value
        stats = self.stats[key]
        if self.norm_type == "bounds_99_woclip":
            low = stats["q01"]
            high = stats["q99"]
            return (value + 1.0) / 2.0 * (high - low + 1e-6) + low
        if self.norm_type == "meanstd":
            return value * (stats["std"] + 1e-6) + stats["mean"]
        if self.norm_type == "minmax":
            return (value + 1.0) / 2.0 * (stats["max"] - stats["min"] + 1e-6) + stats["min"]
        return value

    def normalize(self, value: np.ndarray | torch.Tensor, key: str) -> np.ndarray | torch.Tensor:
        normalized = self._normalize_np(_to_numpy(value), key)
        return _from_numpy(normalized, value)

    def denormalize(self, value: np.ndarray | torch.Tensor, key: str) -> np.ndarray | torch.Tensor:
        denormalized = self._denormalize_np(_to_numpy(value), key)
        return _from_numpy(denormalized, value)


class RunningStats:
    """Simple in-memory stats collector for small real-robot datasets."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return
        self._chunks.append(arr.reshape(-1, arr.shape[-1]))

    def get_statistics(self) -> dict[str, list[float]]:
        if not self._chunks:
            raise ValueError("Cannot compute statistics from an empty RunningStats")
        values = np.concatenate(self._chunks, axis=0)
        return {
            "mean": values.mean(axis=0).astype(np.float32).tolist(),
            "std": values.std(axis=0).astype(np.float32).tolist(),
            "min": values.min(axis=0).astype(np.float32).tolist(),
            "max": values.max(axis=0).astype(np.float32).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).astype(np.float32).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).astype(np.float32).tolist(),
            "q02": np.quantile(values, 0.02, axis=0).astype(np.float32).tolist(),
            "q98": np.quantile(values, 0.98, axis=0).astype(np.float32).tolist(),
        }

    @property
    def count(self) -> int:
        return int(sum(chunk.shape[0] for chunk in self._chunks))


def save_norm_stats(path: str | Path, norm_stats: Mapping[str, Mapping[str, Any]], count: int) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"norm_stats": norm_stats, "count": int(count)}, f, indent=2)
