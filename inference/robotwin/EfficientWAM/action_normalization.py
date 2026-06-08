from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


DEFAULT_ACTION_NORMALIZATION: Dict[str, Any] = {
    "enabled": False,
    "type": "mean_std",
    "stats_path": "",
}


def normalize_action_normalization_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_ACTION_NORMALIZATION)
    if config:
        merged.update(dict(config))
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["type"] = str(merged.get("type", "mean_std"))
    merged["stats_path"] = "" if merged.get("stats_path") is None else str(merged.get("stats_path", ""))
    return merged


def _stats_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    stats = payload.get("robotwin_qpos", payload)
    if not isinstance(stats, Mapping):
        raise ValueError("qpos stats JSON must contain a mapping payload")
    return stats


class RoboTwinQposNormalizer:
    def __init__(
        self,
        *,
        enabled: bool = False,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        config: Optional[Mapping[str, Any]] = None,
    ):
        self._enabled = bool(enabled)
        self._mean = mean.detach().float().cpu() if mean is not None else None
        self._std = std.detach().float().cpu().clamp_min(1e-6) if std is not None else None
        self._config = normalize_action_normalization_config(config)
        self._config["enabled"] = self._enabled
        if self._enabled:
            if self._mean is None or self._std is None:
                raise ValueError("Enabled qpos normalization requires mean and std tensors")
            if self._mean.dim() != 1 or self._std.dim() != 1:
                raise ValueError("qpos normalization mean/std must be 1D tensors")
            if self._mean.shape != self._std.shape:
                raise ValueError(
                    f"qpos normalization mean/std shape mismatch: {tuple(self._mean.shape)} vs {tuple(self._std.shape)}"
                )

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "RoboTwinQposNormalizer":
        normalized = normalize_action_normalization_config(config)
        if not normalized["enabled"]:
            return cls(enabled=False, config=normalized)
        if normalized["type"] != "mean_std":
            raise ValueError(f"Unsupported action_normalization.type: {normalized['type']}")
        stats_payload = normalized.get("stats")
        if isinstance(stats_payload, Mapping):
            stats = _stats_payload(stats_payload)
        else:
            stats_path = normalized.get("stats_path", "")
            if not stats_path:
                raise ValueError("action_normalization stats or stats_path is required when normalization is enabled")
            path = Path(stats_path).expanduser()
            with path.open("r", encoding="utf-8") as f:
                stats = _stats_payload(json.load(f))
        mean = torch.as_tensor(stats.get("mean"), dtype=torch.float32)
        std = torch.as_tensor(stats.get("std"), dtype=torch.float32).clamp_min(1e-6)
        if mean.numel() == 0 or std.numel() == 0:
            raise ValueError("Invalid qpos mean/std stats")
        return cls(enabled=True, mean=mean, std=std, config=normalized)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return tensor
        return self._apply(tensor, inverse=False)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return tensor
        return self._apply(tensor, inverse=True)

    def _apply(self, tensor: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        if self._mean is None or self._std is None:
            raise RuntimeError("qpos normalizer is enabled without loaded stats")
        if tensor.shape[-1] != self._mean.numel():
            raise ValueError(
                f"qpos dim mismatch: expected {self._mean.numel()}, got {tensor.shape[-1]}"
            )
        mean = self._mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self._std.to(device=tensor.device, dtype=tensor.dtype)
        if inverse:
            return tensor * std + mean
        return (tensor - mean) / std
