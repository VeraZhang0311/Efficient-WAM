"""Shared config helpers for RoboTwin train-dataset construction."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch

from data.robotwin2.action_normalization import normalize_action_normalization_config
from data.robotwin2.robotwin_agilex_dataset import RoboTwinTaskDataset


def dataset_type(dataset_cfg: Mapping[str, Any]) -> str:
    return str(dataset_cfg.get("type", "robotwin")).lower()


def output_root(config: Mapping[str, Any], override: Optional[str] = None) -> Optional[str]:
    if override:
        return str(override)
    if config.get("output_root"):
        return str(config["output_root"])
    dataset_cfg = config.get("dataset")
    if isinstance(dataset_cfg, Mapping) and dataset_cfg.get("root"):
        return str(dataset_cfg["root"])
    return None


def source_dataset_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(config.get("source"), Mapping):
        return dict(config["source"])

    dataset_cfg = dict(config["dataset"])
    if dataset_type(dataset_cfg) in {"efficient_wam_train", "train_dataset"}:
        source_cfg = (
            dataset_cfg.get("source")
            or dataset_cfg.get("source_dataset")
            or config.get("source_dataset")
        )
        if not source_cfg:
            raise ValueError(
                "efficient_wam_train configs require dataset.source with the original RoboTwin dataset settings"
            )
        return dict(source_cfg)
    return dataset_cfg


def training_dataset_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    dataset_cfg = dict(config.get("dataset") or {})
    if isinstance(config.get("source"), Mapping):
        dataset_cfg.setdefault("type", "efficient_wam_train")
        dataset_cfg.setdefault("sampling", {"mode": "random_full_epoch"})
        return dataset_cfg
    if dataset_type(dataset_cfg) in {"efficient_wam_train", "train_dataset"}:
        return dataset_cfg
    return {"type": "efficient_wam_train", "sampling": {"mode": "random_full_epoch"}}


def build_robotwin_dataset(
    config: Mapping[str, Any],
    *,
    action_stats_path: Optional[str] = None,
    action_normalization_enabled: Optional[bool] = None,
) -> RoboTwinTaskDataset:
    dataset_cfg = source_dataset_config(config)
    dataset_cfg["action_normalization"] = normalize_action_normalization_config(
        dataset_cfg.get("action_normalization")
    )
    if action_normalization_enabled is not None:
        dataset_cfg["action_normalization"]["enabled"] = bool(action_normalization_enabled)
    if action_stats_path:
        dataset_cfg["action_normalization"]["stats_path"] = str(action_stats_path)
    return RoboTwinTaskDataset(
        dataset_dir=dataset_cfg["root"],
        data_mode=dataset_cfg.get("data_mode", "both"),
        task_mode=dataset_cfg.get("task_mode", "multi"),
        task_name=dataset_cfg.get("task_name"),
        randomized_limit_per_task=dataset_cfg.get("randomized_limit_per_task"),
        num_video_frames=dataset_cfg["num_video_frames"],
        video_size=tuple(dataset_cfg["video_size"]),
        global_downsample_rate=dataset_cfg.get("global_downsample_rate", 3),
        video_action_freq_ratio=dataset_cfg.get("video_action_freq_ratio", 2),
        max_episodes=dataset_cfg.get("max_episodes"),
        action_normalization=dataset_cfg["action_normalization"],
    )


def compact_vae_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    latent_cfg = config.get("latent")
    if isinstance(latent_cfg, Mapping) and latent_cfg.get("vae_path"):
        return dict(latent_cfg)
    model_cfg = config.get("model")
    if isinstance(model_cfg, Mapping) and isinstance(model_cfg.get("compact_wan"), Mapping):
        return dict(model_cfg["compact_wan"])
    student_cfg = config.get("student")
    if isinstance(student_cfg, Mapping):
        return dict(student_cfg)
    raise KeyError("Expected config.model.compact_wan or config.student to provide vae_path/precision")


def vae_path_from_config(config: Mapping[str, Any]) -> str:
    return str(compact_vae_config(config)["vae_path"])


def dtype_from_config(config: Mapping[str, Any]) -> torch.dtype:
    precision = str(compact_vae_config(config).get("precision", "bfloat16")).lower()
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[precision]


def reuse_condition_latent_from_clean(config: Mapping[str, Any]) -> bool:
    latent_cfg = config.get("latent")
    if isinstance(latent_cfg, Mapping) and "reuse_condition_latent_from_clean" in latent_cfg:
        return bool(latent_cfg.get("reuse_condition_latent_from_clean", True))
    performance_cfg = config.get("performance", {})
    return bool(performance_cfg.get("reuse_condition_latent_from_clean", True))


def future_video_size_from_config(config: Mapping[str, Any]) -> Optional[tuple[int, int]]:
    latent_cfg = config.get("latent")
    size = latent_cfg.get("future_video_size") if isinstance(latent_cfg, Mapping) else None
    if size is None:
        return None
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"latent.future_video_size must be [height, width], got {size!r}")
    future_size = (int(size[0]), int(size[1]))
    if any(value <= 0 or value % 32 != 0 for value in future_size):
        raise ValueError(
            "latent.future_video_size must be positive and divisible by 32 for WAN VAE+patch stride, "
            f"got {future_size!r}"
        )
    return future_size
