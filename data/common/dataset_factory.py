from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from data.robotwin2.action_normalization import normalize_action_normalization_config
from data.robotwin2.robotwin_agilex_dataset import RoboTwinTaskDataset
from data.efficient_wam_train_dataset import TRAIN_DATASET_METADATA


def _dataset_type(cfg: Mapping[str, Any]) -> str:
    root = cfg.get("root")
    if cfg.get("type") is None and root and (Path(root).expanduser() / TRAIN_DATASET_METADATA).exists():
        return "efficient_wam_train"
    return str(cfg.get("type", "robotwin")).lower()


def build_training_dataset(dataset_cfg: Mapping[str, Any], *, stage: str | None = None):
    cfg = dict(dataset_cfg)
    dataset_type = _dataset_type(cfg)

    if dataset_type in {"robotwin", "robotwin2"}:
        cfg["action_normalization"] = normalize_action_normalization_config(cfg.get("action_normalization"))
        return RoboTwinTaskDataset(
            dataset_dir=cfg["root"],
            data_mode=cfg.get("data_mode", "both"),
            task_mode=cfg.get("task_mode", "multi"),
            task_name=cfg.get("task_name"),
            randomized_limit_per_task=cfg.get("randomized_limit_per_task"),
            num_video_frames=cfg["num_video_frames"],
            video_size=tuple(cfg["video_size"]),
            global_downsample_rate=cfg.get("global_downsample_rate", 3),
            video_action_freq_ratio=cfg.get("video_action_freq_ratio", 2),
            max_episodes=cfg.get("max_episodes"),
            action_normalization=cfg["action_normalization"],
        )

    if dataset_type in {"efficient_wam_train", "train_dataset"}:
        from data.efficient_wam_train_dataset import EfficientWAMTrainDataset

        return EfficientWAMTrainDataset(
            root=cfg["root"],
            sampling=cfg.get("sampling"),
            language_policy=cfg.get("language_policy"),
            max_open_shards=cfg.get("max_open_shards"),
            include_metadata=cfg.get("include_metadata"),
            validate_runtime=cfg.get("validate_runtime"),
        )

    raise ValueError(f"Unsupported dataset.type={dataset_type!r} for stage={stage}")


def is_robotwin_dataset_config(dataset_cfg: Mapping[str, Any]) -> bool:
    return _dataset_type(dataset_cfg) in {"robotwin", "robotwin2"}


def is_efficient_wam_train_dataset_config(dataset_cfg: Mapping[str, Any]) -> bool:
    return _dataset_type(dataset_cfg) in {"efficient_wam_train", "train_dataset"}

