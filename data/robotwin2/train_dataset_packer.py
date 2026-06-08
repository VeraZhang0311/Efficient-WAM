"""Build the final EfficientWAM train dataset from preprocessed RoboTwin data."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root.resolve()) not in sys.path:
    sys.path.insert(0, str(project_root.resolve()))

from third_party.wan.modules.vae2_2 import Wan2_2_VAE
from data.robotwin2.action_normalization import normalize_action_normalization_config
from data.robotwin2.robotwin_agilex_dataset import RoboTwinTaskDataset
from data.robotwin2.robotwin_data_convert.compute_qpos_stats import compute_qpos_stats_from_files
from data.robotwin2.train_dataset_config import (
    build_robotwin_dataset,
    compact_vae_config,
    dtype_from_config,
    future_video_size_from_config,
    output_root as output_root_from_config,
    reuse_condition_latent_from_clean,
    source_dataset_config,
    training_dataset_config,
    vae_path_from_config,
)
from data.efficient_wam_train_dataset import (
    TRAIN_DATASET_ACTION_STATS,
    TRAIN_DATASET_FORMAT,
    TRAIN_DATASET_LANG_DIR,
    TRAIN_DATASET_METADATA,
    TRAIN_DATASET_SHARD_DIR,
    TRAIN_DATASET_STATS,
    TRAIN_DATASET_VERSION,
)
from data.utils.image_utils import load_video_frames
from train.common import load_yaml_config, setup_logging


logger = logging.getLogger(__name__)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def _phase(rank: int, index: int, total: int, message: str) -> None:
    if rank == 0:
        logger.info("[%d/%d] %s", index, total, message)


def _distributed_info(device: str, timeout_minutes: int) -> tuple[int, int, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    resolved_device = device

    if world_size > 1:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            resolved_device = f"cuda:{local_rank}"
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                timeout=timedelta(minutes=max(1, int(timeout_minutes))),
            )
    return rank, world_size, resolved_device


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _cuda_synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _resize_video_frames(frames: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    batch, frames_per_clip, channels, height, width = frames.shape
    if (height, width) == tuple(size_hw):
        return frames
    resized = F.interpolate(
        frames.reshape(batch * frames_per_clip, channels, height, width).float(),
        size=tuple(size_hw),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).to(dtype=frames.dtype)
    return resized.reshape(batch, frames_per_clip, channels, *size_hw)


def _build_low_resolution_full_video(
    first_frame: torch.Tensor,
    video_frames: torch.Tensor,
    size_hw: tuple[int, int],
) -> torch.Tensor:
    first_low = _resize_video_frames(first_frame.unsqueeze(1), size_hw)
    future_low = _resize_video_frames(video_frames, size_hw)
    full_low = torch.cat([first_low, future_low], dim=1)
    return (full_low * 2.0 - 1.0).permute(0, 2, 1, 3, 4)


def _stats_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    stats = payload.get("robotwin_qpos", payload)
    if not isinstance(stats, dict):
        raise ValueError("qpos stats JSON must contain a mapping payload")
    return stats


def _path_independent_action_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    stats = dict(_stats_payload(payload))
    stats.pop("root", None)
    stats.pop("errors", None)
    return {"robotwin_qpos": stats}


def _source_splits(data_mode: str) -> List[str]:
    if data_mode == "both":
        return ["clean", "randomized"]
    return [str(data_mode)]


def _prepare_action_stats(config: Dict[str, Any], output_root: Path, args: argparse.Namespace) -> Optional[Path]:
    source_cfg = source_dataset_config(config)
    norm_cfg = normalize_action_normalization_config(source_cfg.get("action_normalization"))
    if not norm_cfg["enabled"]:
        return None

    target = output_root / TRAIN_DATASET_ACTION_STATS
    logger.info("Computing action stats: %s", target)
    stats_dataset = build_robotwin_dataset(config, action_normalization_enabled=False)
    qpos_files = sorted({Path(episode["qpos_path"]) for episode in stats_dataset.all_episodes})
    stats = compute_qpos_stats_from_files(
        qpos_files,
        num_workers=max(1, int(args.num_workers)),
        root=Path(source_cfg["root"]).expanduser(),
        splits=_source_splits(str(source_cfg.get("data_mode", "both"))),
    )
    _write_json_atomic(target, _path_independent_action_stats(stats))
    logger.info("Wrote action stats: %s", target)
    return target


def _path_independent_action_normalization(config: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(config)
    payload.pop("stats_path", None)
    if payload.get("enabled"):
        payload["stats_file"] = TRAIN_DATASET_ACTION_STATS
    return payload


def _path_independent_training_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(config)
    payload.pop("root", None)
    payload.pop("source", None)
    payload.pop("source_dataset", None)
    payload.setdefault("type", "efficient_wam_train")
    payload.setdefault("sampling", {"mode": "random_full_epoch"})
    return payload


def _hash_json_payload(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_language_list(path: str) -> List[torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected non-empty list of language embeddings in {path}")

    embeddings: List[torch.Tensor] = []
    for value in payload:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() != 2:
            raise ValueError(f"Expected language embedding [tokens, dim], got {tuple(tensor.shape)} in {path}")
        embeddings.append(tensor.detach().cpu().contiguous())
    return embeddings


def _dtype_from_name(name: str, fallback: Optional[torch.dtype] = None) -> Optional[torch.dtype]:
    normalized = str(name).lower()
    if normalized == "keep":
        return fallback
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[normalized]


def _write_language_shard(
    lang_root: Path,
    shard_index: int,
    embeddings: List[torch.Tensor],
) -> Dict[str, Any]:
    if not embeddings:
        raise ValueError("Cannot write an empty language shard")
    try:
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("Packing train dataset requires safetensors to be installed") from exc

    token_offsets = [0]
    for embedding in embeddings:
        token_offsets.append(token_offsets[-1] + int(embedding.shape[0]))
    token_data = torch.cat(embeddings, dim=0).contiguous()
    offsets_tensor = torch.tensor(token_offsets, dtype=torch.long)

    shard_path = lang_root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.safetensors"
    shard_meta_path = lang_root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_shard_path = shard_path.with_name(f"{shard_path.name}.tmp.{os.getpid()}")
    save_file(
        {
            "token_data": token_data,
            "token_offsets": offsets_tensor,
        },
        str(tmp_shard_path),
    )
    tmp_shard_path.replace(shard_path)
    metadata = {
        "shard_index": int(shard_index),
        "item_count": int(len(embeddings)),
        "token_count": int(token_data.shape[0]),
        "embedding_dim": int(token_data.shape[1]),
        "dtype": str(token_data.dtype).replace("torch.", ""),
    }
    _write_json_atomic(shard_meta_path, metadata)
    return metadata


def _write_language_bank(
    dataset: RoboTwinTaskDataset,
    output_root: Path,
    *,
    lang_items_per_shard: int,
    lang_dtype_name: str,
) -> Dict[int, int]:
    lang_root = output_root / TRAIN_DATASET_LANG_DIR
    groups: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    lang_group_ids: Dict[int, int] = {}

    current_embeddings: List[torch.Tensor] = []
    current_shard_index = 0
    embedding_dim: Optional[int] = None
    target_dtype: Optional[torch.dtype] = None

    def flush_language_shard() -> None:
        nonlocal current_embeddings, current_shard_index
        if not current_embeddings:
            return
        _write_language_shard(lang_root, current_shard_index, current_embeddings)
        current_shard_index += 1
        current_embeddings = []

    iterator = tqdm(dataset.all_episodes, desc="[3/6] packing language bank", unit="episode")
    for episode_index, episode_data in enumerate(iterator):
        embeddings = _load_language_list(episode_data["lang_path"])
        group_lang_ids: List[int] = []
        lang_group_id = episode_index
        lang_group_ids[episode_index] = lang_group_id

        for instruction_idx, embedding in enumerate(embeddings):
            if embedding_dim is None:
                embedding_dim = int(embedding.shape[1])
                target_dtype = _dtype_from_name(lang_dtype_name, fallback=embedding.dtype)
            if int(embedding.shape[1]) != int(embedding_dim):
                raise ValueError(
                    f"Language embedding dim mismatch in {episode_data['lang_path']}: "
                    f"expected {embedding_dim}, got {embedding.shape[1]}"
                )
            if target_dtype is not None:
                embedding = embedding.to(dtype=target_dtype)
            if current_embeddings and len(current_embeddings) >= lang_items_per_shard:
                flush_language_shard()

            lang_id = len(items)
            local_index = len(current_embeddings)
            current_embeddings.append(embedding.contiguous())
            group_lang_ids.append(lang_id)
            items.append(
                {
                    "lang_id": int(lang_id),
                    "lang_group_id": int(lang_group_id),
                    "shard_index": int(current_shard_index),
                    "local_index": int(local_index),
                    "instruction_idx": int(instruction_idx),
                }
            )

        groups.append(
            {
                "lang_group_id": int(lang_group_id),
                "episode_index": int(episode_index),
                "split": episode_data.get("split", ""),
                "task_name": episode_data["task_name"],
                "episode_name": episode_data["episode_name"],
                "lang_ids": group_lang_ids,
            }
        )

    flush_language_shard()
    _write_json_atomic(
        lang_root / "lang.json",
        {
            "format": "efficient_wam_language_bank",
            "version": 1,
            "storage": "flat_token_shards",
            "embedding_dim": int(embedding_dim or 0),
            "item_count": int(len(items)),
            "group_count": int(len(groups)),
            "groups": groups,
            "items": items,
        },
    )
    return lang_group_ids


def _episode_sample_ranges(dataset: RoboTwinTaskDataset) -> Dict[int, Dict[str, int]]:
    ranges: Dict[int, Dict[str, int]] = {}
    for sample_id, (episode_index, _condition_frame_idx) in enumerate(dataset.sample_index):
        entry = ranges.setdefault(
            int(episode_index),
            {
                "first_sample_id": int(sample_id),
                "sample_count": 0,
            },
        )
        entry["sample_count"] += 1
    return ranges


def _summarize_numbers(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _build_dataset_statistics(
    dataset: RoboTwinTaskDataset,
    ranges: Dict[int, Dict[str, int]],
) -> Dict[str, Any]:
    task_stats: Dict[str, Dict[str, Any]] = {}
    split_counts: Dict[str, int] = {}
    all_lengths: List[int] = []
    all_sample_counts: List[int] = []
    error_count = 0

    for episode_index, episode_data in enumerate(dataset.all_episodes):
        task_name = str(episode_data["task_name"])
        split = str(episode_data.get("split", ""))
        sample_count = int(ranges.get(episode_index, {"sample_count": 0})["sample_count"])
        try:
            effective_frame_count = int(dataset._effective_episode_frame_count(episode_data))
        except Exception:
            effective_frame_count = 0
            error_count += 1
        all_lengths.append(effective_frame_count)
        all_sample_counts.append(sample_count)
        split_counts[split] = split_counts.get(split, 0) + 1

        entry = task_stats.setdefault(
            task_name,
            {
                "episode_count": 0,
                "sample_count": 0,
                "splits": {},
                "_effective_frame_counts": [],
                "_sample_counts": [],
            },
        )
        entry["episode_count"] += 1
        entry["sample_count"] += sample_count
        entry["splits"][split] = int(entry["splits"].get(split, 0)) + 1
        entry["_effective_frame_counts"].append(effective_frame_count)
        entry["_sample_counts"].append(sample_count)

    tasks: Dict[str, Dict[str, Any]] = {}
    for task_name, entry in sorted(task_stats.items()):
        lengths = entry.pop("_effective_frame_counts")
        sample_counts = entry.pop("_sample_counts")
        tasks[task_name] = {
            **entry,
            "effective_frame_count": _summarize_numbers(lengths),
            "sample_count_per_episode": _summarize_numbers(sample_counts),
        }

    return {
        "task_count": int(len(tasks)),
        "episode_count": int(len(dataset.all_episodes)),
        "sample_count": int(len(dataset.sample_index)),
        "error_count": int(error_count),
        "splits": split_counts,
        "effective_frame_count": _summarize_numbers(all_lengths),
        "sample_count_per_episode": _summarize_numbers(all_sample_counts),
        "tasks": tasks,
    }


def _write_manifests_and_stats(
    dataset: RoboTwinTaskDataset,
    output_root: Path,
    *,
    lang_group_ids: Dict[int, int],
) -> Dict[str, Any]:
    ranges = _episode_sample_ranges(dataset)
    statistics = _build_dataset_statistics(dataset, ranges)
    episodes_path = output_root / "episodes.jsonl"
    samples_path = output_root / "samples.jsonl"

    fingerprint_hasher = hashlib.sha256()
    with episodes_path.open("w", encoding="utf-8") as episodes_file:
        for episode_index, episode_data in enumerate(dataset.all_episodes):
            sample_range = ranges.get(episode_index, {"first_sample_id": -1, "sample_count": 0})
            payload = {
                "episode_id": int(episode_index),
                "split": episode_data.get("split", ""),
                "task_name": episode_data["task_name"],
                "episode_name": episode_data["episode_name"],
                "first_sample_id": int(sample_range["first_sample_id"]),
                "sample_count": int(sample_range["sample_count"]),
                "lang_group_id": int(lang_group_ids[episode_index]),
            }
            line = json.dumps(payload, ensure_ascii=True, sort_keys=True)
            fingerprint_hasher.update(line.encode("utf-8"))
            fingerprint_hasher.update(b"\n")
            episodes_file.write(line + "\n")

    iterator = tqdm(
        range(len(dataset.sample_index)),
        desc="[3/6] writing sample manifest",
        unit="sample",
    )
    with samples_path.open("w", encoding="utf-8") as samples_file:
        for sample_id in iterator:
            episode_index, condition_frame_idx = dataset.sample_index[sample_id]
            episode_data = dataset.get_episode(episode_index)
            payload = {
                "sample_id": int(sample_id),
                "full_sample_id": int(sample_id),
                "episode_index": int(episode_index),
                "split": episode_data.get("split", ""),
                "task_name": episode_data["task_name"],
                "episode_name": episode_data["episode_name"],
                "condition_frame_idx": int(condition_frame_idx),
                "lang_group_id": int(lang_group_ids[episode_index]),
            }
            line = json.dumps(payload, ensure_ascii=True, sort_keys=True)
            fingerprint_hasher.update(line.encode("utf-8"))
            fingerprint_hasher.update(b"\n")
            samples_file.write(line + "\n")

    _write_json_atomic(output_root / TRAIN_DATASET_STATS, statistics)
    return {
        "ranges": ranges,
        "statistics": statistics,
        "manifest_sha256": fingerprint_hasher.hexdigest(),
    }


def _load_qpos_tensor(path: str) -> torch.Tensor:
    qpos_data = torch.load(path, map_location="cpu")
    if not isinstance(qpos_data, torch.Tensor):
        qpos_data = torch.as_tensor(qpos_data)
    return qpos_data


def _select_robot_data(
    dataset: RoboTwinTaskDataset,
    qpos_data: torch.Tensor,
    action_indices: List[int],
    initial_state_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if initial_state_idx >= len(qpos_data):
        initial_state_idx = len(qpos_data) - 1
    initial_state = qpos_data[initial_state_idx].float()
    actions = []
    for idx in action_indices:
        if idx >= len(qpos_data):
            raise IndexError(f"Action index {idx} out of bounds for qpos data length {len(qpos_data)}")
        actions.append(qpos_data[idx])
    action_sequence = torch.stack(actions).float()
    if dataset.action_normalizer.enabled:
        initial_state = dataset.action_normalizer.normalize(initial_state)
        action_sequence = dataset.action_normalizer.normalize(action_sequence)
    return initial_state.contiguous(), action_sequence.contiguous()


class _TrainSamplePrepDataset(Dataset):
    """Loads raw frames and robot tensors for final shard construction."""

    def __init__(self, dataset: RoboTwinTaskDataset, *, max_qpos_cache: int = 8):
        self.dataset = dataset
        self.max_qpos_cache = max(1, int(max_qpos_cache))
        self._qpos_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.dataset.sample_index)

    def _qpos(self, path: str) -> torch.Tensor:
        cached = self._qpos_cache.get(path)
        if cached is not None:
            self._qpos_cache.move_to_end(path)
            return cached
        tensor = _load_qpos_tensor(path)
        self._qpos_cache[path] = tensor
        self._qpos_cache.move_to_end(path)
        while len(self._qpos_cache) > self.max_qpos_cache:
            self._qpos_cache.popitem(last=False)
        return tensor

    def __getitem__(self, sample_id: int) -> Dict[str, Any]:
        sample_id = int(sample_id)
        episode_index, condition_frame_idx = self.dataset.sample_index[sample_id]
        episode_data = self.dataset.get_episode(episode_index)
        total_frames = self.dataset._effective_episode_frame_count(episode_data)
        resolved_condition_idx, video_indices, action_indices = self.dataset._calculate_sampling_indices(
            total_frames=total_frames,
            condition_frame_idx=condition_frame_idx,
        )
        sampled_frames = load_video_frames(
            episode_data["video_path"],
            [resolved_condition_idx] + video_indices,
            self.dataset.video_size,
        )
        initial_state, action_sequence = _select_robot_data(
            self.dataset,
            self._qpos(episode_data["qpos_path"]),
            action_indices,
            resolved_condition_idx,
        )
        return {
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "episode_index": torch.tensor(int(episode_index), dtype=torch.long),
            "condition_frame_idx": torch.tensor(int(resolved_condition_idx), dtype=torch.long),
            "video_indices": torch.tensor([int(value) for value in video_indices], dtype=torch.long),
            "action_indices": torch.tensor([int(value) for value in action_indices], dtype=torch.long),
            "lang_group_id": torch.tensor(int(episode_index), dtype=torch.long),
            "initial_state": initial_state,
            "action_sequence": action_sequence,
            "first_frame": sampled_frames[0],
            "video_frames": sampled_frames[1:],
            "task_name": episode_data["task_name"],
        }


def _collate_prep_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _sample_ids_sha256(sample_ids: torch.Tensor) -> str:
    ids = sample_ids.detach().cpu().to(dtype=torch.int64).contiguous()
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()


def _validate_shard_tensors(shard_index: int, sample_start: int, tensors: Dict[str, torch.Tensor]) -> None:
    if not tensors:
        raise ValueError(f"Shard {shard_index} has no tensors")
    sample_count = int(next(iter(tensors.values())).shape[0])
    if sample_count <= 0:
        raise ValueError(f"Shard {shard_index} is empty")
    for key, value in tensors.items():
        if int(value.shape[0]) != sample_count:
            raise ValueError(
                f"Shard {shard_index} tensor {key!r} has first dim {value.shape[0]}, expected {sample_count}"
            )
    sample_ids = tensors["sample_ids"].to(dtype=torch.long)
    expected = torch.arange(sample_start, sample_start + sample_count, dtype=torch.long)
    if not torch.equal(sample_ids.cpu(), expected):
        raise ValueError(f"Shard {shard_index} sample_ids are not contiguous from {sample_start}")


def _write_train_shard(
    output_root: Path,
    shard_index: int,
    sample_start: int,
    tensors: Dict[str, torch.Tensor],
) -> int:
    try:
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("Packing train dataset requires safetensors to be installed") from exc

    _validate_shard_tensors(shard_index, sample_start, tensors)
    shard_path = output_root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.safetensors"
    shard_meta_path = output_root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_shard_path = shard_path.with_name(f"{shard_path.name}.tmp.{os.getpid()}")
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(tmp_shard_path))
    tmp_shard_path.replace(shard_path)
    size_bytes = int(shard_path.stat().st_size)
    sample_count = int(tensors["sample_ids"].shape[0])
    _write_json_atomic(
        shard_meta_path,
        {
            "shard_index": int(shard_index),
            "sample_start": int(sample_start),
            "sample_count": int(sample_count),
            "sample_ids_sha256": _sample_ids_sha256(tensors["sample_ids"]),
            "size_bytes": int(size_bytes),
            "tensor_bytes": int(sum(_tensor_bytes(value) for value in tensors.values())),
            "keys": {
                key: {
                    "shape": [int(dim) for dim in value.shape],
                    "dtype": str(value.dtype).replace("torch.", ""),
                }
                for key, value in tensors.items()
            },
        },
    )
    return size_bytes


def _progress_path(output_root: Path, rank: int) -> Path:
    return output_root / f".build_progress_rank{int(rank):03d}.json"


def _write_progress(
    output_root: Path,
    rank: int,
    *,
    samples_done: int,
    shards_done: int,
    written_bytes: int,
    current_task: str = "",
) -> None:
    _write_json_atomic(
        _progress_path(output_root, rank),
        {
            "rank": int(rank),
            "samples_done": int(samples_done),
            "shards_done": int(shards_done),
            "written_bytes": int(written_bytes),
            "current_task": str(current_task),
            "updated_at_unix": time.time(),
        },
    )


def _read_progress_totals(output_root: Path, world_size: int) -> Dict[str, Any]:
    totals: Dict[str, Any] = {
        "samples_done": 0,
        "shards_done": 0,
        "written_bytes": 0,
        "current_tasks": [],
    }
    for rank in range(world_size):
        path = _progress_path(output_root, rank)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        totals["samples_done"] += int(payload.get("samples_done", 0))
        totals["shards_done"] += int(payload.get("shards_done", 0))
        totals["written_bytes"] += int(payload.get("written_bytes", 0))
        current_task = str(payload.get("current_task", ""))
        if current_task:
            totals["current_tasks"].append(current_task)
    return totals


def _cleanup_progress_files(output_root: Path, world_size: int) -> None:
    for rank in range(world_size):
        path = _progress_path(output_root, rank)
        if path.exists():
            path.unlink()
        for tmp_path in output_root.glob(f"{path.name}.tmp.*"):
            tmp_path.unlink()


def _owned_sample_indices(sample_count: int, shard_size: int, rank: int, world_size: int) -> List[int]:
    indices: List[int] = []
    for shard_start in range(0, sample_count, shard_size):
        shard_index = shard_start // shard_size
        if shard_index % world_size != rank:
            continue
        shard_end = min(shard_start + shard_size, sample_count)
        indices.extend(range(shard_start, shard_end))
    return indices


def _write_main_shards(
    dataset: RoboTwinTaskDataset,
    output_root: Path,
    *,
    shard_size: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> Dict[str, int]:
    sample_count = int(len(dataset.sample_index))
    shard_count = (sample_count + shard_size - 1) // shard_size
    owned_indices = _owned_sample_indices(sample_count, shard_size, rank, world_size)
    _write_progress(output_root, rank, samples_done=0, shards_done=0, written_bytes=0)

    progress_bar = None
    last_global_done = 0
    if rank == 0:
        progress_bar = tqdm(
            total=sample_count,
            desc="[4/6] encoding videos + packing shards",
            unit="sample",
        )

    def refresh_progress(force: bool = False) -> None:
        nonlocal last_global_done
        if rank != 0 or progress_bar is None:
            return
        totals = _read_progress_totals(output_root, world_size)
        global_done = min(int(totals["samples_done"]), sample_count)
        delta = global_done - last_global_done
        if delta > 0:
            progress_bar.update(delta)
            last_global_done = global_done
        if force or delta > 0:
            progress_bar.set_postfix(
                {
                    "shards": f"{int(totals['shards_done'])}/{shard_count}",
                    "GB": f"{float(totals['written_bytes']) / (1024 ** 3):.2f}",
                    "rank": f"0/{world_size}",
                }
            )

    encoded = 0
    written_shards = 0
    written_bytes = 0
    verified_condition = False
    current_task = ""

    if owned_indices:
        logger.info(
            "Rank %d/%d packing %d samples across shard_size=%d on %s",
            rank,
            world_size,
            len(owned_indices),
            shard_size,
            args.device,
        )
        prep_dataset = _TrainSamplePrepDataset(dataset, max_qpos_cache=int(args.qpos_cache_size))
        loader_kwargs: Dict[str, Any] = {
            "batch_size": int(args.batch_size),
            "shuffle": False,
            "num_workers": int(args.num_workers),
            "pin_memory": bool(args.pin_memory),
            "collate_fn": _collate_prep_batch,
        }
        if int(args.num_workers) > 0:
            loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
            loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader = DataLoader(
            Subset(prep_dataset, owned_indices),
            **loader_kwargs,
        )
        vae = Wan2_2_VAE(vae_pth=vae_path_from_config(args.config_payload), device=args.device)
        dtype = dtype_from_config(args.config_payload)
        reuse_condition_latent = reuse_condition_latent_from_clean(args.config_payload)
        future_video_size = future_video_size_from_config(args.config_payload)
        multiscale_latents = future_video_size is not None

        current_shard_index: Optional[int] = None
        current_sample_start: Optional[int] = None
        current: Dict[str, List[torch.Tensor]] = {}
        last_progress_write = time.monotonic()

        def reset_current() -> None:
            nonlocal current
            current = {
                "sample_ids": [],
                "episode_indices": [],
                "condition_frame_indices": [],
                "video_indices": [],
                "action_indices": [],
                "lang_group_ids": [],
                "initial_states": [],
                "action_sequences": [],
                "condition_latents": [],
            }
            current["future_latents" if multiscale_latents else "clean_latents"] = []

        def flush_current() -> None:
            nonlocal current_shard_index, current_sample_start, written_shards, written_bytes
            if current_shard_index is None or current_sample_start is None:
                return
            if not current["sample_ids"]:
                return
            tensors = {
                key: torch.stack(values, dim=0).contiguous()
                for key, values in current.items()
            }
            size_bytes = _write_train_shard(
                output_root,
                current_shard_index,
                current_sample_start,
                tensors,
            )
            written_shards += 1
            written_bytes += size_bytes
            current_shard_index = None
            current_sample_start = None
            reset_current()

        reset_current()
        for batch in loader:
            batch_count = int(batch["first_frame"].shape[0])
            current_task = str(batch.get("task_name", [""])[-1])
            first_frame = batch["first_frame"].to(args.device, dtype=dtype, non_blocking=True)
            video_frames = batch["video_frames"].to(args.device, dtype=dtype, non_blocking=True)
            first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
            video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
            full_video = torch.cat([first_frame_norm, video_normalized], dim=2)

            _cuda_synchronize(args.device)
            with torch.no_grad():
                if multiscale_latents:
                    condition_latent = vae.encode(first_frame_norm)
                    future_full_video = _build_low_resolution_full_video(
                        first_frame,
                        video_frames,
                        future_video_size,
                    )
                    future_full_latent = vae.encode(future_full_video)
                    future_latent = future_full_latent[:, :, 1:].contiguous()
                    clean_latent = None
                else:
                    clean_latent = vae.encode(full_video)
                if not multiscale_latents and reuse_condition_latent:
                    condition_latent = clean_latent[:, :, 0:1]
                    if args.verify_condition_latent and not verified_condition:
                        encoded_condition_latent = vae.encode(first_frame_norm)
                        max_abs_diff = (condition_latent.float() - encoded_condition_latent.float()).abs().max()
                        logger.info(
                            "Condition latent reuse check rank=%d max_abs_diff=%.6g",
                            rank,
                            float(max_abs_diff.item()),
                        )
                        verified_condition = True
                elif not multiscale_latents:
                    condition_latent = vae.encode(first_frame_norm)
            _cuda_synchronize(args.device)

            condition_latent_cpu = condition_latent.detach().cpu()
            latent_cpu = (
                future_latent.detach().cpu()
                if multiscale_latents
                else clean_latent.detach().cpu()
            )
            latent_key = "future_latents" if multiscale_latents else "clean_latents"
            for row in range(batch_count):
                sample_id = int(batch["sample_id"][row].item())
                shard_index = sample_id // shard_size
                shard_sample_start = shard_index * shard_size
                if current_shard_index is not None and shard_index != current_shard_index:
                    flush_current()
                if current_shard_index is None:
                    current_shard_index = shard_index
                    current_sample_start = shard_sample_start

                current["sample_ids"].append(batch["sample_id"][row].to(dtype=torch.long).cpu())
                current["episode_indices"].append(batch["episode_index"][row].to(dtype=torch.long).cpu())
                current["condition_frame_indices"].append(batch["condition_frame_idx"][row].to(dtype=torch.long).cpu())
                current["video_indices"].append(batch["video_indices"][row].to(dtype=torch.long).cpu())
                current["action_indices"].append(batch["action_indices"][row].to(dtype=torch.long).cpu())
                current["lang_group_ids"].append(batch["lang_group_id"][row].to(dtype=torch.long).cpu())
                current["initial_states"].append(batch["initial_state"][row].cpu())
                current["action_sequences"].append(batch["action_sequence"][row].cpu())
                current[latent_key].append(latent_cpu[row])
                current["condition_latents"].append(condition_latent_cpu[row])

            encoded += batch_count
            now = time.monotonic()
            if now - last_progress_write >= max(0.5, float(args.progress_interval_seconds)):
                _write_progress(
                    output_root,
                    rank,
                    samples_done=encoded,
                    shards_done=written_shards,
                    written_bytes=written_bytes,
                    current_task=current_task,
                )
                refresh_progress()
                last_progress_write = now
        flush_current()

    _write_progress(
        output_root,
        rank,
        samples_done=encoded,
        shards_done=written_shards,
        written_bytes=written_bytes,
        current_task=current_task,
    )
    refresh_progress(force=True)

    if rank == 0:
        while True:
            totals = _read_progress_totals(output_root, world_size)
            refresh_progress(force=True)
            if int(totals["samples_done"]) >= sample_count:
                break
            time.sleep(max(0.5, float(args.progress_interval_seconds)))
        if progress_bar is not None:
            progress_bar.close()

    _barrier()
    if rank == 0:
        totals = _read_progress_totals(output_root, world_size)
        return {
            "sample_count": int(sample_count),
            "shard_count": int(shard_count),
            "written_bytes": int(totals["written_bytes"]),
        }
    return {
        "sample_count": int(sample_count),
        "shard_count": int(shard_count),
        "written_bytes": int(written_bytes),
    }


def _verify_generated_shards(output_root: Path, *, sample_count: int, shard_count: int, shard_size: int) -> None:
    required_keys = {
        "sample_ids",
        "episode_indices",
        "condition_frame_indices",
        "video_indices",
        "action_indices",
        "lang_group_ids",
        "initial_states",
        "action_sequences",
        "condition_latents",
    }
    latent_keys = {"clean_latents", "future_latents"}
    covered = 0
    for shard_index in tqdm(range(shard_count), desc="[5/6] verifying shards", unit="shard"):
        meta_path = output_root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.json"
        data_path = output_root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.safetensors"
        if not meta_path.exists() or not data_path.exists():
            raise FileNotFoundError(f"Missing generated train shard: {data_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample_start = int(meta["sample_start"])
        shard_sample_count = int(meta["sample_count"])
        expected_start = shard_index * shard_size
        expected_count = min(shard_size, sample_count - expected_start)
        if sample_start != expected_start or shard_sample_count != expected_count:
            raise RuntimeError(
                f"Shard {shard_index} range mismatch: got start={sample_start} count={shard_sample_count}, "
                f"expected start={expected_start} count={expected_count}"
            )
        keys = set(meta.get("keys", {}).keys())
        missing = required_keys - keys
        if missing:
            raise RuntimeError(f"Shard {shard_index} metadata missing keys: {sorted(missing)}")
        present_latent_keys = keys.intersection(latent_keys)
        if not present_latent_keys:
            raise RuntimeError(f"Shard {shard_index} metadata missing one latent key from {sorted(latent_keys)}")
        for key in required_keys | present_latent_keys:
            shape = meta["keys"][key]["shape"]
            if int(shape[0]) != shard_sample_count:
                raise RuntimeError(
                    f"Shard {shard_index} key {key} first dim {shape[0]} != {shard_sample_count}"
                )
        covered += shard_sample_count
    if covered != sample_count:
        raise RuntimeError(f"Generated shards cover {covered} samples, expected {sample_count}")


def _prepare_output_root(output_root: Path, rank: int, *, overwrite: bool) -> Path:
    temp_root = output_root.with_name(f"{output_root.name}.tmp_build")
    if rank == 0:
        if output_root.exists() and not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Pass --overwrite to replace it.")
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)
    _barrier()
    return temp_root


def _finalize_output_root(temp_root: Path, output_root: Path, rank: int, *, overwrite: bool) -> None:
    _barrier()
    if rank == 0:
        if output_root.exists():
            if not overwrite:
                raise FileExistsError(f"Output root already exists: {output_root}")
            shutil.rmtree(output_root)
        temp_root.replace(output_root)
    _barrier()


def run(args: argparse.Namespace) -> None:
    config = args.config_payload if hasattr(args, "config_payload") else load_yaml_config(args.config)
    args.config_payload = config
    build_cfg = dict(config.get("build") or {})
    language_cfg = dict(config.get("language") or {})
    latent_cfg = dict(config.get("latent") or {})

    args.device = args.device or build_cfg.get("device", "cuda")
    args.batch_size = int(args.batch_size if args.batch_size is not None else build_cfg.get("batch_size", 64))
    args.num_workers = int(args.num_workers if args.num_workers is not None else build_cfg.get("num_workers", 4))
    args.pin_memory = bool(build_cfg.get("pin_memory", True))
    args.persistent_workers = bool(build_cfg.get("persistent_workers", True))
    args.prefetch_factor = int(build_cfg.get("prefetch_factor", 2))
    args.qpos_cache_size = int(build_cfg.get("qpos_cache_size", 8))
    args.shard_size = int(args.shard_size if args.shard_size is not None else latent_cfg.get("shard_size", 4096))
    args.lang_items_per_shard = int(
        args.lang_items_per_shard
        if args.lang_items_per_shard is not None
        else language_cfg.get("items_per_shard", 64)
    )
    args.lang_dtype = args.lang_dtype or language_cfg.get("dtype", "keep")
    args.log_level = args.log_level or build_cfg.get("log_level", "INFO")
    args.dist_timeout_minutes = int(
        args.dist_timeout_minutes
        if args.dist_timeout_minutes is not None
        else build_cfg.get("dist_timeout_minutes", 240)
    )
    args.progress_interval_seconds = float(
        args.progress_interval_seconds
        if args.progress_interval_seconds is not None
        else build_cfg.get("progress_interval_seconds", 2.0)
    )
    if not args.verify_condition_latent:
        args.verify_condition_latent = bool(build_cfg.get("verify_condition_latent", False))

    rank, world_size, device = _distributed_info(str(args.device), args.dist_timeout_minutes)
    args.device = device
    setup_logging(args.log_level, rank=rank)

    output_root_value = args.output_root or output_root_from_config(config)
    if not output_root_value:
        raise ValueError("Provide --output-root or set output_root in the dataset build config")
    output_root = Path(output_root_value).expanduser()
    temp_root = _prepare_output_root(output_root, rank, overwrite=bool(args.overwrite))

    _phase(rank, 1, 6, "preparing output and action stats")
    if rank == 0:
        _prepare_action_stats(config, temp_root, args)
    _barrier()

    action_stats_path = temp_root / TRAIN_DATASET_ACTION_STATS
    action_stats_value = str(action_stats_path) if action_stats_path.exists() else None

    _phase(rank, 2, 6, "scanning preprocessed RoboTwin episodes")
    dataset = build_robotwin_dataset(config, action_stats_path=action_stats_value)
    if args.shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {args.shard_size}")

    if rank == 0:
        logger.info("Final train dataset output: %s", output_root)
        logger.info("Temporary build root      : %s", temp_root)
        logger.info("Samples=%d episodes=%d shard_size=%d world_size=%d", len(dataset), len(dataset.all_episodes), args.shard_size, world_size)
        logger.info("VAE: %s", vae_path_from_config(config))
        logger.info("VAE config: %s", compact_vae_config(config))

    _phase(rank, 3, 6, "packing language bank and manifests")
    if rank == 0:
        lang_group_ids = _write_language_bank(
            dataset,
            temp_root,
            lang_items_per_shard=max(1, int(args.lang_items_per_shard)),
            lang_dtype_name=args.lang_dtype,
        )
        manifest_info = _write_manifests_and_stats(
            dataset,
            temp_root,
            lang_group_ids=lang_group_ids,
        )
    else:
        manifest_info = {}
    _barrier()

    _phase(rank, 4, 6, "encoding videos and writing final train shards")
    shard_info = _write_main_shards(
        dataset,
        temp_root,
        shard_size=int(args.shard_size),
        args=args,
        rank=rank,
        world_size=world_size,
    )
    _barrier()

    if rank == 0:
        _phase(rank, 5, 6, "verifying generated dataset")
        _verify_generated_shards(
            temp_root,
            sample_count=int(shard_info["sample_count"]),
            shard_count=int(shard_info["shard_count"]),
            shard_size=int(args.shard_size),
        )
        statistics = manifest_info["statistics"]
        train_dataset_cfg = _path_independent_training_dataset_config(training_dataset_config(config))
        sampling_cfg = dict(train_dataset_cfg.get("sampling") or {"mode": "random_full_epoch"})
        language_policy = str(
            train_dataset_cfg.get("language_policy")
            or config.get("language", {}).get("policy", "random_per_access")
        )
        fingerprint_payload = {
            "format": TRAIN_DATASET_FORMAT,
            "version": TRAIN_DATASET_VERSION,
            "manifest_sha256": manifest_info["manifest_sha256"],
            "sample_count": int(shard_info["sample_count"]),
            "episode_count": int(len(dataset.all_episodes)),
            "shard_size": int(args.shard_size),
            "data": {
                "num_video_frames": int(dataset.num_video_frames),
                "video_size": [int(v) for v in dataset.video_size],
                "global_downsample_rate": int(dataset.global_downsample_rate),
                "video_action_freq_ratio": int(dataset.video_action_freq_ratio),
                "action_chunk_size": int(dataset.action_chunk_size),
                "action_normalization": _path_independent_action_normalization(dataset.action_normalization_config),
            },
            "latent": {
                "precision": str(compact_vae_config(config).get("precision", "bfloat16")),
                "reuse_condition_latent_from_clean": bool(reuse_condition_latent_from_clean(config)),
                "future_video_size": (
                    [int(value) for value in future_video_size_from_config(config)]
                    if future_video_size_from_config(config) is not None
                    else None
                ),
            },
            "language": {
                "dtype": str(args.lang_dtype),
                "items_per_shard": int(args.lang_items_per_shard),
            },
        }
        metadata = {
            "format": TRAIN_DATASET_FORMAT,
            "version": TRAIN_DATASET_VERSION,
            "sample_layout": "exhaustive",
            "sample_count": int(shard_info["sample_count"]),
            "episode_count": int(len(dataset.all_episodes)),
            "shard_count": int(shard_info["shard_count"]),
            "shard_size": int(args.shard_size),
            "dataset_fingerprint": _hash_json_payload(fingerprint_payload),
            "manifest_sha256": manifest_info["manifest_sha256"],
            "created_at_unix": time.time(),
            "written_bytes": int(shard_info["written_bytes"]),
            "statistics": {
                "path": TRAIN_DATASET_STATS,
                "task_count": int(statistics["task_count"]),
                "episode_count": int(statistics["episode_count"]),
                "sample_count": int(statistics["sample_count"]),
                "error_count": int(statistics["error_count"]),
                "effective_frame_count": statistics["effective_frame_count"],
                "sample_count_per_episode": statistics["sample_count_per_episode"],
            },
            "storage": {
                "shard_dir": TRAIN_DATASET_SHARD_DIR,
                "shard_size": int(args.shard_size),
            },
            "training_dataset": train_dataset_cfg,
            "data": fingerprint_payload["data"],
            "latent": fingerprint_payload["latent"],
            "language": {
                "storage": "bank",
                "policy": language_policy,
                "dtype": str(args.lang_dtype),
            },
            "sampling": sampling_cfg,
            "runtime_validation": {
                "default": False,
                "notes": "Conversion validates shard alignment; training __getitem__ skips per-sample checks unless validate_runtime=true.",
            },
        }
        _phase(rank, 6, 6, "writing dataset metadata")
        _write_json_atomic(temp_root / TRAIN_DATASET_METADATA, metadata)
        _cleanup_progress_files(temp_root, world_size)
        logger.info("Packed train dataset: samples=%d shards=%d bytes=%.2f GB", shard_info["sample_count"], shard_info["shard_count"], shard_info["written_bytes"] / (1024 ** 3))

    _finalize_output_root(temp_root, output_root, rank, overwrite=bool(args.overwrite))
    if rank == 0:
        logger.info("Final dataset is ready: %s", output_root)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the final EfficientWAM train dataset")
    parser.add_argument("--config", type=str, default="configs/robotwin/train_dataset.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lang-items-per-shard", type=int, default=None)
    parser.add_argument("--lang-dtype", choices=["keep", "bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--verify-condition-latent", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--dist-timeout-minutes", type=int, default=None)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    run(_build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
