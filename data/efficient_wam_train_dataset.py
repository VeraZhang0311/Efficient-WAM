from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import torch
from torch.utils.data import Dataset, Sampler


TRAIN_DATASET_FORMAT = "efficient_wam_train_dataset"
TRAIN_DATASET_VERSION = 2
TRAIN_DATASET_METADATA = "dataset.json"
TRAIN_DATASET_STATS = "stats.json"
TRAIN_DATASET_ACTION_STATS = "action_stats.json"
TRAIN_DATASET_SHARD_DIR = "shards"
TRAIN_DATASET_LANG_DIR = "lang"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class _SafetensorShardCache:
    """Small per-worker LRU cache for safetensors shard handles."""

    def __init__(self, max_open_shards: int = 8):
        self.max_open_shards = max(1, int(max_open_shards))
        self._handles: OrderedDict[int, Any] = OrderedDict()

    def get(self, shard_index: int, path: Path):
        shard_index = int(shard_index)
        handle = self._handles.get(shard_index)
        if handle is not None:
            self._handles.move_to_end(shard_index)
            return handle

        try:
            from safetensors import safe_open
        except Exception as exc:  # pragma: no cover - environment dependency
            raise RuntimeError("EfficientWAM train dataset requires safetensors to be installed") from exc

        handle = safe_open(str(path), framework="pt", device="cpu")
        self._handles[shard_index] = handle
        self._handles.move_to_end(shard_index)
        while len(self._handles) > self.max_open_shards:
            self._handles.popitem(last=False)
        return handle


class PerEpisodeRandomSampler(Sampler[int]):
    """Sample episode-uniform random items for a fixed-size epoch."""

    def __init__(
        self,
        episodes: List[Dict[str, int]],
        *,
        samples_per_episode: int,
        seed: int = 0,
    ):
        self.episodes = [
            {
                "episode_id": int(episode["episode_id"]),
                "first_sample_id": int(episode["first_sample_id"]),
                "sample_count": int(episode["sample_count"]),
            }
            for episode in episodes
            if int(episode.get("sample_count", 0)) > 0
        ]
        if not self.episodes:
            raise ValueError("per_episode_random sampling requires at least one episode with samples")
        self.samples_per_episode = int(samples_per_episode)
        if self.samples_per_episode <= 0:
            raise ValueError(f"samples_per_episode must be positive, got {samples_per_episode}")
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.episodes) * self.samples_per_episode

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        total_samples = len(self.episodes) * self.samples_per_episode
        for _ in range(total_samples):
            episode = rng.choice(self.episodes)
            first_sample_id = int(episode["first_sample_id"])
            sample_count = int(episode["sample_count"])
            offset = rng.randrange(sample_count)
            yield first_sample_id + int(offset)


class RandomFullEpochSampler(Sampler[int]):
    """Visit every sample once per epoch with a fresh shard-aware shuffle."""

    def __init__(self, sample_count: int, *, shard_size: int, seed: int = 0):
        self.sample_count = int(sample_count)
        self.shard_size = max(1, int(shard_size))
        if self.sample_count <= 0:
            raise ValueError(f"sample_count must be positive, got {sample_count}")
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.sample_count

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shard_count = (self.sample_count + self.shard_size - 1) // self.shard_size
        shard_indices = list(range(shard_count))
        rng.shuffle(shard_indices)
        for shard_index in shard_indices:
            start = shard_index * self.shard_size
            end = min(start + self.shard_size, self.sample_count)
            sample_ids = list(range(start, end))
            rng.shuffle(sample_ids)
            yield from sample_ids


class EfficientWAMTrainDataset(Dataset):
    """Final EfficientWAM train dataset backed by one training shard stream.

    Conversion-time checks guarantee that each sample's latents, actions, robot
    state, frame indices, and language group id were written together. Runtime
    reading stays lean: no raw RoboTwin scan, no sidecar join, and no per-sample
    validation unless ``validate_runtime`` is explicitly enabled for debugging.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        sampling: Optional[Mapping[str, Any]] = None,
        language_policy: Optional[str] = None,
        max_open_shards: Optional[int] = None,
        include_metadata: Optional[bool] = None,
        validate_runtime: Optional[bool] = None,
    ):
        self.root = Path(root).expanduser()
        self.metadata = _read_json(self.root / TRAIN_DATASET_METADATA)
        if self.metadata.get("format") != TRAIN_DATASET_FORMAT:
            raise ValueError(
                f"Unsupported train dataset format at {self.root}: {self.metadata.get('format')!r}"
            )
        version = int(self.metadata.get("version", -1))
        if version != TRAIN_DATASET_VERSION:
            raise ValueError(
                f"Unsupported train dataset version at {self.root}: {version}; "
                f"expected {TRAIN_DATASET_VERSION}"
            )

        self.sample_count = int(self.metadata["sample_count"])
        self.training_dataset_config = dict(self.metadata.get("training_dataset", {}) or {})
        self.shard_size = int(self.metadata["shard_size"])
        self.storage = dict(self.metadata.get("storage", {}) or {})
        self.shard_dir = str(self.storage.get("shard_dir", TRAIN_DATASET_SHARD_DIR))
        self.sampling = dict(
            sampling
            or self.training_dataset_config.get("sampling")
            or self.metadata.get("sampling", {})
            or {}
        )
        self.sampling_mode = str(self.sampling.get("mode", "random_full_epoch"))
        self.language_policy = str(
            language_policy
            or self.training_dataset_config.get("language_policy")
            or self.sampling.get("language_policy")
            or self.metadata.get("language", {}).get("policy", "random_per_access")
        )
        if include_metadata is None:
            include_metadata = self.training_dataset_config.get("include_metadata", False)
        if validate_runtime is None:
            validate_runtime = self.training_dataset_config.get("validate_runtime", False)
        if max_open_shards is None:
            max_open_shards = int(self.training_dataset_config.get("max_open_shards", 8))
        self.include_metadata = bool(include_metadata)
        self.validate_runtime = bool(validate_runtime)

        stats_path = self.root / TRAIN_DATASET_STATS
        self.statistics = _read_json(stats_path) if stats_path.exists() else dict(self.metadata.get("statistics", {}) or {})
        self.episodes = list(_iter_jsonl(self.root / "episodes.jsonl"))
        self.episodes_with_samples: List[Dict[str, int]] = []
        for episode in self.episodes:
            sample_count = int(episode.get("sample_count", 0))
            if sample_count <= 0:
                continue
            first_sample_id = episode.get("first_sample_id", episode.get("first_full_sample_id"))
            if first_sample_id is None:
                raise ValueError(f"Episode row is missing first_sample_id: {episode}")
            self.episodes_with_samples.append(
                {
                    "episode_id": int(episode["episode_id"]),
                    "first_sample_id": int(first_sample_id),
                    "sample_count": sample_count,
                }
            )

        lang_root = self.root / TRAIN_DATASET_LANG_DIR
        self.lang_metadata = _read_json(lang_root / "lang.json")
        self.lang_groups: Dict[int, List[int]] = {
            int(group["lang_group_id"]): [int(lang_id) for lang_id in group["lang_ids"]]
            for group in self.lang_metadata.get("groups", [])
        }
        self.lang_items: Dict[int, Dict[str, int]] = {
            int(item["lang_id"]): {
                "shard_index": int(item["shard_index"]),
                "local_index": int(item["local_index"]),
                "instruction_idx": int(item.get("instruction_idx", 0)),
            }
            for item in self.lang_metadata.get("items", [])
        }

        self._data_shards = _SafetensorShardCache(max_open_shards=max_open_shards)
        self._lang_shards = _SafetensorShardCache(max_open_shards=max_open_shards)
        self._lang_offsets: Dict[int, torch.Tensor] = {}
        self._samples_metadata: Optional[List[Dict[str, Any]]] = None
        if self.include_metadata:
            self._samples_metadata = list(_iter_jsonl(self.root / "samples.jsonl"))

    @property
    def dataloader_shuffle(self) -> bool:
        return self.make_sampler() is None

    def __len__(self) -> int:
        return self.sample_count

    def make_sampler(self) -> Optional[Sampler[int]]:
        mode = self.sampling_mode
        if mode == "per_episode_random":
            return PerEpisodeRandomSampler(
                self.episodes_with_samples,
                samples_per_episode=int(self.sampling.get("samples_per_episode", 10)),
                seed=int(self.sampling.get("seed", 0)),
            )
        if mode == "random_full_epoch":
            return RandomFullEpochSampler(
                self.sample_count,
                shard_size=int(self.sampling.get("shuffle_shard_size", self.shard_size)),
                seed=int(self.sampling.get("seed", 0)),
            )
        raise ValueError(f"Unsupported EfficientWAM train sampling mode: {mode!r}")

    def _data_shard_path(self, shard_index: int) -> Path:
        return self.root / self.shard_dir / f"shard_{int(shard_index):06d}.safetensors"

    def _lang_shard_path(self, shard_index: int) -> Path:
        return (
            self.root
            / TRAIN_DATASET_LANG_DIR
            / TRAIN_DATASET_SHARD_DIR
            / f"shard_{int(shard_index):06d}.safetensors"
        )

    def _load_language_embedding(self, lang_group_id: int, sample_id: int) -> torch.Tensor:
        lang_ids = self.lang_groups.get(int(lang_group_id))
        if not lang_ids:
            raise KeyError(f"Missing language group {lang_group_id}")

        if self.language_policy == "first":
            lang_id = lang_ids[0]
        elif self.language_policy == "deterministic":
            lang_id = lang_ids[int(sample_id) % len(lang_ids)]
        else:
            lang_id = random.choice(lang_ids)

        item = self.lang_items[int(lang_id)]
        shard_index = int(item["shard_index"])
        local_index = int(item["local_index"])
        shard = self._lang_shards.get(shard_index, self._lang_shard_path(shard_index))
        offsets = self._lang_offsets.get(shard_index)
        if offsets is None:
            offsets = shard.get_tensor("token_offsets").to(dtype=torch.long)
            self._lang_offsets[shard_index] = offsets
        start = int(offsets[local_index].item())
        end = int(offsets[local_index + 1].item())
        return shard.get_slice("token_data")[start:end].clone()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_id = int(index)
        if sample_id < 0 or sample_id >= self.sample_count:
            raise IndexError(f"sample_id {sample_id} out of range for {self.sample_count}")

        shard_index = sample_id // self.shard_size
        local_index = sample_id % self.shard_size
        shard = self._data_shards.get(shard_index, self._data_shard_path(shard_index))
        shard_keys = set(shard.keys())

        if self.validate_runtime:
            stored_sample_id = int(shard.get_slice("sample_ids")[local_index].item())
            if stored_sample_id != sample_id:
                raise RuntimeError(
                    f"Train shard sample_id mismatch: requested {sample_id}, stored {stored_sample_id}"
                )

        lang_group_id = int(shard.get_slice("lang_group_ids")[local_index].item())
        sample: Dict[str, Any] = {
            "condition_latent": shard.get_slice("condition_latents")[local_index].clone(),
            "initial_state": shard.get_slice("initial_states")[local_index].clone(),
            "action_sequence": shard.get_slice("action_sequences")[local_index].clone(),
            "condition_frame_idx": int(shard.get_slice("condition_frame_indices")[local_index].item()),
            "video_indices": shard.get_slice("video_indices")[local_index].clone().tolist(),
            "action_indices": shard.get_slice("action_indices")[local_index].clone().tolist(),
            "full_sample_id": sample_id,
            "sample_id": sample_id,
            "episode_index": int(shard.get_slice("episode_indices")[local_index].item()),
            "source_sample_index": sample_id,
            "lang_group_id": lang_group_id,
        }
        if "future_latents" in shard_keys:
            sample["future_latent"] = shard.get_slice("future_latents")[local_index].clone()
        else:
            sample["clean_latent"] = shard.get_slice("clean_latents")[local_index].clone()
        sample["language_embedding"] = self._load_language_embedding(lang_group_id, sample_id)

        if self._samples_metadata is not None:
            sample.update(self._samples_metadata[sample_id])

        return sample
