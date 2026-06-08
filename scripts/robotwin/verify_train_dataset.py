"""Verify the final EfficientWAM train dataset structure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

project_root = Path(__file__).resolve().parents[2]
if str(project_root.resolve()) not in sys.path:
    sys.path.insert(0, str(project_root.resolve()))

from data.efficient_wam_train_dataset import (  # noqa: E402
    TRAIN_DATASET_FORMAT,
    TRAIN_DATASET_LANG_DIR,
    TRAIN_DATASET_METADATA,
    TRAIN_DATASET_SHARD_DIR,
    TRAIN_DATASET_VERSION,
)


REQUIRED_SHARD_KEYS = {
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
LATENT_SHARD_KEYS = {"clean_latents", "future_latents"}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _sha256_int64_tensor(tensor) -> str:
    ids = tensor.detach().cpu().to(dtype=tensor.dtype).contiguous()
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()


def _verify_metadata(root: Path) -> Dict[str, Any]:
    metadata_path = root / TRAIN_DATASET_METADATA
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {TRAIN_DATASET_METADATA}: {metadata_path}")
    metadata = _read_json(metadata_path)
    if metadata.get("format") != TRAIN_DATASET_FORMAT:
        raise RuntimeError(f"Unsupported dataset format: {metadata.get('format')!r}")
    if int(metadata.get("version", -1)) != TRAIN_DATASET_VERSION:
        raise RuntimeError(
            f"Unsupported dataset version: {metadata.get('version')!r}; expected {TRAIN_DATASET_VERSION}"
        )
    for name in ["episodes.jsonl", "samples.jsonl", "stats.json"]:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
    action_norm = metadata.get("data", {}).get("action_normalization", {})
    if isinstance(action_norm, dict) and action_norm.get("enabled") and not (root / "action_stats.json").exists():
        raise FileNotFoundError(f"Missing action stats: {root / 'action_stats.json'}")
    if not (root / TRAIN_DATASET_LANG_DIR / "lang.json").exists():
        raise FileNotFoundError(f"Missing language metadata: {root / TRAIN_DATASET_LANG_DIR / 'lang.json'}")
    return metadata


def _verify_episodes(root: Path, metadata: Dict[str, Any]) -> None:
    episodes = list(_iter_jsonl(root / "episodes.jsonl"))
    if len(episodes) != int(metadata["episode_count"]):
        raise RuntimeError(f"Episode count mismatch: {len(episodes)} != {metadata['episode_count']}")

    covered = 0
    last_end = 0
    for episode in episodes:
        sample_count = int(episode.get("sample_count", 0))
        if sample_count <= 0:
            continue
        first_sample_id = int(episode["first_sample_id"])
        if first_sample_id != last_end:
            raise RuntimeError(
                f"Episode sample ranges are not contiguous at episode {episode.get('episode_id')}: "
                f"first_sample_id={first_sample_id}, expected={last_end}"
            )
        last_end = first_sample_id + sample_count
        covered += sample_count
    if covered != int(metadata["sample_count"]):
        raise RuntimeError(f"Episode ranges cover {covered} samples, expected {metadata['sample_count']}")


def _verify_shard(root: Path, shard_index: int, *, shard_size: int, sample_count: int) -> int:
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("verify_train_dataset.py requires safetensors to be installed") from exc

    meta_path = root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.json"
    data_path = root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.safetensors"
    if not meta_path.exists() or not data_path.exists():
        raise FileNotFoundError(f"Missing shard files for shard {shard_index}: {data_path}")

    meta = _read_json(meta_path)
    expected_start = shard_index * shard_size
    expected_count = min(shard_size, sample_count - expected_start)
    if int(meta["sample_start"]) != expected_start or int(meta["sample_count"]) != expected_count:
        raise RuntimeError(f"Shard {shard_index} range metadata is wrong: {meta}")

    keys = set(meta.get("keys", {}).keys())
    missing = REQUIRED_SHARD_KEYS - keys
    if missing:
        raise RuntimeError(f"Shard {shard_index} missing keys: {sorted(missing)}")
    if not keys.intersection(LATENT_SHARD_KEYS):
        raise RuntimeError(f"Shard {shard_index} missing one latent key from {sorted(LATENT_SHARD_KEYS)}")

    with safe_open(str(data_path), framework="pt", device="cpu") as shard:
        shard_keys = set(shard.keys())
        missing_data = REQUIRED_SHARD_KEYS - shard_keys
        if missing_data:
            raise RuntimeError(f"Shard {shard_index} data missing keys: {sorted(missing_data)}")
        if not shard_keys.intersection(LATENT_SHARD_KEYS):
            raise RuntimeError(f"Shard {shard_index} data missing one latent key from {sorted(LATENT_SHARD_KEYS)}")
        sample_ids = shard.get_tensor("sample_ids")
        if int(sample_ids.shape[0]) != expected_count:
            raise RuntimeError(f"Shard {shard_index} sample_ids length mismatch")
        digest = _sha256_int64_tensor(sample_ids)
        if digest != str(meta.get("sample_ids_sha256")):
            raise RuntimeError(f"Shard {shard_index} sample_ids checksum mismatch")
        for key in REQUIRED_SHARD_KEYS | shard_keys.intersection(LATENT_SHARD_KEYS):
            shape = shard.get_slice(key).get_shape()
            if int(shape[0]) != expected_count:
                raise RuntimeError(f"Shard {shard_index} key {key} first dim {shape[0]} != {expected_count}")
    return expected_count


def verify(root: Path, *, max_shards: int | None = None) -> None:
    metadata = _verify_metadata(root)
    _verify_episodes(root, metadata)

    shard_count = int(metadata["shard_count"])
    if max_shards is not None:
        shard_indices: List[int] = list(range(min(shard_count, int(max_shards))))
    else:
        shard_indices = list(range(shard_count))
    covered = 0
    for shard_index in shard_indices:
        covered += _verify_shard(
            root,
            shard_index,
            shard_size=int(metadata["shard_size"]),
            sample_count=int(metadata["sample_count"]),
        )

    suffix = "" if max_shards is None else f" (checked first {len(shard_indices)} shards)"
    print(
        f"OK: {root} format={metadata['format']} version={metadata['version']} "
        f"samples={metadata['sample_count']} shards={metadata['shard_count']}{suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a final EfficientWAM train dataset")
    parser.add_argument("dataset_root", help="Path containing dataset.json and shards/")
    parser.add_argument("--max-shards", type=int, default=None, help="Check only the first N shards")
    args = parser.parse_args()
    verify(Path(args.dataset_root).expanduser(), max_shards=args.max_shards)


if __name__ == "__main__":
    main()
