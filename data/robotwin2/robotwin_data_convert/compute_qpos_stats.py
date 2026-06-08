#!/usr/bin/env python3
"""Compute per-dimension qpos mean/std stats for converted RoboTwin data."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm


torch.set_num_threads(1)


def _iter_qpos_files(root: Path, splits: list[str]) -> list[Path]:
    files: list[Path] = []
    for split in splits:
        split_root = root / split
        if not split_root.exists():
            continue
        files.extend(path for path in split_root.glob("*/qpos/*.pt") if path.is_file())
    return sorted(files)


def _chunk_list(items: list[Path], chunk_count: int) -> list[list[str]]:
    if not items:
        return []
    chunk_count = max(1, min(chunk_count, len(items)))
    chunk_size = math.ceil(len(items) / chunk_count)
    return [[str(path) for path in items[index : index + chunk_size]] for index in range(0, len(items), chunk_size)]


def _to_2d_float(path: str) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.detach().to(dtype=torch.float64, device="cpu")
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"Expected qpos tensor [T, D], got shape={tuple(tensor.shape)}")
    return tensor


def _stats_worker(paths: list[str]) -> dict[str, object]:
    count = 0
    total = None
    total_sq = None
    qpos_min = None
    qpos_max = None
    used_files = 0
    errors: list[tuple[str, str]] = []

    for path in paths:
        try:
            qpos = _to_2d_float(path)
            if qpos.numel() == 0:
                raise ValueError("Empty qpos tensor")
            rows = int(qpos.shape[0])
            file_sum = qpos.sum(dim=0)
            file_sum_sq = qpos.square().sum(dim=0)
            file_min = qpos.min(dim=0).values
            file_max = qpos.max(dim=0).values
            if total is None:
                total = file_sum
                total_sq = file_sum_sq
                qpos_min = file_min
                qpos_max = file_max
            else:
                if qpos.shape[1] != total.shape[0]:
                    raise ValueError(f"Qpos dim mismatch: expected {total.shape[0]}, got {qpos.shape[1]}")
                total += file_sum
                total_sq += file_sum_sq
                qpos_min = torch.minimum(qpos_min, file_min)
                qpos_max = torch.maximum(qpos_max, file_max)
            count += rows
            used_files += 1
        except Exception as exc:  # noqa: BLE001 - report all bad files in the stats artifact.
            errors.append((path, str(exc)))

    return {
        "count": count,
        "used_files": used_files,
        "sum": total,
        "sum_sq": total_sq,
        "min": qpos_min,
        "max": qpos_max,
        "errors": errors,
    }


def _merge_worker_stats(results: Iterable[dict[str, object]]) -> dict[str, object]:
    count = 0
    used_files = 0
    total = None
    total_sq = None
    qpos_min = None
    qpos_max = None
    errors: list[tuple[str, str]] = []

    for result in results:
        result_count = int(result["count"])
        used_files += int(result["used_files"])
        errors.extend(result["errors"])
        if result_count == 0:
            continue
        result_sum = result["sum"]
        result_sum_sq = result["sum_sq"]
        result_min = result["min"]
        result_max = result["max"]
        if not all(isinstance(value, torch.Tensor) for value in [result_sum, result_sum_sq, result_min, result_max]):
            continue
        if total is None:
            total = result_sum.clone()
            total_sq = result_sum_sq.clone()
            qpos_min = result_min.clone()
            qpos_max = result_max.clone()
        else:
            total += result_sum
            total_sq += result_sum_sq
            qpos_min = torch.minimum(qpos_min, result_min)
            qpos_max = torch.maximum(qpos_max, result_max)
        count += result_count

    if count == 0 or total is None or total_sq is None or qpos_min is None or qpos_max is None:
        raise RuntimeError("No usable qpos files were found")

    mean = total / float(count)
    variance = (total_sq / float(count)) - mean.square()
    std = variance.clamp_min(0.0).sqrt().clamp_min(1e-6)
    return {
        "count": count,
        "used_files": used_files,
        "mean": mean,
        "std": std,
        "min": qpos_min,
        "max": qpos_max,
        "errors": errors,
    }


def compute_qpos_stats_from_files(
    files: list[Path],
    *,
    num_workers: int,
    root: Path | None = None,
    splits: list[str] | None = None,
) -> dict[str, object]:
    if not files:
        raise RuntimeError("No qpos files were provided")

    start = time.time()
    chunks = _chunk_list(files, max(1, num_workers * 4))
    if num_workers <= 1:
        results = [_stats_worker(chunk) for chunk in tqdm(chunks, desc="qpos stats")]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers, maxtasksperchild=32) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_stats_worker, chunks, chunksize=1),
                    total=len(chunks),
                    desc="qpos stats",
                )
            )
    merged = _merge_worker_stats(results)
    elapsed = time.time() - start

    mean = merged["mean"]
    std = merged["std"]
    qpos_min = merged["min"]
    qpos_max = merged["max"]
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    assert isinstance(qpos_min, torch.Tensor)
    assert isinstance(qpos_max, torch.Tensor)

    return {
        "robotwin_qpos": {
            "type": "mean_std",
            "mean": mean.to(dtype=torch.float32).tolist(),
            "std": std.to(dtype=torch.float32).tolist(),
            "min": qpos_min.to(dtype=torch.float32).tolist(),
            "max": qpos_max.to(dtype=torch.float32).tolist(),
            "qpos_dim": int(mean.numel()),
            "file_count": int(merged["used_files"]),
            "total_frames": int(merged["count"]),
            "root": str(root) if root is not None else "",
            "splits": list(splits or []),
            "num_workers": int(max(1, num_workers)),
            "processing_time_seconds": elapsed,
            "error_count": len(merged["errors"]),
            "errors": [{"path": path, "error": error} for path, error in merged["errors"][:100]],
        }
    }


def compute_qpos_stats(root: Path, splits: list[str], num_workers: int) -> dict[str, object]:
    files = _iter_qpos_files(root, splits)
    if not files:
        raise RuntimeError(f"No qpos files found under {root} for splits={splits}")
    return compute_qpos_stats_from_files(files, num_workers=num_workers, root=root, splits=splits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RoboTwin qpos per-dimension mean/std stats")
    parser.add_argument("--root", required=True, help="Converted RoboTwin root containing clean/randomized splits")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--splits", nargs="+", default=["clean", "randomized"], help="Dataset splits to include")
    parser.add_argument("--num-workers", type=int, default=max(1, mp.cpu_count() // 2), help="Worker process count")
    args = parser.parse_args()

    stats = compute_qpos_stats(
        root=Path(args.root).expanduser().resolve(),
        splits=list(args.splits),
        num_workers=max(1, int(args.num_workers)),
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    payload = stats["robotwin_qpos"]
    print(
        f"Saved qpos stats to {output} | files={payload['file_count']} "
        f"frames={payload['total_frames']} dim={payload['qpos_dim']}"
    )


if __name__ == "__main__":
    main()
