"""Prepare PCA statistics for Stage 1 distillation."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Dict
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root.resolve()) not in sys.path:
    sys.path.insert(0, str(project_root.resolve()))

from third_party.wan.utils.fm import FlowMatchScheduler
from data.robotwin2.robotwin_agilex_dataset import RoboTwinTaskDataset
from models.wan_model import WanVideoModel
from train.common import build_arg_parser, load_yaml_config, setup_logging


def _build_dataset(cfg: Dict[str, object]) -> RoboTwinTaskDataset:
    dataset_cfg = cfg["dataset"]
    return RoboTwinTaskDataset(
        dataset_dir=dataset_cfg["root"],
        data_mode=dataset_cfg.get("data_mode", "both"),
        task_mode=dataset_cfg.get("task_mode", "multi"),
        task_name=dataset_cfg.get("task_name"),
        num_video_frames=dataset_cfg["num_video_frames"],
        global_downsample_rate=dataset_cfg["global_downsample_rate"],
        video_action_freq_ratio=dataset_cfg["video_action_freq_ratio"],
        video_size=tuple(dataset_cfg["video_size"]),
        action_normalization=dataset_cfg.get("action_normalization"),
    )


def _build_teacher(cfg: Dict[str, object], device: str) -> WanVideoModel:
    teacher_cfg = cfg["teacher"]
    return WanVideoModel.from_pretrained(
        checkpoint_path=teacher_cfg["checkpoint_path"],
        vae_path=teacher_cfg.get("vae_path"),
        config_path=teacher_cfg.get("config_path"),
        device=device,
        precision=teacher_cfg.get("precision", "bfloat16"),
        load_vae=bool(teacher_cfg.get("load_vae", True)),
    )


def _prepare_video_latents(
    batch: Dict[str, torch.Tensor],
    teacher: WanVideoModel,
    scheduler: FlowMatchScheduler,
    seed: int,
    future_video_size: tuple[int, int] | None = None,
) -> Dict[str, torch.Tensor]:
    first_frame = batch["first_frame"]
    video_frames = batch["video_frames"]
    batch_size = video_frames.shape[0]

    first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
    video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
    full_video = torch.cat([first_frame_norm, video_normalized], dim=2)

    with torch.no_grad():
        condition_latent = teacher.encode_video(first_frame_norm.to(teacher.precision))
        if future_video_size is not None:
            full_frames = torch.cat([first_frame.unsqueeze(1), video_frames], dim=1)
            batch_full, time_full, channels, height, width = full_frames.shape
            resized = F.interpolate(
                full_frames.reshape(batch_full * time_full, channels, height, width).float(),
                size=future_video_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).to(dtype=full_frames.dtype)
            future_video = (resized.reshape(batch_full, time_full, channels, *future_video_size) * 2.0 - 1.0)
            future_video = future_video.permute(0, 2, 1, 3, 4)
            clean_future_latent = teacher.encode_video(future_video.to(teacher.precision))[:, :, 1:].contiguous()
            clean_latent = clean_future_latent
        else:
            clean_latent = teacher.encode_video(full_video.to(teacher.precision))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    timestep_id = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), generator=generator)
    t_embed = scheduler.timesteps[timestep_id].to(dtype=teacher.precision, device=teacher.device)
    sigma = scheduler.sigmas[timestep_id].to(dtype=teacher.precision, device=teacher.device).view(batch_size, 1, 1, 1, 1)
    noise = torch.randn(clean_latent.shape, generator=generator, dtype=torch.float32).to(
        device=teacher.device,
        dtype=teacher.precision,
    )
    x_t = clean_latent * (1 - sigma) + noise * sigma
    if future_video_size is None:
        x_t[:, :, 0:1] = condition_latent
    else:
        x_t = {"condition_latent": condition_latent, "future_latent": x_t}
    return {
        "x_t": (
            {key: value.to(dtype=teacher.precision, device=teacher.device) for key, value in x_t.items()}
            if isinstance(x_t, dict)
            else x_t.to(dtype=teacher.precision, device=teacher.device)
        ),
        "clean_latent": clean_latent.to(dtype=teacher.precision, device=teacher.device),
        "condition_latent": condition_latent.to(dtype=teacher.precision, device=teacher.device),
        "t": t_embed.to(dtype=teacher.precision, device=teacher.device),
    }


def _future_video_size(config: Dict[str, object]) -> tuple[int, int] | None:
    dataset_cfg = config.get("dataset", {})
    size = dataset_cfg.get("future_video_size") if isinstance(dataset_cfg, dict) else None
    if size is None:
        return None
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"dataset.future_video_size must be [height, width], got {size!r}")
    return int(size[0]), int(size[1])


def _distributed_info(device: str) -> tuple[int, int, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    resolved_device = device

    if world_size > 1:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            resolved_device = f"cuda:{local_rank}"
            backend = "nccl"
        else:
            backend = "gloo"
        if not dist.is_initialized():
            if backend == "nccl":
                device_id = torch.device(resolved_device)
                try:
                    dist.init_process_group(backend=backend, device_id=device_id)
                except TypeError:
                    dist.init_process_group(backend=backend)
            else:
                dist.init_process_group(backend=backend)
    return rank, world_size, resolved_device


def _distributed_barrier(device: str) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def _rank_item_count(total_items: int, rank: int, world_size: int) -> int:
    if rank >= total_items:
        return 0
    return (total_items - 1 - rank) // world_size + 1


def _rank_token_budget(max_tokens: int, rank: int, world_size: int) -> int:
    if world_size <= 1:
        return max_tokens
    base = max_tokens // world_size
    remainder = max_tokens % world_size
    return base + (1 if rank < remainder else 0)


def _write_progress_file(out_dir: Path, rank: int, completed_items: int) -> None:
    progress_path = out_dir / f".pca_progress_rank{rank:03d}.txt"
    tmp_path = out_dir / f".pca_progress_rank{rank:03d}.tmp"
    tmp_path.write_text(str(int(completed_items)))
    tmp_path.replace(progress_path)


def _read_total_progress(out_dir: Path, world_size: int) -> int:
    total = 0
    for rank in range(world_size):
        progress_path = out_dir / f".pca_progress_rank{rank:03d}.txt"
        if not progress_path.exists():
            continue
        try:
            total += int(progress_path.read_text().strip() or "0")
        except ValueError:
            continue
    return total


def _cleanup_progress_files(out_dir: Path, world_size: int) -> None:
    for rank in range(world_size):
        for suffix in ("txt", "tmp"):
            path = out_dir / f".pca_progress_rank{rank:03d}.{suffix}"
            if path.exists():
                path.unlink()


def _sample_spec_from_index(
    global_sample_index: int,
    dataset: RoboTwinTaskDataset,
    subclips_per_episode: int,
    states_per_subclip: int,
) -> tuple[int, int, int]:
    if dataset.total_episodes == 0:
        raise ValueError("Dataset contains no episodes for PCA preparation")
    episode_slot = global_sample_index // (subclips_per_episode * states_per_subclip)
    within_episode = global_sample_index % (subclips_per_episode * states_per_subclip)
    episode_index = episode_slot % dataset.total_episodes
    subclip_id = within_episode // states_per_subclip
    state_id = within_episode % states_per_subclip
    return episode_index, subclip_id, state_id


def _teacher_feature_dim(teacher: WanVideoModel) -> int:
    dim = getattr(teacher.wan_model, "dim", None)
    if dim is None:
        raise AttributeError("WAN model does not expose hidden dimension via wan_model.dim")
    return int(dim)


def _normalized_tokens(hidden: torch.Tensor) -> torch.Tensor:
    tokens = hidden.detach().reshape(-1, hidden.shape[-1]).float()
    return F.layer_norm(tokens, (tokens.shape[-1],))


def _subsample_tokens(tokens: torch.Tensor, max_tokens: int, seed: int) -> torch.Tensor:
    if max_tokens <= 0 or tokens.shape[0] == 0:
        return tokens[:0]
    if tokens.shape[0] <= max_tokens:
        return tokens
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(tokens.shape[0], generator=generator)[:max_tokens]
    return tokens[indices]


def run_pca_prep(config: Dict[str, object], device: str) -> Path:
    rank, world_size, device = _distributed_info(device)
    dataset = _build_dataset(config)
    teacher = _build_teacher(config, device=device)
    teacher.eval()
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=1000, training=True)

    teacher_layers = sorted(
        set(config["distill"]["hidden_teacher_layers"] + config["distill"]["motion_teacher_layers"])
    )
    max_tokens = int(config["pca_prep"].get("max_tokens_per_layer", 50000))
    per_rank_token_budget = _rank_token_budget(max_tokens, rank, world_size)
    token_counts = {layer: 0 for layer in teacher_layers}
    feature_dim = _teacher_feature_dim(teacher)
    stats_dtype = torch.float64
    layer_stats = {
        layer: {
            "count": torch.zeros(1, device=device, dtype=stats_dtype),
            "sum": torch.zeros(feature_dim, device=device, dtype=stats_dtype),
            "xtx": torch.zeros((feature_dim, feature_dim), device=device, dtype=stats_dtype),
        }
        for layer in teacher_layers
    }

    requested_episodes = int(config["pca_prep"]["episodes"])
    subclips_per_episode = int(config["pca_prep"]["subclips_per_episode"])
    states_per_subclip = int(config["pca_prep"]["states_per_subclip"])
    total_samples = requested_episodes * subclips_per_episode * states_per_subclip
    rank_total_samples = _rank_item_count(total_samples, rank, world_size)
    per_rank_sample_token_budget = max(1, math.ceil(per_rank_token_budget / max(1, rank_total_samples)))
    out_dir = Path(config["artifacts"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_progress_file(out_dir, rank, 0)
    if world_size > 1:
        _distributed_barrier(device)
    progress = tqdm(
        total=total_samples,
        desc="Stage1 PCA prep",
        dynamic_ncols=True,
        disable=rank != 0,
    )
    completed_items = 0
    last_progress_write = time.monotonic()
    last_progress_poll = time.monotonic()

    for global_sample_index in range(rank, total_samples, world_size):
        episode_index, subclip_id, state_id = _sample_spec_from_index(
            global_sample_index,
            dataset,
            subclips_per_episode=subclips_per_episode,
            states_per_subclip=states_per_subclip,
        )
        raw = dataset.build_indexed_sample(
            episode_index,
            subclip_id=subclip_id,
            state_id=state_id,
            subclips_per_episode=subclips_per_episode,
            states_per_subclip=states_per_subclip,
            instruction_idx=state_id,
        )
        batch = {
            "first_frame": raw["first_frame"].unsqueeze(0).to(device),
            "video_frames": raw["video_frames"].unsqueeze(0).to(device),
        }
        prepared = _prepare_video_latents(
            batch,
            teacher,
            scheduler,
            seed=global_sample_index,
            future_video_size=_future_video_size(config),
        )
        with torch.no_grad():
            if isinstance(prepared["x_t"], dict):
                hidden_features = teacher.get_multiscale_layer_features(
                    condition_latent=prepared["x_t"]["condition_latent"],
                    future_latent=prepared["x_t"]["future_latent"],
                    timestep=prepared["t"],
                    text_embeddings=[raw["language_embedding"].to(device)],
                    layer_indices=teacher_layers,
                )
            else:
                hidden_features = teacher.get_layer_features(
                    prepared["x_t"],
                    prepared["t"],
                    [raw["language_embedding"].to(device)],
                    layer_indices=teacher_layers,
                )
        for layer, hidden in zip(teacher_layers, hidden_features[:-1]):
            if token_counts[layer] >= per_rank_token_budget:
                continue
            tokens = _normalized_tokens(hidden)
            tokens = _subsample_tokens(
                tokens,
                max_tokens=per_rank_sample_token_budget,
                seed=(global_sample_index + 1) * 1000 + int(layer),
            )
            remaining = per_rank_token_budget - token_counts[layer]
            tokens = tokens[:remaining]
            if tokens.numel() == 0:
                continue
            tokens = tokens.to(device=device, dtype=stats_dtype)
            layer_stats[layer]["sum"] += tokens.sum(dim=0)
            layer_stats[layer]["xtx"] += tokens.transpose(0, 1) @ tokens
            layer_stats[layer]["count"] += tokens.shape[0]
            token_counts[layer] += tokens.shape[0]
        completed_items += 1
        if world_size == 1:
            progress.update(1)
            progress.set_postfix({
                "episode": episode_index,
                "subclip": subclip_id,
                "state": state_id,
            })
        else:
            now = time.monotonic()
            if now - last_progress_write >= 1.0 or completed_items == rank_total_samples:
                _write_progress_file(out_dir, rank, completed_items)
                last_progress_write = now
            if rank == 0 and now - last_progress_poll >= 1.0:
                progress.n = _read_total_progress(out_dir, world_size)
                progress.refresh()
                last_progress_poll = now

    if world_size > 1:
        _write_progress_file(out_dir, rank, completed_items)
        _distributed_barrier(device)
        if rank == 0:
            progress.n = _read_total_progress(out_dir, world_size)
            progress.refresh()
            _cleanup_progress_files(out_dir, world_size)
        _distributed_barrier(device)
    progress.close()

    if world_size > 1:
        for layer in teacher_layers:
            dist.all_reduce(layer_stats[layer]["count"], op=dist.ReduceOp.SUM)
            dist.all_reduce(layer_stats[layer]["sum"], op=dist.ReduceOp.SUM)
            dist.all_reduce(layer_stats[layer]["xtx"], op=dist.ReduceOp.SUM)

    payload = {"projection_dim": int(config["distill"]["projection_dim"]), "layers": {}}
    output_path = out_dir / "pca_stats.pt"

    if rank == 0:
        projection_dim = int(config["distill"]["projection_dim"])
        for layer in teacher_layers:
            count = int(layer_stats[layer]["count"].item())
            if count <= 0:
                raise RuntimeError(f"No PCA samples collected for teacher layer {layer}")

            sum_vec = layer_stats[layer]["sum"]
            xtx = layer_stats[layer]["xtx"]
            mean = sum_vec / count
            covariance = xtx / count - torch.outer(mean, mean)
            covariance = 0.5 * (covariance + covariance.transpose(0, 1))

            eigvals, eigvecs = torch.linalg.eigh(covariance)
            keep = min(projection_dim, eigvecs.shape[1], count)
            order = torch.argsort(eigvals, descending=True)[:keep]
            components = eigvecs[:, order].contiguous().to(dtype=torch.float32, device="cpu")
            payload["layers"][str(layer)] = {
                "mean": mean.to(dtype=torch.float32, device="cpu"),
                "components": components,
            }

        torch.save(payload, output_path)

    if world_size > 1:
        _distributed_barrier(device)
    return output_path


def main() -> None:
    parser = build_arg_parser("Prepare Stage 1 PCA stats", "configs/robotwin/stage1_pca_prep.yaml")
    args = parser.parse_args()
    setup_logging(args.log_level, rank=int(os.environ.get("RANK", "0")))
    config = load_yaml_config(args.config)
    try:
        output_path = run_pca_prep(config, device=args.device)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Saved PCA stats to {output_path}")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
