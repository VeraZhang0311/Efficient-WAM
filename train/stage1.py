"""Stage 1 EfficientWAM distillation training entry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import gc
import re
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
if str(project_root.resolve()) not in sys.path:
    sys.path.insert(0, str(project_root.resolve()))

from third_party.wan.utils.fm import FlowMatchScheduler
from data.common.dataset_factory import build_training_dataset, is_efficient_wam_train_dataset_config
from data.stage1_dataset import Stage1DistillDataset, stage1_collate_fn
from models.compact_wan import CompactWANConfig, CompactWANModel
from models.stage1_distill_teacher import Stage1DistillTeacherConfig, Stage1DistillTeacherProvider
from models.stage1_distill_heads import DistillHeadConfig, Stage1DistillHeads
from utils.scheduler import create_scheduler

from train.common import (
    add_speed_metrics,
    apply_cli_overrides,
    build_accelerator,
    build_arg_parser,
    LogTracker,
    get_dataloader_config,
    get_effective_global_batch_size,
    get_export_dir,
    get_epoch_progress_metrics,
    get_learning_rate_metrics,
    get_optimizer_config,
    get_per_device_batch_size,
    get_run_dir,
    init_experiment_trackers,
    load_yaml_config,
    logger,
    make_dataset_sampler,
    reduce_metrics_for_log,
    setup_logging,
)


@dataclass
class Stage1System:
    bundle: nn.Module
    student: CompactWANModel
    distill_heads: Stage1DistillHeads | None
    teacher_provider: Stage1DistillTeacherProvider | None
    config: Dict[str, Any]


class Stage1TrainingBundle(nn.Module):
    def __init__(self, student: CompactWANModel, distill_heads: Stage1DistillHeads | None):
        super().__init__()
        self.student = student
        self.distill_heads = distill_heads

    def forward(
        self,
        x_t: torch.Tensor | Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        layer_indices: List[int],
        hidden_anchor_layers: List[int],
        motion_anchor_layers: List[int],
        num_frames: int,
        condition_tokens: int = 0,
    ) -> Dict[str, Any]:
        if isinstance(x_t, dict):
            video_pred, hidden_features = self.student.forward_multiscale_with_features(
                condition_latent=x_t["condition_latent"],
                future_latent=x_t["future_latent"],
                timestep=timestep,
                text_embeddings=text_embeddings,
                layer_indices=layer_indices,
            )
        else:
            video_pred, hidden_features = self.student.forward_with_features(
                x_t,
                timestep,
                text_embeddings,
                layer_indices=layer_indices,
            )
        if self.distill_heads is None:
            return {"video_pred": video_pred}

        hidden_student = {layer: hidden_features[layer] for layer in hidden_anchor_layers}
        motion_student = {layer: hidden_features[layer] for layer in motion_anchor_layers}
        hidden_projected = self.distill_heads.project_hidden(hidden_student)
        motion_projected = self.distill_heads.project_motion(motion_student)
        motion_deltas = self.distill_heads.build_motion_deltas(
            motion_projected,
            num_frames=num_frames,
            condition_tokens=condition_tokens,
        )
        return {
            "video_pred": video_pred,
            "hidden_projected": hidden_projected,
            "motion_deltas": motion_deltas,
        }


class Stage1AccelerateTrainer:
    """Distributed Stage 1 trainer backed by Accelerate/DeepSpeed."""

    def __init__(self, config: Dict[str, Any], accelerator):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self._validate_config()
        self.train_loader = self._build_dataloader()
        self.system = self._build_system_staggered()
        self.optimizer, self.scheduler = self._build_optimizer_and_scheduler()
        self.fm_train_scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)
        self.timestep_sampling_weights = self._build_timestep_sampling_weights()
        self.global_step = 0
        self._condition_latent_reuse_checked = False

        self.system.bundle, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.system.bundle,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        )

    def _unwrapped_bundle(self) -> Stage1TrainingBundle:
        return self.accelerator.unwrap_model(self.system.bundle)

    def _student(self) -> CompactWANModel:
        return self._unwrapped_bundle().student

    def _teacher_mode(self) -> str:
        return str(self.config.get("teacher", {}).get("mode", "distill"))

    @staticmethod
    def _normalize_student_init_mode(value: Any) -> str:
        mode = str(value).lower().replace("-", "_")
        return {
            "slice": "sliced",
            "slicing": "sliced",
            "teacher_slice": "sliced",
            "teacher_sliced": "sliced",
            "from_teacher": "sliced",
            "random_init": "random",
            "from_config": "random",
        }.get(mode, mode)

    def _student_init_mode(self) -> str:
        student_cfg = self.config.get("student", {})
        return self._normalize_student_init_mode(student_cfg.get("init_mode", "sliced"))

    def _is_gt_only_mode(self) -> bool:
        return self._teacher_mode() in {"none", "gt_only"}

    def _validate_config(self) -> None:
        teacher_mode = self._teacher_mode()
        if teacher_mode not in {"distill", "none", "gt_only"}:
            raise ValueError(
                f"Unsupported Stage 1 teacher mode: {teacher_mode}. "
                "Expected distill, none, or gt_only."
            )
        student_init_mode = self._student_init_mode()
        if student_init_mode not in {"sliced", "random"}:
            raise ValueError(
                f"Unsupported Stage 1 student init_mode: {student_init_mode}. "
                "Expected sliced or random."
            )

    def _build_dataloader(self) -> DataLoader:
        dataset_cfg = self.config["dataset"]
        if not is_efficient_wam_train_dataset_config(dataset_cfg):
            raise ValueError(
                "Stage 1 training now requires a built EfficientWAM train dataset. "
                "Set dataset.root to a directory containing dataset.json, or run "
                "scripts/robotwin/build_train_dataset.py first."
            )
        raw_dataset = build_training_dataset(dataset_cfg, stage="stage1")
        sampler = make_dataset_sampler(raw_dataset)
        dataset = Stage1DistillDataset(raw_dataset)
        dl_cfg = get_dataloader_config(self.config)
        return DataLoader(
            dataset,
            batch_size=get_per_device_batch_size(self.config),
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=stage1_collate_fn,
            drop_last=True,
            **dl_cfg,
        )

    def _student_config(self) -> CompactWANConfig:
        student_cfg = self.config["student"]
        future_video_size = student_cfg.get("future_video_size")
        return CompactWANConfig(
            checkpoint_path=student_cfg["checkpoint_path"],
            vae_path=student_cfg["vae_path"],
            config_path=student_cfg.get("config_path"),
            precision=student_cfg.get("precision", "bfloat16"),
            dim=int(student_cfg["dim"]),
            ffn_dim=int(student_cfg["ffn_dim"]),
            num_heads=int(student_cfg["num_heads"]),
            num_layers=int(student_cfg["num_layers"]),
            head_dim=int(student_cfg.get("head_dim", 128)),
            future_video_size=tuple(int(value) for value in future_video_size) if future_video_size else None,
            hidden_anchor_layers=list(student_cfg["hidden_anchor_layers"]),
            motion_anchor_layers=list(student_cfg["motion_anchor_layers"]),
            teacher_layer_mapping=list(student_cfg["teacher_layer_mapping"]),
        )

    def _student_init_checkpoint_path(self) -> Path:
        return get_run_dir(self.config) / "init" / "stage1_init.pt"

    @staticmethod
    def _strip_compact_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if any(key.startswith("student.") for key in state_dict):
            return {
                key[len("student.") :]: value
                for key, value in state_dict.items()
                if key.startswith("student.")
            }
        return state_dict

    @staticmethod
    def _normalize_optional_video_size(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        return tuple(int(item) for item in value)

    def _validate_student_init_checkpoint(
        self,
        checkpoint_path: Path,
        payload: Dict[str, Any],
        compact_cfg: CompactWANConfig,
    ) -> None:
        exported_cfg = payload.get("compact_wan_config")
        if exported_cfg is None:
            raise RuntimeError(f"Student init checkpoint is missing compact_wan_config: {checkpoint_path}")
        expected = {
            "dim": compact_cfg.dim,
            "ffn_dim": compact_cfg.ffn_dim,
            "num_heads": compact_cfg.num_heads,
            "num_layers": compact_cfg.num_layers,
            "head_dim": compact_cfg.head_dim,
            "hidden_anchor_layers": list(compact_cfg.hidden_anchor_layers),
            "motion_anchor_layers": list(compact_cfg.motion_anchor_layers),
            "teacher_layer_mapping": list(compact_cfg.teacher_layer_mapping),
        }
        mismatches = []
        for key in [
            "dim",
            "ffn_dim",
            "num_heads",
            "num_layers",
            "head_dim",
            "hidden_anchor_layers",
            "motion_anchor_layers",
            "teacher_layer_mapping",
        ]:
            if exported_cfg.get(key) != expected[key]:
                mismatches.append(f"{key}: expected {expected[key]!r}, got {exported_cfg.get(key)!r}")
        if mismatches:
            raise RuntimeError(
                f"Student init checkpoint metadata mismatch for {checkpoint_path}: " + "; ".join(mismatches)
            )
        exported_future_size = self._normalize_optional_video_size(exported_cfg.get("future_video_size"))
        expected_future_size = self._normalize_optional_video_size(compact_cfg.future_video_size)
        if exported_future_size != expected_future_size:
            logger.warning(
                "Student init checkpoint future_video_size differs from current Stage 1 config for %s: "
                "expected %r, got %r. Loading is allowed because future_video_size controls runtime "
                "multiscale layout and does not change compact WAN parameter shapes.",
                checkpoint_path,
                expected_future_size,
                exported_future_size,
            )

    def _save_student_init_checkpoint(self, student: CompactWANModel, checkpoint_path: Path) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        cpu_state = {
            key: value.detach().cpu()
            for key, value in student.state_dict().items()
        }
        try:
            torch.save(
                {
                    "model": cpu_state,
                    "compact_wan_config": student.metadata(),
                    "source": "stage1_distributed_student_init",
                },
                tmp_path,
            )
            tmp_path.replace(checkpoint_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        finally:
            del cpu_state
            gc.collect()
        logger.info("Saved distributed Stage 1 student init checkpoint to %s", checkpoint_path)

    def _load_student_init_checkpoint(
        self,
        compact_cfg: CompactWANConfig,
        checkpoint_path: Path,
    ) -> CompactWANModel:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Student init checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict) or "model" not in payload:
            raise TypeError(f"Unsupported student init checkpoint format at {checkpoint_path}")
        self._validate_student_init_checkpoint(checkpoint_path, payload, compact_cfg)
        if not isinstance(payload["model"], dict):
            raise TypeError(f"Student init checkpoint model payload is not a state dict: {checkpoint_path}")
        state_dict = self._strip_compact_prefixes(payload["model"])
        student = CompactWANModel.from_config(compact_cfg, device=str(self.device))
        student.load_state_dict(state_dict, strict=True)
        del payload, state_dict
        gc.collect()
        logger.info("Loaded distributed Stage 1 student init checkpoint from %s", checkpoint_path)
        return student

    def _cleanup_student_init_checkpoint(self, checkpoint_path: Path) -> None:
        try:
            checkpoint_path.unlink(missing_ok=True)
            logger.info("Removed distributed Stage 1 student init checkpoint %s", checkpoint_path)
        except OSError as exc:
            logger.warning("Failed to remove distributed Stage 1 student init checkpoint %s: %s", checkpoint_path, exc)
            return
        try:
            checkpoint_path.parent.rmdir()
        except OSError:
            pass

    def _build_student(
        self,
        init_checkpoint_path: Path | None = None,
        save_init_checkpoint: bool = False,
    ) -> CompactWANModel:
        compact_cfg = self._student_config()
        if init_checkpoint_path is not None and not save_init_checkpoint:
            return self._load_student_init_checkpoint(compact_cfg, init_checkpoint_path)

        init_mode = self._student_init_mode()
        if init_mode == "random":
            logger.info("Initializing Stage 1 compact student from config with random WAN weights")
            student = CompactWANModel.from_config(compact_cfg, device=str(self.device))
        else:
            logger.info("Initializing Stage 1 compact student by structured slicing from teacher checkpoint")
            student = CompactWANModel.from_teacher_checkpoint(compact_cfg, device=str(self.device))
        if init_checkpoint_path is not None and save_init_checkpoint:
            self._save_student_init_checkpoint(student, init_checkpoint_path)
        return student

    def _build_distill_heads(self, compact_cfg: CompactWANConfig) -> Stage1DistillHeads:
        distill_cfg = DistillHeadConfig(
            hidden_dim=compact_cfg.dim,
            projection_dim=int(self.config["distill"]["projection_dim"]),
            hidden_anchor_layers=compact_cfg.hidden_anchor_layers,
            motion_anchor_layers=compact_cfg.motion_anchor_layers,
        )
        return Stage1DistillHeads(distill_cfg)

    def _build_teacher_provider(self) -> Stage1DistillTeacherProvider:
        teacher_cfg = self.config["teacher"]
        distill_teacher_cfg = Stage1DistillTeacherConfig(
            checkpoint_path=teacher_cfg["checkpoint_path"],
            vae_path=teacher_cfg.get("vae_path"),
            config_path=teacher_cfg.get("config_path"),
            precision=teacher_cfg.get("precision", "bfloat16"),
            load_vae=bool(teacher_cfg.get("load_vae", False)),
            hidden_anchor_teacher_layers=list(self.config["distill"]["hidden_teacher_layers"]),
            motion_anchor_teacher_layers=list(self.config["distill"]["motion_teacher_layers"]),
        )
        return Stage1DistillTeacherProvider(
            distill_teacher_cfg,
            student_hidden_layers=self.config["student"]["hidden_anchor_layers"],
            student_motion_layers=self.config["student"]["motion_anchor_layers"],
            pca_stats_path=teacher_cfg["pca_stats_path"],
            device=str(self.device),
        )

    def _build_system(
        self,
        student_init_checkpoint_path: Path | None = None,
        save_student_init_checkpoint: bool = False,
    ) -> Stage1System:
        student = self._build_student(
            init_checkpoint_path=student_init_checkpoint_path,
            save_init_checkpoint=save_student_init_checkpoint,
        )
        if self._is_gt_only_mode():
            heads = None
            teacher_provider = None
        else:
            heads = self._build_distill_heads(student.config)
            teacher_provider = self._build_teacher_provider()
        bundle = Stage1TrainingBundle(student, heads).to(self.device)
        return Stage1System(
            bundle=bundle,
            student=student,
            distill_heads=heads,
            teacher_provider=teacher_provider,
            config=self.config,
        )

    def _build_system_staggered(self) -> Stage1System:
        if self.accelerator.num_processes <= 1:
            return self._build_system()

        student_init_checkpoint_path = self._student_init_checkpoint_path()

        student: CompactWANModel | None = None
        if self.accelerator.is_main_process:
            logger.info("Building Stage 1 compact student on rank 0/%d", self.accelerator.num_processes)
            student = self._build_student(
                init_checkpoint_path=student_init_checkpoint_path,
                save_init_checkpoint=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_main_process:
            logger.info(
                "Loading Stage 1 compact student init checkpoint on rank %d/%d",
                self.accelerator.process_index,
                self.accelerator.num_processes,
            )
            student = self._build_student(
                init_checkpoint_path=student_init_checkpoint_path,
                save_init_checkpoint=False,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.accelerator.wait_for_everyone()
        if student is None:
            raise RuntimeError(f"Rank {self.accelerator.process_index} did not build Stage 1 compact student")

        if self._is_gt_only_mode():
            heads = None
            teacher_provider = None
        else:
            heads = self._build_distill_heads(student.config)
            teacher_provider = None
            for rank in range(self.accelerator.num_processes):
                if self.accelerator.process_index == rank:
                    logger.info(
                        "Building Stage 1 distillation teacher on rank %d/%d",
                        rank,
                        self.accelerator.num_processes,
                    )
                    teacher_provider = self._build_teacher_provider()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self.accelerator.wait_for_everyone()
            if teacher_provider is None:
                raise RuntimeError(f"Rank {self.accelerator.process_index} did not build Stage 1 distillation teacher")

        bundle = Stage1TrainingBundle(student, heads).to(self.device)
        system = Stage1System(
            bundle=bundle,
            student=student,
            distill_heads=heads,
            teacher_provider=teacher_provider,
            config=self.config,
        )

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._cleanup_student_init_checkpoint(student_init_checkpoint_path)
        self.accelerator.wait_for_everyone()
        return system

    def _build_optimizer_and_scheduler(self):
        opt_cfg = get_optimizer_config(self.config)
        params = [p for p in self.system.bundle.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=opt_cfg["learning_rate"], weight_decay=opt_cfg["weight_decay"])

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.training = type(
            "TrainingCfg",
            (),
            {
                "scheduler_type": self.config.get("scheduler_type", "cosine"),
                "max_steps": int(self.config["max_steps"]),
                "lr_schedule_steps": int(self.config.get("lr_schedule_steps", self.config["max_steps"])),
                "lr_cycle_steps": int(
                    self.config.get(
                        "lr_cycle_steps",
                        self.config.get("lr_schedule_steps", self.config["max_steps"]),
                    )
                ),
                "lr_restart_decay": float(self.config.get("lr_restart_decay", 1.0)),
                "warmup_steps": int(self.config.get("warmup_steps", 0)),
                "learning_rate": opt_cfg["learning_rate"],
                "min_lr_ratio": opt_cfg["min_lr_ratio"],
                "min_lr": self.config.get("min_lr"),
            },
        )()
        return optimizer, create_scheduler(optimizer, cfg)

    def _reuse_condition_latent_from_clean(self) -> bool:
        performance_cfg = self.config.get("performance", {})
        return bool(performance_cfg.get("reuse_condition_latent_from_clean", True))

    def _verify_condition_latent_reuse(self) -> bool:
        performance_cfg = self.config.get("performance", {})
        return bool(performance_cfg.get("verify_condition_latent_reuse", False))

    def _build_timestep_sampling_weights(self) -> torch.Tensor:
        distill_cfg = self.config.get("distill", {})
        sampler_cfg = distill_cfg.get("timestep_sampler", {})
        sampler_type = str(sampler_cfg.get("type", "uniform")).lower()
        num_steps = int(self.fm_train_scheduler.num_train_timesteps)
        sigmas = self.fm_train_scheduler.sigmas[:num_steps].float()
        if sampler_type == "uniform":
            weights = torch.ones_like(sigmas)
        elif sampler_type in {"sigma_aware", "sigma_mixture"}:
            width = max(float(sampler_cfg.get("width", 0.22)), 1e-6)
            uniform_weight = float(sampler_cfg.get("uniform_weight", 0.25))
            weights = torch.full_like(sigmas, uniform_weight)
            for name, default_center, default_weight in [
                ("low", 0.25, 1.0),
                ("mid", 0.50, 1.25),
                ("high", 0.75, 0.75),
            ]:
                center = float(sampler_cfg.get(f"{name}_center", default_center))
                weight = float(sampler_cfg.get(f"{name}_weight", default_weight))
                weights = weights + weight * torch.exp(-0.5 * ((sigmas - center) / width).pow(2))
        else:
            raise ValueError(f"Unsupported Stage 1 timestep sampler type: {sampler_type}")
        return weights.clamp_min(1e-8) / weights.sum().clamp_min(1e-8)

    def _sample_timestep_ids(self, batch_size: int) -> torch.Tensor:
        weights = self.timestep_sampling_weights.to(device=self.device)
        return torch.multinomial(weights, batch_size, replacement=True)

    @staticmethod
    def _masked_mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred.float() - target.float()).pow(2).flatten(1).mean(dim=1)

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def _sigma_loss_weights(self, sigma: torch.Tensor, name: str) -> torch.Tensor:
        cfg = self.config.get("distill", {}).get("sigma_loss_weights", {}).get(name, {})
        mode = str(cfg.get("mode", "uniform")).lower()
        sigma = sigma.float().view(-1)
        floor = float(cfg.get("floor", 0.0))
        if mode == "uniform":
            weights = torch.ones_like(sigma)
        elif mode in {"low", "low_mid"}:
            max_sigma = float(cfg.get("max_sigma", 0.75))
            softness = max(float(cfg.get("softness", 0.12)), 1e-6)
            gate = torch.sigmoid((max_sigma - sigma) / softness)
            weights = floor + (1.0 - floor) * gate
        elif mode in {"high", "mid_high"}:
            min_sigma = float(cfg.get("min_sigma", 0.25))
            softness = max(float(cfg.get("softness", 0.12)), 1e-6)
            gate = torch.sigmoid((sigma - min_sigma) / softness)
            weights = floor + (1.0 - floor) * gate
        elif mode == "mid":
            center = float(cfg.get("center", 0.50))
            width = max(float(cfg.get("width", 0.25)), 1e-6)
            gate = torch.exp(-0.5 * ((sigma - center) / width).pow(2))
            weights = floor + (1.0 - floor) * gate
        else:
            raise ValueError(f"Unsupported sigma loss weight mode for {name}: {mode}")
        return weights.clamp_min(1e-8)

    def _to_text_embedding_list(self, batch: Dict[str, Any]) -> List[torch.Tensor]:
        values = batch.get("text_embeddings", batch.get("language_embedding"))
        if values is None:
            raise KeyError("Stage 1 batch missing text embeddings")
        dtype = self._student().video_model.precision
        return [value.to(self.device, dtype=dtype, non_blocking=True) for value in values]

    def _prepare_distill_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        student = self._student()
        dtype = student.video_model.precision
        if "future_latent" in batch:
            condition_latent = batch["condition_latent"].to(self.device, dtype=dtype, non_blocking=True)
            clean_future_latent = batch["future_latent"].to(self.device, dtype=dtype, non_blocking=True)
            batch_size = clean_future_latent.shape[0]
            timestep_id = self._sample_timestep_ids(batch_size)
            timesteps = self.fm_train_scheduler.timesteps.to(dtype=dtype, device=self.device)
            sigmas = self.fm_train_scheduler.sigmas.to(dtype=dtype, device=self.device)
            t = timesteps[timestep_id]
            sigma = sigmas[timestep_id].view(batch_size, 1, 1, 1, 1)
            noise = torch.randn_like(clean_future_latent, dtype=dtype)
            future_latent = clean_future_latent * (1 - sigma) + noise * sigma
            future_target = noise - clean_future_latent
            condition_tokens = int(
                condition_latent.shape[2]
                * (condition_latent.shape[3] // 2)
                * (condition_latent.shape[4] // 2)
            )
            return {
                **batch,
                "x_t": {
                    "condition_latent": condition_latent,
                    "future_latent": future_latent,
                },
                "condition_latent": condition_latent,
                "clean_future_latent": clean_future_latent,
                "video_target": future_target,
                "t": t,
                "sigma": sigma,
                "timestep_id": timestep_id,
                "noise_seed": timestep_id.detach().clone(),
                "condition_tokens": condition_tokens,
                "num_motion_frames": int(clean_future_latent.shape[2]),
                "text_embeddings": self._to_text_embedding_list(batch),
            }

        if "clean_latent" in batch and "condition_latent" in batch:
            clean_latent = batch["clean_latent"].to(self.device, dtype=dtype, non_blocking=True)
            condition_latent = batch["condition_latent"].to(self.device, dtype=dtype, non_blocking=True)
            batch_size = clean_latent.shape[0]
        else:
            first_frame = batch["first_frame"].to(self.device, dtype=dtype, non_blocking=True)
            video_frames = batch["video_frames"].to(self.device, dtype=dtype, non_blocking=True)
            batch_size = video_frames.shape[0]

            first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
            video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
            full_video = torch.cat([first_frame_norm, video_normalized], dim=2)

            with torch.no_grad():
                clean_latent = student.encode_video(full_video)
                if self._reuse_condition_latent_from_clean():
                    condition_latent = clean_latent[:, :, 0:1]
                    if self._verify_condition_latent_reuse() and not self._condition_latent_reuse_checked:
                        encoded_condition_latent = student.encode_video(first_frame_norm)
                        max_abs_diff = (condition_latent.float() - encoded_condition_latent.float()).abs().max()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "Stage 1 condition latent reuse check max_abs_diff=%.6g",
                                float(max_abs_diff.item()),
                            )
                        self._condition_latent_reuse_checked = True
                else:
                    condition_latent = student.encode_video(first_frame_norm)

        timestep_id = self._sample_timestep_ids(batch_size)
        timesteps = self.fm_train_scheduler.timesteps.to(dtype=dtype, device=self.device)
        sigmas = self.fm_train_scheduler.sigmas.to(dtype=dtype, device=self.device)
        t = timesteps[timestep_id]
        sigma = sigmas[timestep_id].view(batch_size, 1, 1, 1, 1)
        noise = torch.randn_like(clean_latent, dtype=dtype)
        x_t = clean_latent * (1 - sigma) + noise * sigma
        x_t[:, :, 0:1] = condition_latent
        video_target = noise - clean_latent
        video_target[:, :, 0:1] = 0

        return {
            **batch,
            "x_t": x_t,
            "clean_latent": clean_latent,
            "condition_latent": condition_latent,
            "video_target": video_target,
            "t": t,
            "sigma": sigma,
            "timestep_id": timestep_id,
            "noise_seed": timestep_id.detach().clone(),
            "text_embeddings": self._to_text_embedding_list(batch),
        }

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self._prepare_distill_batch(batch)

    @staticmethod
    def _scheduled_weight(
        progress: float,
        config: Dict[str, Any],
        schedule_key: str,
        default: float,
    ) -> float:
        distill_cfg = config.get("distill", {})
        schedule = distill_cfg.get(schedule_key)
        if schedule is None:
            return float(default)
        boundaries = distill_cfg.get("schedule_boundaries", [1.0] * len(schedule))
        for boundary, weight in zip(boundaries, schedule):
            if progress <= float(boundary):
                return float(weight)
        return float(schedule[-1])

    @classmethod
    def distill_weights(cls, progress: float, config: Dict[str, Any]) -> Dict[str, float]:
        distill_cfg = config.get("distill", {})
        return {
            "gt": cls._scheduled_weight(
                progress,
                config,
                "lambda_gt_schedule",
                float(distill_cfg.get("lambda_gt", 1.0)),
            ),
            "hidden": cls._scheduled_weight(progress, config, "lambda_hidden_schedule", 0.0),
            "motion": cls._scheduled_weight(progress, config, "lambda_motion_schedule", 0.0),
            "velocity": cls._scheduled_weight(progress, config, "lambda_velocity_schedule", 0.0),
        }

    def _forward_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prepared = self._prepare_batch(batch)
        compact_cfg = self._student().config
        progress = min(1.0, float(self.global_step + 1) / float(self.config["max_steps"]))
        weights = self.distill_weights(progress, self.config)
        enable_hidden_kd = (not self._is_gt_only_mode()) and weights["hidden"] > 0.0
        compute_motion_kd = (not self._is_gt_only_mode()) and weights["motion"] > 0.0
        enable_velocity_kd = (not self._is_gt_only_mode()) and weights["velocity"] > 0.0
        teacher_active = enable_hidden_kd or compute_motion_kd or enable_velocity_kd
        if self._is_gt_only_mode():
            layer_indices = []
            hidden_anchor_layers = []
            motion_anchor_layers = []
        else:
            hidden_anchor_layers = compact_cfg.hidden_anchor_layers if enable_hidden_kd else []
            motion_anchor_layers = compact_cfg.motion_anchor_layers if compute_motion_kd else []
            layer_indices = sorted(set(hidden_anchor_layers + motion_anchor_layers))
        num_motion_frames = (
            int(prepared["num_motion_frames"])
            if "num_motion_frames" in prepared
            else int(prepared["clean_latent"].shape[2])
        )
        outputs = self.system.bundle(
            prepared["x_t"],
            prepared["t"],
            prepared["text_embeddings"],
            layer_indices,
            hidden_anchor_layers,
            motion_anchor_layers,
            num_motion_frames,
            int(prepared.get("condition_tokens", 0)),
        )
        video_pred_masked = outputs["video_pred"].clone()
        if "clean_future_latent" not in prepared:
            video_pred_masked[:, :, 0:1] = 0
        future_loss_scale = 1.0
        if "clean_future_latent" in prepared:
            condition_frames = int(prepared["condition_latent"].shape[2])
            future_frames = int(prepared["clean_future_latent"].shape[2])
            future_loss_scale = future_frames / max(1, condition_frames + future_frames)
        batch_size = int(video_pred_masked.shape[0])
        sigma_values = prepared["sigma"].float().view(batch_size, -1)[:, 0]
        sigma_weight_gt = self._sigma_loss_weights(sigma_values, "gt")
        sigma_weight_hidden = self._sigma_loss_weights(sigma_values, "hidden")
        sigma_weight_motion = self._sigma_loss_weights(sigma_values, "motion")
        sigma_weight_velocity = self._sigma_loss_weights(sigma_values, "velocity")
        gt_loss_vec = self._masked_mse_per_sample(video_pred_masked, prepared["video_target"]) * future_loss_scale
        gt_loss = self._weighted_mean(gt_loss_vec, sigma_weight_gt)
        zero_loss = gt_loss.detach().new_zeros(())

        if self._is_gt_only_mode():
            return {
                "total_loss": float(weights["gt"]) * gt_loss.float(),
                "gt_loss": gt_loss.detach(),
                "hidden_loss": zero_loss,
                "motion_loss": zero_loss,
                "velocity_loss": zero_loss,
                "lambda_gt": gt_loss.detach().new_tensor(float(weights["gt"])),
                "lambda_hidden": zero_loss,
                "lambda_motion": zero_loss,
                "lambda_velocity": zero_loss,
                "sigma_mean": sigma_values.detach().mean(),
                "sigma_low_ratio": (sigma_values < 0.33).float().mean().detach(),
                "sigma_mid_ratio": ((sigma_values >= 0.33) & (sigma_values < 0.66)).float().mean().detach(),
                "sigma_high_ratio": (sigma_values >= 0.66).float().mean().detach(),
                "sigma_weight_gt_mean": sigma_weight_gt.detach().mean(),
                "sigma_weight_hidden_mean": sigma_weight_hidden.detach().mean(),
                "sigma_weight_motion_mean": sigma_weight_motion.detach().mean(),
                "sigma_weight_velocity_mean": sigma_weight_velocity.detach().mean(),
            }

        if teacher_active and self.system.teacher_provider is None:
            raise RuntimeError("Stage 1 distillation mode requires a teacher provider")
        teacher_targets = (
            self.system.teacher_provider.get_teacher_targets(
                prepared,
                include_hidden=enable_hidden_kd,
                include_motion=compute_motion_kd,
                include_video_pred=enable_velocity_kd,
            )
            if teacher_active
            else None
        )

        if enable_hidden_kd:
            if teacher_targets is None:
                raise RuntimeError("Hidden KD requested but teacher targets were not computed")
            hidden_loss_vec = Stage1DistillHeads.hidden_cosine_loss_per_sample(
                outputs["hidden_projected"],
                teacher_targets.hidden_targets,
            )
            hidden_loss = self._weighted_mean(hidden_loss_vec, sigma_weight_hidden)
        else:
            hidden_loss = zero_loss

        if compute_motion_kd:
            if teacher_targets is None:
                raise RuntimeError("Motion KD requested but teacher targets were not computed")
            motion_loss_vec = Stage1DistillHeads.motion_cosine_loss_per_sample(
                outputs["motion_deltas"],
                teacher_targets.motion_targets,
            )
            motion_loss = self._weighted_mean(motion_loss_vec, sigma_weight_motion)
        else:
            motion_loss = zero_loss

        if enable_velocity_kd:
            if teacher_targets is None:
                raise RuntimeError("Velocity KD requested but teacher targets were not computed")
            if teacher_targets.video_pred is None:
                raise RuntimeError("Teacher velocity KD requested but teacher returned no video_pred")
            teacher_video_pred_masked = teacher_targets.video_pred.detach().clone()
            if "clean_future_latent" not in prepared:
                teacher_video_pred_masked[:, :, 0:1] = 0
            velocity_loss_vec = (
                self._masked_mse_per_sample(video_pred_masked, teacher_video_pred_masked)
                * future_loss_scale
            )
            velocity_loss = self._weighted_mean(
                velocity_loss_vec,
                sigma_weight_velocity,
            )
        else:
            velocity_loss = zero_loss

        total_loss = (
            float(weights["gt"]) * gt_loss.float()
            + float(weights["hidden"]) * hidden_loss.float()
            + float(weights["motion"]) * motion_loss.float()
            + float(weights["velocity"]) * velocity_loss.float()
        )
        return {
            "total_loss": total_loss,
            "gt_loss": gt_loss.detach(),
            "hidden_loss": hidden_loss.detach(),
            "motion_loss": motion_loss.detach(),
            "velocity_loss": velocity_loss.detach(),
            "lambda_gt": gt_loss.detach().new_tensor(float(weights["gt"])),
            "lambda_hidden": gt_loss.detach().new_tensor(float(weights["hidden"])),
            "lambda_motion": gt_loss.detach().new_tensor(float(weights["motion"])),
            "lambda_velocity": gt_loss.detach().new_tensor(float(weights["velocity"])),
            "sigma_mean": sigma_values.detach().mean(),
            "sigma_low_ratio": (sigma_values < 0.33).float().mean().detach(),
            "sigma_mid_ratio": ((sigma_values >= 0.33) & (sigma_values < 0.66)).float().mean().detach(),
            "sigma_high_ratio": (sigma_values >= 0.66).float().mean().detach(),
            "sigma_weight_gt_mean": sigma_weight_gt.detach().mean(),
            "sigma_weight_hidden_mean": sigma_weight_hidden.detach().mean(),
            "sigma_weight_motion_mean": sigma_weight_motion.detach().mean(),
            "sigma_weight_velocity_mean": sigma_weight_velocity.detach().mean(),
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float] | None:
        self.system.bundle.train()
        grad_norm = None
        with self.accelerator.accumulate(self.system.bundle):
            losses = self._forward_losses(batch)
            self.accelerator.backward(losses["total_loss"])
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.system.bundle.parameters(),
                    max_norm=float(self.config.get("grad_clip_norm", 1.0)),
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        if not self.accelerator.sync_gradients:
            return None
        metrics = {
            "loss/total": float(losses["total_loss"].item()),
            "loss/gt": float(losses["gt_loss"].item()),
            "loss/hidden": float(losses["hidden_loss"].item()),
            "loss/motion": float(losses["motion_loss"].item()),
            "loss/velocity": float(losses["velocity_loss"].item()),
            "loss_weight/gt": float(losses["lambda_gt"].item()),
            "loss_weight/hidden": float(losses["lambda_hidden"].item()),
            "loss_weight/motion": float(losses["lambda_motion"].item()),
            "loss_weight/velocity": float(losses["lambda_velocity"].item()),
            "sigma/mean": float(losses["sigma_mean"].item()),
            "sigma/low_ratio": float(losses["sigma_low_ratio"].item()),
            "sigma/mid_ratio": float(losses["sigma_mid_ratio"].item()),
            "sigma/high_ratio": float(losses["sigma_high_ratio"].item()),
            "sigma_loss_weight/gt_mean": float(losses["sigma_weight_gt_mean"].item()),
            "sigma_loss_weight/hidden_mean": float(losses["sigma_weight_hidden_mean"].item()),
            "sigma_loss_weight/motion_mean": float(losses["sigma_weight_motion_mean"].item()),
            "sigma_loss_weight/velocity_mean": float(losses["sigma_weight_velocity_mean"].item()),
        }
        if grad_norm is not None:
            metrics["grad/norm"] = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
        metrics.update(get_learning_rate_metrics(self.optimizer))
        return metrics

    def _resume_if_needed(self) -> None:
        resume_from = self.config.get("resume_from")
        if not resume_from:
            return
        self.accelerator.load_state(str(resume_from))
        match = re.search(r"step_(\d+)", str(resume_from))
        if match:
            self.global_step = int(match.group(1))
        logger.info("Resumed Stage 1 training from %s at step %d", resume_from, self.global_step)

    def save_checkpoint(self) -> None:
        ckpt_dir = get_run_dir(self.config) / f"step_{self.global_step}"
        self.accelerator.save_state(str(ckpt_dir))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            logger.info("Saved Stage 1 resume checkpoint to %s", ckpt_dir)
            self.export_compact_wan()

    def export_compact_wan(self) -> None:
        export_dir = get_export_dir(self.config)
        export_dir.mkdir(parents=True, exist_ok=True)
        bundle_state = self.accelerator.get_state_dict(self.system.bundle)
        compact_state = {
            key[len("student.") :]: value.cpu()
            for key, value in bundle_state.items()
            if key.startswith("student.")
        }
        output_path = export_dir / f"stage1_step_{self.global_step}.pt"
        torch.save(
            {
                "model": compact_state,
                "global_step": self.global_step,
                "config": self.config,
                "compact_wan_config": self._student().metadata(),
            },
            output_path,
        )
        logger.info("Exported Stage 1 compact WAN checkpoint to %s", output_path)

    def run(self) -> None:
        if self.accelerator.is_main_process:
            logger.info("Built Stage 1 EfficientWAM system")
            logger.info("Teacher mode: %s", self._teacher_mode())
            logger.info("Student init mode: %s", self._student_init_mode())
            logger.info("GT-only baseline: %s", self._is_gt_only_mode())
            logger.info("Student WAN layers: %d", self._student().config.num_layers)
            logger.info("Hidden anchors: %s", self._student().config.hidden_anchor_layers)
            logger.info("Motion anchors: %s", self._student().config.motion_anchor_layers)
            logger.info("Run directory: %s", get_run_dir(self.config))
            
            weights = self.distill_weights(0.0, self.config)
            weight_str = " ".join([f"{k}={v}" for k, v in weights.items()])
            logger.info("")
            logger.info("Stage1 training config")
            logger.info("  max_steps      : %d", int(self.config["max_steps"]))
            logger.info("  log_interval   : %d", int(self.config.get("log_interval", 10)))
            logger.info("  loss_weights   : %s", weight_str)

        self.log_tracker = LogTracker("stage1", int(self.config["max_steps"]))
        self._resume_if_needed()
        self._last_log_step = self.global_step
        self._last_log_time = time.monotonic()
        samples_per_step = get_effective_global_batch_size(self.config, self.accelerator.num_processes)
        log_interval = int(self.config.get("log_interval", 10))
        checkpoint_interval = int(self.config.get("checkpoint_interval", 1000))
        data_iter = iter(self.train_loader)

        while self.global_step < int(self.config["max_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            metrics = self.train_step(batch)
            if metrics is None:
                continue
            self.global_step += 1
            metrics.update(
                get_epoch_progress_metrics(
                    self.global_step,
                    self.train_loader,
                    gradient_accumulation_steps=int(self.config.get("gradient_accumulation_steps", 1)),
                    world_size=self.accelerator.num_processes,
                )
            )

            if self.global_step % log_interval == 0:
                self._last_log_step, self._last_log_time = add_speed_metrics(
                    metrics,
                    self.global_step,
                    self._last_log_step,
                    self._last_log_time,
                    samples_per_step=samples_per_step,
                )
                log_metrics = reduce_metrics_for_log(self.accelerator, metrics)
                self.accelerator.log(log_metrics, step=self.global_step)
                if self.accelerator.is_main_process:
                    lines_to_log = self.log_tracker.log_step(
                        self.global_step,
                        log_metrics,
                        loss_keys=[
                            ("total", "loss/total"),
                            ("gt", "loss/gt"),
                        ],
                        lr_keys=[("model", "lr/model")],
                        extra_keys=[
                            ("mean", "sigma/mean"),
                            ("low", "sigma/low_ratio"),
                            ("mid", "sigma/mid_ratio"),
                            ("high", "sigma/high_ratio"),
                        ],
                    )
                    for line in lines_to_log:
                        logger.info("%s", line)

            if self.global_step % checkpoint_interval == 0:
                self.save_checkpoint()

        if self.global_step == 0 or self.global_step % checkpoint_interval != 0:
            self.save_checkpoint()


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    parser = build_arg_parser(
        description="Stage 1 EfficientWAM distillation training",
        default_config="configs/robotwin/stage1_video_distill.yaml",
    )
    args = parser.parse_args()
    config = apply_cli_overrides(load_yaml_config(args.config), args)
    accelerator = build_accelerator(args, config)
    setup_logging(args.log_level, rank=accelerator.process_index)
    init_experiment_trackers(accelerator, config, "efficient_wam_stage1", default_project="stage1")
    trainer = Stage1AccelerateTrainer(config=config, accelerator=accelerator)
    
    # Optional: compile model for further acceleration (PyTorch 2.0+)
    # trainer.system.bundle = torch.compile(trainer.system.bundle)
    
    trainer.run()
    accelerator.end_training()


if __name__ == "__main__":
    main()
