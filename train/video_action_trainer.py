"""Shared EfficientWAM video-action training utilities for Stage 2 and Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import json
from pathlib import Path
import re
import sys
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
if str(project_root.resolve()) not in sys.path:
    sys.path.insert(0, str(project_root.resolve()))

from third_party.wan.utils.fm import FlowMatchScheduler
from data.common.dataset_factory import build_training_dataset, is_efficient_wam_train_dataset_config
from data.video_action_dataset import VideoActionDataset, video_action_collate_fn
from data.efficient_wam_train_dataset import TRAIN_DATASET_ACTION_STATS, TRAIN_DATASET_METADATA
from models.compact_wan import CompactWANConfig, CompactWANModel
from models.small_wam import SmallWAMActionConfig, SmallWAMActionModel
from utils.scheduler import create_scheduler

from train.common import (
    add_speed_metrics,
    apply_cli_overrides,
    build_accelerator,
    build_arg_parser,
    LogTracker,
    get_dataloader_config,
    get_effective_global_batch_size,
    get_epoch_progress_metrics,
    get_export_dir,
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
class VideoActionSystem:
    model: SmallWAMActionModel
    config: Dict[str, Any]


class VideoActionTrainer:
    """Distributed video-action trainer backed by Accelerate/DeepSpeed."""

    def __init__(self, config: Dict[str, Any], accelerator):
        self.config = config
        self.stage_name = str(config.get("training_stage", "stage2")).lower()
        if self.stage_name not in {"stage2", "stage3"}:
            raise ValueError(f"training_stage must be stage2 or stage3, got {self.stage_name!r}")
        self.stage_title = "Stage 2" if self.stage_name == "stage2" else "Stage 3"
        self.accelerator = accelerator
        self.device = accelerator.device
        self.train_loader = self._build_dataloader()
        self.system = self._build_system()
        self.optimizer, self.scheduler = self._build_optimizer_and_scheduler()
        self._verify_trainable_setup_before_prepare()
        self.fm_train_scheduler_action = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.fm_train_scheduler_action.set_timesteps(num_inference_steps=1000, training=True)
        self.fm_train_scheduler_video = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.fm_train_scheduler_video.set_timesteps(num_inference_steps=1000, training=True)
        self._cache_scheduler_tensors()
        self.global_step = 0
        self._condition_latent_reuse_checked = False

        self.system.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.system.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        )

    def _model(self) -> SmallWAMActionModel:
        return self.accelerator.unwrap_model(self.system.model)

    def _cache_scheduler_tensors(self) -> None:
        dtype = self.system.model.compact_wan.video_model.precision
        self._action_timesteps = self.fm_train_scheduler_action.timesteps.to(device=self.device, dtype=dtype)
        self._action_sigmas = self.fm_train_scheduler_action.sigmas.to(device=self.device, dtype=dtype)
        self._video_timesteps = self.fm_train_scheduler_video.timesteps.to(device=self.device, dtype=dtype)
        self._video_sigmas = self.fm_train_scheduler_video.sigmas.to(device=self.device, dtype=dtype)

    def _reuse_condition_latent_from_clean(self) -> bool:
        performance_cfg = self.config.get("performance", {})
        return bool(performance_cfg.get("reuse_condition_latent_from_clean", True))

    def _verify_condition_latent_reuse(self) -> bool:
        performance_cfg = self.config.get("performance", {})
        return bool(performance_cfg.get("verify_condition_latent_reuse", False))

    def _build_dataloader(self) -> DataLoader:
        dataset_cfg = self.config["dataset"]
        if not is_efficient_wam_train_dataset_config(dataset_cfg):
            raise ValueError(
                "Video-action training requires a built EfficientWAM train dataset. "
                "Set dataset.root to a directory containing dataset.json, or run "
                "scripts/robotwin/build_train_dataset.py first."
            )
        raw_dataset = build_training_dataset(dataset_cfg, stage="video_action")
        sampler = make_dataset_sampler(raw_dataset)
        dataset = VideoActionDataset(raw_dataset)
        dl_cfg = get_dataloader_config(self.config)
        return DataLoader(
            dataset,
            batch_size=get_per_device_batch_size(self.config),
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=video_action_collate_fn,
            drop_last=True,
            **dl_cfg,
        )

    @staticmethod
    def _extract_compact_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if any(key.startswith("student.") for key in state_dict):
            return {
                key[len("student.") :]: value
                for key, value in state_dict.items()
                if key.startswith("student.")
            }
        if any(key.startswith("compact_wan.") for key in state_dict):
            return {
                key[len("compact_wan.") :]: value
                for key, value in state_dict.items()
                if key.startswith("compact_wan.")
            }
        return state_dict

    @staticmethod
    def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if any(key.startswith("module.") for key in state_dict):
            return {
                key[len("module.") :] if key.startswith("module.") else key: value
                for key, value in state_dict.items()
            }
        return state_dict

    @staticmethod
    def _normalize_optional_video_size(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        return tuple(int(item) for item in value)

    @classmethod
    def _validate_compact_checkpoint_metadata(
        cls,
        checkpoint_path: str,
        payload: Dict[str, Any],
        compact_cfg: CompactWANConfig,
    ) -> None:
        exported_cfg = payload.get("compact_wan_config")
        if exported_cfg is None:
            raise RuntimeError(
                "Stage 1 compact checkpoint is missing compact_wan_config metadata: "
                f"{checkpoint_path}"
            )

        shape_expected = {
            "dim": compact_cfg.dim,
            "ffn_dim": compact_cfg.ffn_dim,
            "num_heads": compact_cfg.num_heads,
            "num_layers": compact_cfg.num_layers,
            "head_dim": compact_cfg.head_dim,
        }
        provenance_expected = {
            "hidden_anchor_layers": list(compact_cfg.hidden_anchor_layers),
            "motion_anchor_layers": list(compact_cfg.motion_anchor_layers),
            "teacher_layer_mapping": list(compact_cfg.teacher_layer_mapping),
        }
        mismatches = []
        for key, expected_value in shape_expected.items():
            actual_value = exported_cfg.get(key)
            if actual_value != expected_value:
                mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
        if mismatches:
            raise RuntimeError(
                "Stage 1 compact checkpoint architecture mismatch for "
                f"{checkpoint_path}: " + "; ".join(mismatches)
            )
        provenance_mismatches = []
        for key, expected_value in provenance_expected.items():
            actual_value = exported_cfg.get(key)
            if actual_value != expected_value:
                provenance_mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
        if provenance_mismatches:
            logger.warning(
                "Stage 1 compact checkpoint training metadata differs from current config for %s: %s",
                checkpoint_path,
                "; ".join(provenance_mismatches),
            )
        exported_future_size = cls._normalize_optional_video_size(exported_cfg.get("future_video_size"))
        expected_future_size = cls._normalize_optional_video_size(compact_cfg.future_video_size)
        if exported_future_size != expected_future_size:
            logger.warning(
                "Stage 1 compact checkpoint future_video_size differs from current config for %s: "
                "expected %r, got %r. Loading is allowed because future_video_size controls the "
                "runtime multiscale token layout and does not change compact WAN parameter shapes.",
                checkpoint_path,
                expected_future_size,
                exported_future_size,
            )

    def _build_compact_wan(self) -> CompactWANModel:
        wan_cfg = self.config["model"]["compact_wan"]
        future_video_size = wan_cfg.get("future_video_size")
        compact_cfg = CompactWANConfig(
            checkpoint_path=wan_cfg["checkpoint_path"],
            vae_path=wan_cfg["vae_path"],
            config_path=wan_cfg.get("config_path"),
            precision=wan_cfg.get("precision", "bfloat16"),
            dim=int(wan_cfg["dim"]),
            ffn_dim=int(wan_cfg["ffn_dim"]),
            num_heads=int(wan_cfg["num_heads"]),
            num_layers=int(wan_cfg["num_layers"]),
            head_dim=int(wan_cfg.get("head_dim", 128)),
            future_video_size=tuple(int(value) for value in future_video_size) if future_video_size else None,
        )
        checkpoint_path = self.config["model"].get("compact_wan_checkpoint")
        if not checkpoint_path:
            raise ValueError(
                "Video-action training requires model.compact_wan_checkpoint to point to an exported Stage 1 compact WAN checkpoint"
            )

        compact_wan = CompactWANModel.from_config(compact_cfg, device=str(self.device))
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported Stage 1 compact checkpoint format at {checkpoint_path}")
        self._validate_compact_checkpoint_metadata(checkpoint_path, payload if isinstance(payload, dict) else {}, compact_cfg)
        compact_state = self._extract_compact_state(state_dict)
        compact_wan.load_state_dict(compact_state, strict=True)
        logger.info("Loaded Stage 1 compact WAN checkpoint %s", checkpoint_path)
        return compact_wan

    def _load_action_expert_init(self, model: SmallWAMActionModel) -> None:
        model_cfg = self.config["model"]
        checkpoint_path = model_cfg.get("action_init_checkpoint")
        if not checkpoint_path:
            logger.info("Action Expert init checkpoint disabled; using random initialization")
            return

        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported Action Expert init checkpoint format at {checkpoint_path}")

        state_dict = self._strip_module_prefix(state_dict)
        action_state = {
            key[len("action_expert.") :]: value
            for key, value in state_dict.items()
            if key.startswith("action_expert.")
        }
        if not action_state:
            raise RuntimeError(
                "Action Expert init checkpoint did not contain any action_expert.* parameters: "
                f"{checkpoint_path}"
            )

        missing, unexpected = model.action_expert.load_state_dict(action_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Action Expert init checkpoint does not match current Action Expert: "
                f"missing={missing}, unexpected={unexpected}, checkpoint={checkpoint_path}"
            )
        logger.info(
            "Loaded Action Expert init from %s (%d tensors)",
            checkpoint_path,
            len(action_state),
        )

    @classmethod
    def _validate_efficient_wam_checkpoint_metadata(
        cls,
        checkpoint_path: str,
        payload: Dict[str, Any],
        model_cfg: Dict[str, Any],
    ) -> None:
        exported_model_cfg = payload.get("config", {}).get("model") if isinstance(payload, dict) else None
        if not exported_model_cfg:
            logger.warning(
                "EfficientWAM checkpoint %s has no config metadata; architecture validation skipped",
                checkpoint_path,
            )
            return

        checks = [
            ("compact_wan", "dim"),
            ("compact_wan", "ffn_dim"),
            ("compact_wan", "num_heads"),
            ("compact_wan", "num_layers"),
            ("compact_wan", "head_dim"),
            ("action_expert", "dim"),
            ("action_expert", "ffn_dim"),
            ("action_expert", "num_layers"),
            ("action_expert", "chunk_size"),
            ("action_expert", "state_dim"),
            ("action_expert", "action_dim"),
        ]
        mismatches = []
        for section, key in checks:
            expected = model_cfg.get(section, {}).get(key)
            actual = exported_model_cfg.get(section, {}).get(key)
            if expected is not None and actual is not None and expected != actual:
                mismatches.append(f"{section}.{key}: expected {expected!r}, got {actual!r}")
        if mismatches:
            raise RuntimeError(
                "EfficientWAM init checkpoint architecture mismatch for "
                f"{checkpoint_path}: " + "; ".join(mismatches)
            )

        expected_future_size = cls._normalize_optional_video_size(
            model_cfg.get("compact_wan", {}).get("future_video_size")
        )
        exported_future_size = cls._normalize_optional_video_size(
            exported_model_cfg.get("compact_wan", {}).get("future_video_size")
        )
        if exported_future_size != expected_future_size:
            logger.warning(
                "EfficientWAM init checkpoint future_video_size differs from current config for %s: "
                "expected %r, got %r. Loading is allowed because future_video_size controls runtime "
                "multiscale layout and does not change parameter shapes.",
                checkpoint_path,
                expected_future_size,
                exported_future_size,
            )

    def _load_efficient_wam_init(self, model: SmallWAMActionModel) -> bool:
        model_cfg = self.config["model"]
        checkpoint_path = model_cfg.get("efficient_wam_init_checkpoint")
        if not checkpoint_path:
            return False

        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported EfficientWAM init checkpoint format at {checkpoint_path}")

        self._validate_efficient_wam_checkpoint_metadata(
            checkpoint_path,
            payload if isinstance(payload, dict) else {},
            model_cfg,
        )
        state_dict = self._strip_module_prefix(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "EfficientWAM init checkpoint does not match current model: "
                f"missing={missing}, unexpected={unexpected}, checkpoint={checkpoint_path}"
            )
        logger.info("Loaded full EfficientWAM init from %s (%d tensors)", checkpoint_path, len(state_dict))
        return True

    def _build_system(self) -> VideoActionSystem:
        compact_wan = self._build_compact_wan()
        model_cfg = self.config["model"]
        small_wam_cfg = SmallWAMActionConfig(
            compact_wan=compact_wan.config,
            action_dim=int(model_cfg["action_expert"]["action_dim"]),
            state_dim=int(model_cfg["action_expert"]["state_dim"]),
            chunk_size=int(model_cfg["action_expert"]["chunk_size"]),
            ae_dim=int(model_cfg["action_expert"]["dim"]),
            ae_ffn_dim=int(model_cfg["action_expert"]["ffn_dim"]),
            ae_num_layers=int(model_cfg["action_expert"]["num_layers"]),
            wan_frozen=bool(model_cfg["wan_frozen"]),
        )
        model = SmallWAMActionModel(config=small_wam_cfg, compact_wan=compact_wan).to(self.device)
        loaded_full_init = self._load_efficient_wam_init(model)
        if loaded_full_init:
            if model_cfg.get("action_init_checkpoint"):
                logger.info("Applying Action Expert init after full EfficientWAM init")
                self._load_action_expert_init(model)
        else:
            self._load_action_expert_init(model)
        self._configure_trainable_parameters(model)
        return VideoActionSystem(model=model, config=self.config)

    @staticmethod
    def _count_trainable_parameters(parameters) -> int:
        return sum(p.numel() for p in parameters if p.requires_grad)

    @staticmethod
    def _optimizer_param_count(optimizer, parameters) -> int:
        param_ids = {id(p) for p in parameters}
        return sum(
            p.numel()
            for group in optimizer.param_groups
            for p in group["params"]
            if id(p) in param_ids
        )

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self._model().parameters() if p.requires_grad]

    def _configure_trainable_parameters(self, model: SmallWAMActionModel) -> None:
        for param in model.action_expert.parameters():
            param.requires_grad_(True)
        for param in model.compact_wan.parameters():
            param.requires_grad_(False)
        if not model.config.wan_frozen:
            for param in model.compact_wan.video_model.wan_model.parameters():
                param.requires_grad_(True)

    def _build_optimizer_and_scheduler(self):
        opt_cfg = get_optimizer_config(self.config)
        model = self.system.model
        if not model.config.wan_frozen:
            missing_lrs = [
                key
                for key in ("action_learning_rate", "video_learning_rate")
                if key not in self.config
            ]
            if missing_lrs:
                raise ValueError(
                    "Stage 3 joint mode requires explicit lr keys: "
                    + ", ".join(missing_lrs)
                )

        action_params = [p for p in model.action_expert.parameters() if p.requires_grad]
        if not action_params:
            raise RuntimeError("Video-action training expected trainable Action Expert parameters")

        param_groups = [
            {
                "name": "action_expert",
                "params": action_params,
                "lr": opt_cfg["action_learning_rate"],
                "weight_decay": opt_cfg["action_weight_decay"],
            }
        ]
        if not model.config.wan_frozen:
            video_params = [
                p
                for p in model.compact_wan.video_model.wan_model.parameters()
                if p.requires_grad
            ]
            if not video_params:
                raise RuntimeError("Stage 3 joint mode expected trainable compact WAN video expert parameters")
            param_groups.append(
                {
                    "name": "video_expert",
                    "params": video_params,
                    "lr": opt_cfg["video_learning_rate"],
                    "weight_decay": opt_cfg["video_weight_decay"],
                }
            )

        optimizer = AdamW(param_groups, lr=param_groups[0]["lr"], weight_decay=opt_cfg["weight_decay"])

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
                "learning_rate": param_groups[0]["lr"],
                "min_lr_ratio": opt_cfg["min_lr_ratio"],
                "min_lr": self.config.get("min_lr"),
            },
        )()
        return optimizer, create_scheduler(optimizer, cfg)

    def _verify_trainable_setup_before_prepare(self) -> None:
        model = self.system.model
        self._verify_trainable_setup(model, "before distributed wrapping", verify_optimizer=True)

    def _verify_trainable_setup_after_prepare(self) -> None:
        model = self._model()
        self._verify_trainable_setup(model, "after distributed wrapping", verify_optimizer=False)

    def _verify_trainable_setup(
        self,
        model: SmallWAMActionModel,
        context: str,
        verify_optimizer: bool,
    ) -> None:
        action_trainable = self._count_trainable_parameters(model.action_expert.parameters())
        video_trainable = self._count_trainable_parameters(model.compact_wan.video_model.wan_model.parameters())
        wan_trainable = self._count_trainable_parameters(model.compact_wan.parameters())
        action_optimizer_params = 0
        video_optimizer_params = 0
        wan_optimizer_params = 0
        if verify_optimizer:
            action_optimizer_params = self._optimizer_param_count(self.optimizer, model.action_expert.parameters())
            video_optimizer_params = self._optimizer_param_count(
                self.optimizer,
                model.compact_wan.video_model.wan_model.parameters(),
            )
            wan_optimizer_params = self._optimizer_param_count(self.optimizer, model.compact_wan.parameters())

        if action_trainable == 0:
            raise RuntimeError(
                f"Video-action training expected Action Expert trainable {context}, "
                f"found action_trainable={action_trainable}"
            )
        if verify_optimizer and action_optimizer_params != action_trainable:
            raise RuntimeError(
                f"Video-action training expected Action Expert fully included in optimizer {context}, "
                f"found action_trainable={action_trainable}, action_optimizer_params={action_optimizer_params}"
            )

        if model.config.wan_frozen:
            if wan_trainable != 0:
                raise RuntimeError(
                    f"Stage 2 frozen mode expected compact WAN frozen {context}, "
                    f"found wan_trainable={wan_trainable}"
                )
            if verify_optimizer and wan_optimizer_params != 0:
                raise RuntimeError(
                    f"Stage 2 frozen mode expected compact WAN excluded from optimizer {context}, "
                    f"found wan_optimizer_params={wan_optimizer_params}"
                )
        elif video_trainable == 0:
            raise RuntimeError(
                f"Stage 3 joint mode expected video expert trainable {context}, "
                f"found video_trainable={video_trainable}"
            )
        elif wan_trainable != video_trainable:
            raise RuntimeError(
                f"Stage 3 joint mode expected only compact WAN video expert trainable {context}, "
                f"found wan_trainable={wan_trainable}, video_trainable={video_trainable}"
            )
        elif verify_optimizer and video_optimizer_params != video_trainable:
            raise RuntimeError(
                f"Stage 3 joint mode expected video expert fully included in optimizer {context}, "
                f"found video_trainable={video_trainable}, video_optimizer_params={video_optimizer_params}"
            )
        elif verify_optimizer and wan_optimizer_params != video_optimizer_params:
            raise RuntimeError(
                f"Stage 3 joint mode expected only compact WAN video expert in optimizer {context}, "
                f"found wan_optimizer_params={wan_optimizer_params}, video_optimizer_params={video_optimizer_params}"
            )

        if self.accelerator.is_main_process:
            mode = "frozen-video" if model.config.wan_frozen else "joint-video-action"
            logger.info(
                "%s trainable setup verified %s: mode=%s action_trainable=%d action_optimizer=%d "
                "video_trainable=%d video_optimizer=%d compact_wan_trainable=%d compact_wan_optimizer=%d",
                self.stage_title,
                context,
                mode,
                action_trainable,
                action_optimizer_params,
                video_trainable,
                video_optimizer_params,
                wan_trainable,
                wan_optimizer_params,
            )

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        model = self._model()
        prepared = dict(batch)
        dtype = model.compact_wan.video_model.precision
        prepared["initial_state"] = prepared["initial_state"].to(self.device, dtype=dtype, non_blocking=True)
        actions = prepared["action_sequence"].to(self.device, dtype=dtype, non_blocking=True)
        batch_size = actions.shape[0]

        if "clean_latent" in prepared and "condition_latent" in prepared:
            clean_latent = prepared["clean_latent"].to(self.device, dtype=dtype, non_blocking=True)
            condition_latent = prepared["condition_latent"].to(self.device, dtype=dtype, non_blocking=True)
        elif "future_latent" in prepared and "condition_latent" in prepared:
            clean_future_latent = prepared["future_latent"].to(self.device, dtype=dtype, non_blocking=True)
            condition_latent = prepared["condition_latent"].to(self.device, dtype=dtype, non_blocking=True)
            clean_latent = None
        else:
            first_frame = prepared["first_frame"].to(self.device, dtype=dtype, non_blocking=True)
            video_frames = prepared["video_frames"].to(self.device, dtype=dtype, non_blocking=True)
            prepared["first_frame"] = first_frame
            prepared["video_frames"] = video_frames

            first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
            video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
            full_video = torch.cat([first_frame_norm, video_normalized], dim=2)

            with torch.no_grad():
                clean_latent = model.compact_wan.encode_video(full_video)
                if self._reuse_condition_latent_from_clean():
                    condition_latent = clean_latent[:, :, 0:1]
                    if self._verify_condition_latent_reuse() and not self._condition_latent_reuse_checked:
                        encoded_condition_latent = model.compact_wan.encode_video(first_frame_norm)
                        max_abs_diff = (condition_latent.float() - encoded_condition_latent.float()).abs().max()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "Video-action condition latent reuse check max_abs_diff=%.6g",
                                float(max_abs_diff.item()),
                            )
                        self._condition_latent_reuse_checked = True
                else:
                    condition_latent = model.compact_wan.encode_video(first_frame_norm)

        video_timestep_id = torch.randint(
            0,
            self.fm_train_scheduler_video.num_train_timesteps,
            (batch_size,),
            device=self.device,
        )
        video_t = self._video_timesteps[video_timestep_id]
        video_sigma = self._video_sigmas[video_timestep_id].view(batch_size, 1, 1, 1, 1)
        if clean_latent is None:
            video_noise = torch.randn_like(clean_future_latent, dtype=dtype)
            future_latent = clean_future_latent * (1 - video_sigma) + video_noise * video_sigma
            video_target = video_noise - clean_future_latent
            prepared["condition_latent"] = condition_latent
            prepared["future_latent"] = future_latent
        else:
            video_noise = torch.randn_like(clean_latent, dtype=dtype)
            video_latent = clean_latent * (1 - video_sigma) + video_noise * video_sigma
            video_latent[:, :, 0:1] = condition_latent
            video_target = video_noise - clean_latent
            video_target[:, :, 0:1] = 0
            prepared["video_latent"] = video_latent

        timestep_id = torch.randint(0, self.fm_train_scheduler_action.num_train_timesteps, (batch_size,), device=self.device)
        action_t = self._action_timesteps[timestep_id]
        sigma = self._action_sigmas[timestep_id].view(batch_size, 1, 1)
        noise = torch.randn_like(actions, dtype=dtype)
        prepared["noisy_actions"] = actions * (1 - sigma) + noise * sigma
        prepared["action_target"] = noise - actions
        prepared["action_t"] = action_t
        prepared["video_target"] = video_target
        prepared["video_t"] = video_t
        prepared["text_embeddings"] = [
            value.to(self.device, dtype=dtype, non_blocking=True)
            for value in batch["text_embeddings"]
        ]
        return prepared

    @staticmethod
    def _regression_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction.float(), target.float(), reduction="mean")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float] | None:
        self.system.model.train()
        grad_norm = None
        with self.accelerator.accumulate(self.system.model):
            prepared = self._prepare_batch(batch)
            outputs = self.system.model(prepared)
            action_loss = self._regression_loss(outputs["action_pred"], prepared["action_target"])
            if "future_latent" in prepared:
                video_loss = self._regression_loss(outputs["video_pred"], prepared["video_target"])
            else:
                video_loss = self._regression_loss(
                    outputs["video_pred"][:, :, 1:],
                    prepared["video_target"][:, :, 1:],
                )
            loss_cfg = self.config.get("loss", {})
            action_weight = float(loss_cfg.get("action_weight", 1.0))
            video_weight = float(loss_cfg.get("video_weight", 0.05))
            total_loss = action_weight * action_loss + video_weight * video_loss
            self.accelerator.backward(total_loss)
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self._trainable_parameters(),
                    max_norm=float(self.config.get("grad_clip_norm", 1.0)),
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        if not self.accelerator.sync_gradients:
            return None
        grad_norm_value = 0.0
        if grad_norm is not None:
            grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
        return {
            "loss/total": float(total_loss.detach().item()),
            "loss/action": float(action_loss.detach().item()),
            "loss/video": float(video_loss.detach().item()),
            "loss_weight/action": action_weight,
            "loss_weight/video": video_weight,
            "grad/norm": grad_norm_value,
            **get_learning_rate_metrics(self.optimizer),
        }

    def _resume_if_needed(self) -> None:
        resume_from = self.config.get("resume_from")
        if not resume_from:
            return
        self.accelerator.load_state(str(resume_from))
        match = re.search(r"step_(\d+)", str(resume_from))
        if match:
            self.global_step = int(match.group(1))
        logger.info("Resumed %s training from %s at step %d", self.stage_title, resume_from, self.global_step)

    def save_checkpoint(self) -> None:
        ckpt_dir = get_run_dir(self.config) / f"step_{self.global_step}"
        self.accelerator.save_state(str(ckpt_dir))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            logger.info("Saved %s resume checkpoint to %s", self.stage_title, ckpt_dir)
            self.export_efficient_wam()

    def export_efficient_wam(self) -> None:
        export_dir = get_export_dir(self.config)
        export_dir.mkdir(parents=True, exist_ok=True)
        model_state = {key: value.cpu() for key, value in self.accelerator.get_state_dict(self.system.model).items()}
        output_path = export_dir / f"{self.stage_name}_step_{self.global_step}.pt"
        torch.save(
            {
                "model": model_state,
                "global_step": self.global_step,
                "config": self._checkpoint_config(),
            },
            output_path,
        )
        logger.info("Exported EfficientWAM checkpoint to %s", output_path)

    @staticmethod
    def _action_normalization_from_train_dataset(dataset_root: str | Path) -> Dict[str, Any] | None:
        root = Path(dataset_root).expanduser()
        metadata_path = root / TRAIN_DATASET_METADATA
        if not metadata_path.exists():
            return None
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        data_cfg = metadata.get("data", {})
        norm_cfg = data_cfg.get("action_normalization")
        if not isinstance(norm_cfg, dict):
            return None

        exported_norm = {
            key: norm_cfg[key]
            for key in ("enabled", "type")
            if key in norm_cfg
        }
        stats_path = root / TRAIN_DATASET_ACTION_STATS
        if exported_norm.get("enabled"):
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"Action normalization is enabled but {TRAIN_DATASET_ACTION_STATS} is missing: {stats_path}"
                )
            with stats_path.open("r", encoding="utf-8") as f:
                stats_payload = json.load(f)
            exported_norm["stats"] = stats_payload.get("robotwin_qpos", stats_payload)
        stats_file = norm_cfg.get("stats_file")
        if stats_file:
            exported_norm["stats_file"] = str(stats_file)
        return exported_norm

    def _checkpoint_config(self) -> Dict[str, Any]:
        checkpoint_config = dict(self.config)
        checkpoint_config["training_stage"] = self.stage_name
        dataset_cfg = checkpoint_config.get("dataset")
        dataset_root = dataset_cfg.get("root") if isinstance(dataset_cfg, dict) else None
        if dataset_root:
            norm_cfg = self._action_normalization_from_train_dataset(dataset_root)
            if norm_cfg is not None:
                checkpoint_config["action_normalization"] = norm_cfg
        return checkpoint_config

    def run(self) -> None:
        self._verify_trainable_setup_after_prepare()
        if self.accelerator.is_main_process:
            logger.info("Built %s EfficientWAM system", self.stage_title)
            logger.info(
                "%s mode: %s",
                self.stage_title,
                "frozen-video" if self._model().config.wan_frozen else "joint-video-action",
            )
            logger.info("WAN frozen: %s", self._model().config.wan_frozen)
            logger.info("AE layers: %d", self._model().config.ae_num_layers)
            logger.info("Run directory: %s", get_run_dir(self.config))
            
            loss_cfg = self.config.get("loss", {})
            action_w = float(loss_cfg.get("action_weight", 1.0))
            video_w = float(loss_cfg.get("video_weight", 0.05))
            
            logger.info("")
            logger.info("Video-action training config")
            logger.info("  max_steps      : %d", int(self.config["max_steps"]))
            logger.info("  log_interval   : %d", int(self.config.get("log_interval", 10)))
            logger.info("  loss_weights   : action=%s video=%s", action_w, video_w)

        loss_precision = 3 if self._model().config.wan_frozen else 4
        self.log_tracker = LogTracker(self.stage_name, int(self.config["max_steps"]), loss_precision=loss_precision)
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
                            ("action", "loss/action"),
                            ("video", "loss/video"),
                        ],
                        lr_keys=[
                            ("action", "lr/action_expert"),
                            ("video", "lr/video_expert"),
                        ],
                    )
                    for line in lines_to_log:
                        logger.info("%s", line)

            if self.global_step % checkpoint_interval == 0:
                self.save_checkpoint()

        if self.global_step == 0 or self.global_step % checkpoint_interval != 0:
            self.save_checkpoint()


def run_video_action_training(
    *,
    default_config: str,
    description: str,
    default_project: str,
) -> None:
    parser = build_arg_parser(description=description, default_config=default_config)
    args = parser.parse_args()
    config = apply_cli_overrides(load_yaml_config(args.config), args)
    config.setdefault("training_stage", default_project)
    accelerator = build_accelerator(args, config)
    setup_logging(args.log_level, rank=accelerator.process_index)
    init_experiment_trackers(
        accelerator,
        config,
        f"efficient_wam_{default_project}",
        default_project=default_project,
    )
    trainer = VideoActionTrainer(config=config, accelerator=accelerator)
    trainer.run()
    accelerator.end_training()


def main() -> None:
    run_video_action_training(
        default_config="configs/robotwin/stage2_action.yaml",
        description="Stage 2 EfficientWAM action training",
        default_project="stage2",
    )


if __name__ == "__main__":
    main()
