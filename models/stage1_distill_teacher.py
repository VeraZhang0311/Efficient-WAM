from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .teacher_targets import TeacherTargetProvider, TeacherTargets
from .wan_model import WanVideoModel


@dataclass
class Stage1DistillTeacherConfig:
    checkpoint_path: str
    vae_path: Optional[str] = None
    config_path: Optional[str] = None
    precision: str = "bfloat16"
    load_vae: bool = False
    hidden_anchor_teacher_layers: List[int] = None
    motion_anchor_teacher_layers: List[int] = None

    def __post_init__(self) -> None:
        if self.hidden_anchor_teacher_layers is None:
            self.hidden_anchor_teacher_layers = [1, 4, 8, 14, 20, 30]
        if self.motion_anchor_teacher_layers is None:
            self.motion_anchor_teacher_layers = [11, 17, 23, 30]


class FixedPCATeacherProjector:
    """Fixed teacher-side projector loaded from PCA prep artifacts."""

    def __init__(self, stats_path: str, device: str = "cpu"):
        payload = torch.load(stats_path, map_location=device)
        self.projection_dim = int(payload["projection_dim"])
        self.layer_stats = payload["layers"]
        self.device = device
        self._tensor_cache: Dict[tuple[str, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _layer_tensors(
        self,
        layer: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_key = str(layer)
        if layer_key not in self.layer_stats:
            available = ", ".join(sorted(self.layer_stats.keys(), key=int))
            raise KeyError(f"PCA stats missing teacher layer {layer}; available layers: {available}")

        cache_key = (layer_key, str(device), dtype)
        cached = self._tensor_cache.get(cache_key)
        if cached is not None:
            return cached

        stats = self.layer_stats[layer_key]
        mean = stats["mean"].to(device=device, dtype=dtype, non_blocking=True)
        components = stats["components"].to(device=device, dtype=dtype, non_blocking=True)
        self._tensor_cache[cache_key] = (mean, components)
        return mean, components

    def project(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        mean, components = self._layer_tensors(layer, hidden.device, hidden.dtype)
        centered = F.layer_norm(hidden, (hidden.shape[-1],)) - mean
        return centered @ components


class Stage1DistillTeacherProvider(TeacherTargetProvider):
    """Teacher backend for Stage 1 distillation."""

    def __init__(
        self,
        config: Stage1DistillTeacherConfig,
        student_hidden_layers: List[int],
        student_motion_layers: List[int],
        pca_stats_path: str,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device
        self.student_hidden_layers = list(student_hidden_layers)
        self.student_motion_layers = list(student_motion_layers)
        self.pca_projector = FixedPCATeacherProjector(pca_stats_path, device="cpu")
        self.teacher = WanVideoModel.from_pretrained(
            checkpoint_path=config.checkpoint_path,
            vae_path=config.vae_path,
            config_path=config.config_path,
            device=device,
            precision=config.precision,
            load_vae=config.load_vae,
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def get_teacher_targets(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        include_hidden: bool = True,
        include_motion: bool = True,
        include_video_pred: bool = True,
    ) -> TeacherTargets:
        x_t = batch["x_t"]
        timestep = batch["t"]
        text_embeddings = batch["text_embeddings"]
        hidden_teacher_layers = self.config.hidden_anchor_teacher_layers if include_hidden else []
        motion_teacher_layers = self.config.motion_anchor_teacher_layers if include_motion else []
        hidden_layers = sorted(set(hidden_teacher_layers + motion_teacher_layers))
        if isinstance(x_t, dict):
            features = self.teacher.get_multiscale_layer_features(
                condition_latent=x_t["condition_latent"],
                future_latent=x_t["future_latent"],
                timestep=timestep,
                text_embeddings=text_embeddings,
                layer_indices=hidden_layers,
            )
        else:
            features = self.teacher.get_layer_features(
                video_latent=x_t,
                timestep=timestep,
                text_embeddings=text_embeddings,
                layer_indices=hidden_layers,
            )

        raw_by_layer = {layer: feature for layer, feature in zip(hidden_layers, features[:-1])}

        hidden_targets: Dict[int, torch.Tensor] = {}
        for student_layer, teacher_layer in zip(self.student_hidden_layers, hidden_teacher_layers):
            hidden_targets[student_layer] = self.pca_projector.project(teacher_layer, raw_by_layer[teacher_layer])

        motion_targets: Dict[int, torch.Tensor] = {}
        latent_frames = (
            int(batch["num_motion_frames"])
            if "num_motion_frames" in batch
            else int(batch["clean_latent"].shape[2])
        )
        condition_tokens = int(batch.get("condition_tokens", 0))
        for student_layer, teacher_layer in zip(self.student_motion_layers, motion_teacher_layers):
            projected = self.pca_projector.project(teacher_layer, raw_by_layer[teacher_layer])
            if condition_tokens:
                projected = projected[:, condition_tokens:]
            batch_size, num_tokens, dim = projected.shape
            if num_tokens % latent_frames != 0:
                raise ValueError(
                    f"Teacher layer {teacher_layer} produced {num_tokens} tokens incompatible with {latent_frames} latent frames"
                )
            tokens_per_frame = num_tokens // latent_frames
            frames = projected.reshape(batch_size, latent_frames, tokens_per_frame, dim).mean(dim=2)
            motion_targets[student_layer] = frames[:, 1:] - frames[:, :-1]
        return TeacherTargets(
            hidden_targets=hidden_targets,
            motion_targets=motion_targets,
            video_pred=features[-1] if include_video_pred else None,
        )
