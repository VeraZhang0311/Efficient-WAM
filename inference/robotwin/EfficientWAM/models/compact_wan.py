from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .wan_model import WanVideoModel


@dataclass
class CompactWANConfig:
    """Configuration for the Stage 1/2 compact WAN backbone."""

    checkpoint_path: str
    vae_path: str
    config_path: Optional[str] = None
    precision: str = "bfloat16"
    dim: int = 2048
    ffn_dim: int = 8192
    num_heads: int = 16
    num_layers: int = 12
    head_dim: int = 128
    future_video_size: Optional[Tuple[int, int]] = None
    hidden_anchor_layers: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 12])
    motion_anchor_layers: List[int] = field(default_factory=lambda: [6, 8, 10, 12])
    teacher_layer_mapping: List[int] = field(
        default_factory=lambda: [1, 2, 4, 6, 8, 11, 14, 17, 20, 23, 26, 30]
    )

    def to_wan_model_config(self) -> Dict[str, int]:
        return {
            "dim": self.dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }


class CompactWANModel(nn.Module):
    """Compact WAN backbone used by EfficientWAM."""

    def __init__(self, config: CompactWANConfig, video_model: WanVideoModel):
        super().__init__()
        self.config = config
        self.video_model = video_model

    @classmethod
    def from_teacher_checkpoint(
        cls,
        config: CompactWANConfig,
        device: str = "cuda",
    ) -> "CompactWANModel":
        """Build the compact WAN wrapper from the teacher checkpoint.

        The full structured slicing initialization is wired into `WanVideoModel`
        in a later refactor step. For now this method centralizes construction so
        the trainer APIs are stable while we deepen the WAN loading path.
        """

        video_model = WanVideoModel.from_pretrained_compact(
            checkpoint_path=config.checkpoint_path,
            vae_path=config.vae_path,
            student_model_config=config.to_wan_model_config(),
            teacher_layer_mapping=config.teacher_layer_mapping,
            config_path=config.config_path or config.checkpoint_path,
            device=device,
            precision=config.precision,
        )
        return cls(config=config, video_model=video_model)

    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        return self.video_model.encode_video(video_pixels)

    def decode_video(self, video_latents: torch.Tensor) -> torch.Tensor:
        return self.video_model.decode_video(video_latents)

    def prepare_video_tokens(
        self,
        video_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.video_model.prepare_video_tokens(video_latent)

    @property
    def is_multiscale(self) -> bool:
        return self.config.future_video_size is not None

    def prepare_multiscale_video_tokens(
        self,
        condition_latent: torch.Tensor,
        future_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor | int], torch.Tensor]:
        return self.video_model.prepare_multiscale_video_tokens(condition_latent, future_latent)

    def prepare_text_context(self, text_embeddings: List[torch.Tensor]) -> torch.Tensor:
        return self.video_model.prepare_text_context(text_embeddings)

    def prepare_time_embeddings(
        self,
        timestep: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.video_model.prepare_time_embeddings(timestep, seq_len)

    def apply_video_head(
        self,
        video_tokens: torch.Tensor,
        video_time_emb: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        return self.video_model.apply_video_head(video_tokens, video_time_emb, grid_sizes)

    def apply_multiscale_video_head(
        self,
        video_tokens: torch.Tensor,
        video_time_emb: torch.Tensor,
        layout: Dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        return self.video_model.apply_multiscale_video_head(video_tokens, video_time_emb, layout)

    def apply_multiscale_rope(
        self,
        heads: torch.Tensor,
        layout: Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        return self.video_model.apply_multiscale_rope(heads, layout, freqs)

    def extract_hidden_features(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        layer_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        return self.video_model.get_layer_features(
            video_latent=x_t,
            timestep=timestep,
            text_embeddings=text_embeddings,
            layer_indices=layer_indices,
        )

    def forward_with_features(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        layer_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        if layer_indices is None:
            layer_indices = sorted(set(self.config.hidden_anchor_layers + self.config.motion_anchor_layers))
        features = self.extract_hidden_features(
            x_t=x_t,
            timestep=timestep,
            text_embeddings=text_embeddings,
            layer_indices=layer_indices,
        )
        hidden = {layer: feats for layer, feats in zip(layer_indices, features[:-1])}
        final_output = features[-1]
        return final_output, hidden

    def metadata(self) -> Dict[str, object]:
        return asdict(self.config)
