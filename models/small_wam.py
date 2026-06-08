from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

from .action_expert import ActionExpert, ActionExpertConfig
from .compact_wan import CompactWANConfig, CompactWANModel
from third_party.wan.modules.attention import flash_attention
from third_party.wan.modules.model import sinusoidal_embedding_1d


@dataclass
class SmallWAMActionConfig:
    compact_wan: CompactWANConfig
    action_dim: int = 14
    state_dim: int = 14
    chunk_size: int = 16
    ae_dim: int = 768
    ae_ffn_dim: int = 3072
    ae_num_layers: int = 12
    wan_frozen: bool = True


class SmallWAMActionModel(nn.Module):
    """EfficientWAM MoT model for video-action flow matching."""

    def __init__(
        self,
        config: SmallWAMActionConfig,
        compact_wan: CompactWANModel,
        action_expert: Optional[ActionExpert] = None,
    ):
        super().__init__()
        self.config = config
        self.compact_wan = compact_wan
        if config.ae_num_layers != config.compact_wan.num_layers:
            raise ValueError(
                "EfficientWAM MoT requires one action expert block per compact WAN block: "
                f"got ae_num_layers={config.ae_num_layers}, compact_wan.num_layers={config.compact_wan.num_layers}"
            )

        wan_cfg = {
            "dim": config.compact_wan.dim,
            "num_heads": config.compact_wan.num_heads,
            "head_dim": config.compact_wan.head_dim,
        }
        self.action_expert = action_expert or ActionExpert(
            ActionExpertConfig(
                dim=config.ae_dim,
                ffn_dim=config.ae_ffn_dim,
                num_layers=config.ae_num_layers,
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                chunk_size=config.chunk_size,
                video_feature_dim=config.compact_wan.dim,
            ),
            wan_cfg,
        )
        if config.wan_frozen:
            for param in self.compact_wan.parameters():
                param.requires_grad_(False)

        self.enable_teacache = False
        self.teacache_delta = 0.0
        self.teacache_accumulated_diff = 0.0
        self.teacache_previous_indicator: Optional[torch.Tensor] = None
        self.teacache_cached_video_residual: Optional[torch.Tensor] = None
        self.teacache_cached_action_residual: Optional[torch.Tensor] = None
        self.teacache_step_idx = 0
        self.teacache_num_steps = 0
        self.teacache_poly_coefficients: Optional[tuple[float, ...]] = None
        self.teacache_sync_distributed = False
        self.teacache_force_last_step = False
        self.teacache_num_cache_hits = 0
        self.teacache_num_cache_misses = 0
        self._rope_freq_grid_cache: Dict[tuple[object, ...], torch.Tensor] = {}

    @staticmethod
    def _expand_time_for_tokens(t: torch.Tensor, seq_len: int) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)
        return t

    def _build_action_time_embeddings(
        self,
        t: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._expand_time_for_tokens(t, seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=t.is_cuda):
            batch = t.size(0)
            flat_t = t.flatten()
            emb = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, flat_t)
                .unflatten(0, (batch, seq_len))
                .float()
                .to(t.device)
            )
            adaln = self.action_expert.time_projection(emb).unflatten(2, (6, self.config.ae_dim))
        return emb, adaln

    def _build_video_time_embeddings(
        self,
        t: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.compact_wan.prepare_time_embeddings(t, seq_len)

    @staticmethod
    def _block_modulation(block, adaln_params: torch.Tensor):
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=adaln_params.is_cuda):
            return (block.modulation.unsqueeze(0) + adaln_params).chunk(6, dim=2)

    @staticmethod
    def _action_qkv_for_wan(block, norm_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = block.wan_action_qkv.to(device=norm_action.device, dtype=norm_action.dtype)
        action_qkv = torch.einsum("btd,hdf->bthf", norm_action, qkv[0]), \
            torch.einsum("btd,hdf->bthf", norm_action, qkv[1]), \
            torch.einsum("btd,hdf->bthf", norm_action, qkv[2])
        action_q_h, action_k_h, action_v_h = action_qkv
        batch, seq_len, heads, head_dim = action_q_h.shape
        action_q = block.wan_action_norm_q(action_q_h.flatten(2)).view(batch, seq_len, heads, head_dim)
        action_k = block.wan_action_norm_k(action_k_h.flatten(2)).view(batch, seq_len, heads, head_dim)
        action_v = action_v_h.view(batch, seq_len, heads, head_dim)
        return action_q, action_k, action_v

    def _cached_rope_freq_grid(
        self,
        freqs: torch.Tensor,
        grid_shape: tuple[int, int, int],
        complex_dim: int,
    ) -> torch.Tensor:
        f, h, w = (int(value) for value in grid_shape)
        key = (
            freqs.device.type,
            freqs.device.index,
            str(freqs.dtype),
            int(freqs.data_ptr()),
            int(complex_dim),
            f,
            h,
            w,
        )
        cached = self._rope_freq_grid_cache.get(key)
        if cached is not None:
            return cached

        c_f = complex_dim - 2 * (complex_dim // 3)
        c_h = complex_dim // 3
        c_w = complex_dim // 3
        fpart, hpart, wpart = freqs.split([c_f, c_h, c_w], dim=1)
        freq_grid = torch.cat(
            [
                fpart[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                hpart[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                wpart[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, complex_dim).contiguous()
        self._rope_freq_grid_cache[key] = freq_grid
        return freq_grid

    def _rope_apply_exact_grid(
        self,
        heads: torch.Tensor,
        grid_shape: tuple[int, int, int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, num_heads, head_dim = heads.shape
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        expected_seq_len = int(grid_shape[0]) * int(grid_shape[1]) * int(grid_shape[2])
        if seq_len != expected_seq_len:
            raise ValueError(
                "RoPE exact-grid fast path received mismatched sequence length: "
                f"{seq_len} != {expected_seq_len}"
            )
        complex_dim = head_dim // 2
        with torch.amp.autocast("cuda", enabled=False):
            freq_grid = self._cached_rope_freq_grid(freqs, grid_shape, complex_dim)
            heads_complex = torch.view_as_complex(
                heads.to(torch.float64).reshape(batch, seq_len, num_heads, complex_dim, 2)
            ).contiguous()
            rotated = heads_complex * freq_grid
            return torch.view_as_real(rotated).reshape(batch, seq_len, num_heads, head_dim).float()

    def _multiscale_joint_attention_fast(
        self,
        attn: nn.Module,
        norm_video: torch.Tensor,
        action_q: torch.Tensor,
        action_k: torch.Tensor,
        action_v: torch.Tensor,
        joint_seq_lens: torch.Tensor,
        grid_sizes: Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
        attention_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = norm_video.shape[:2]
        action_len = action_q.shape[1]
        condition_seq_len = int(grid_sizes["condition_seq_len"])
        future_seq_len = int(grid_sizes["future_seq_len"])
        if condition_seq_len + future_seq_len != seq_len:
            raise ValueError(
                "Multiscale sequence layout does not match video tokens: "
                f"{condition_seq_len} + {future_seq_len} != {seq_len}"
            )
        condition_grid_shape = grid_sizes.get("condition_grid_shape")
        future_grid_shape = grid_sizes.get("future_grid_shape")
        if condition_grid_shape is None or future_grid_shape is None:
            video_q = attn.norm_q(attn.q(norm_video)).view(batch, seq_len, attn.num_heads, attn.head_dim)
            video_k = attn.norm_k(attn.k(norm_video)).view(batch, seq_len, attn.num_heads, attn.head_dim)
            video_v = attn.v(norm_video).view(batch, seq_len, attn.num_heads, attn.head_dim)
            video_q = self.compact_wan.apply_multiscale_rope(video_q, grid_sizes, freqs)
            video_k = self.compact_wan.apply_multiscale_rope(video_k, grid_sizes, freqs)
            attended = flash_attention(
                q=torch.cat([video_q, action_q.to(dtype=attention_dtype)], dim=1),
                k=torch.cat([video_k, action_k.to(dtype=attention_dtype)], dim=1),
                v=torch.cat([video_v, action_v.to(dtype=attention_dtype)], dim=1),
                k_lens=joint_seq_lens,
                window_size=attn.window_size,
            )
            return attn.o(attended[:, :seq_len].flatten(2)), attended[:, seq_len:]

        condition_grid_shape = tuple(int(value) for value in condition_grid_shape)
        future_grid_shape = tuple(int(value) for value in future_grid_shape)
        video_q = attn.norm_q(attn.q(norm_video)).view(batch, seq_len, attn.num_heads, attn.head_dim)
        video_k = attn.norm_k(attn.k(norm_video)).view(batch, seq_len, attn.num_heads, attn.head_dim)
        video_v = attn.v(norm_video).view(batch, seq_len, attn.num_heads, attn.head_dim)

        total_len = seq_len + action_len
        q_cat = video_q.new_empty((batch, total_len, attn.num_heads, attn.head_dim), dtype=torch.float32)
        k_cat = video_k.new_empty((batch, total_len, attn.num_heads, attn.head_dim), dtype=torch.float32)
        v_cat = video_v.new_empty((batch, total_len, attn.num_heads, attn.head_dim), dtype=video_v.dtype)

        condition_slice = slice(0, condition_seq_len)
        future_slice = slice(condition_seq_len, seq_len)
        action_slice = slice(seq_len, total_len)
        q_cat[:, condition_slice] = self._rope_apply_exact_grid(
            video_q[:, condition_slice],
            condition_grid_shape,
            freqs,
        )
        q_cat[:, future_slice] = self._rope_apply_exact_grid(
            video_q[:, future_slice],
            future_grid_shape,
            freqs,
        )
        q_cat[:, action_slice] = action_q.to(dtype=q_cat.dtype)
        k_cat[:, condition_slice] = self._rope_apply_exact_grid(
            video_k[:, condition_slice],
            condition_grid_shape,
            freqs,
        )
        k_cat[:, future_slice] = self._rope_apply_exact_grid(
            video_k[:, future_slice],
            future_grid_shape,
            freqs,
        )
        k_cat[:, action_slice] = action_k.to(dtype=k_cat.dtype)
        v_cat[:, :seq_len] = video_v
        v_cat[:, action_slice] = action_v.to(dtype=v_cat.dtype)

        attended = flash_attention(
            q=q_cat,
            k=k_cat,
            v=v_cat,
            k_lens=joint_seq_lens,
            window_size=attn.window_size,
        )
        return attn.o(attended[:, :seq_len].flatten(2)), attended[:, seq_len:]

    def _joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_modulation: tuple[torch.Tensor, ...],
        action_modulation: tuple[torch.Tensor, ...],
        layer_idx: int,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor | Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wan_layer = self.compact_wan.video_model.wan_model.blocks[layer_idx]
        action_block = self.action_expert.blocks[layer_idx]
        video_mod = video_modulation
        action_mod = action_modulation

        norm_video = wan_layer.norm1(video_tokens).float() * (1 + video_mod[1].squeeze(2)) + video_mod[0].squeeze(2)
        norm_action = (
            action_block.norm1(action_tokens).float() * (1 + action_mod[1].squeeze(2))
            + action_mod[0].squeeze(2)
        )
        action_q, action_k, action_v = self._action_qkv_for_wan(action_block, norm_action)

        joint_seq_lens = seq_lens + action_tokens.shape[1]
        attention_dtype = self.compact_wan.video_model.precision
        with torch.autocast("cuda", dtype=attention_dtype, enabled=norm_video.is_cuda):
            if isinstance(grid_sizes, dict):
                attn = wan_layer.self_attn
                video_out, action_out_h = self._multiscale_joint_attention_fast(
                    attn,
                    norm_video,
                    action_q,
                    action_k,
                    action_v,
                    joint_seq_lens,
                    grid_sizes,
                    freqs,
                    attention_dtype,
                )
            else:
                video_out, action_out_h, _ = wan_layer.self_attn(
                    norm_video,
                    joint_seq_lens,
                    grid_sizes,
                    freqs,
                    action_q=action_q.to(dtype=attention_dtype),
                    action_k=action_k.to(dtype=attention_dtype),
                    action_v=action_v.to(dtype=attention_dtype),
                )

        action_out_h = action_out_h.flatten(2).to(
            device=action_block.wan_action_o.weight.device,
            dtype=action_block.wan_action_o.weight.dtype,
        )
        action_out = action_block.wan_action_o(action_out_h)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=video_tokens.is_cuda):
            video_tokens = video_tokens + video_out * video_mod[2].squeeze(2)
            action_tokens = action_tokens + action_out * action_mod[2].squeeze(2)
        return video_tokens, action_tokens

    def configure_teacache(
        self,
        enabled: bool,
        delta: float = 0.0,
        num_steps: Optional[int] = None,
        poly_coefficients: Optional[Sequence[float]] = None,
        sync_distributed: bool = False,
        force_last_step: bool = False,
    ) -> None:
        self.enable_teacache = bool(enabled)
        self.teacache_delta = float(delta)
        if self.teacache_delta < 0:
            raise ValueError("teacache_delta must be non-negative")
        self.teacache_num_steps = int(num_steps or 0)
        self.teacache_poly_coefficients = (
            tuple(float(coef) for coef in poly_coefficients)
            if poly_coefficients is not None
            else None
        )
        self.teacache_sync_distributed = bool(sync_distributed)
        self.teacache_force_last_step = bool(force_last_step)
        self.reset_teacache(num_steps=num_steps)

    def reset_teacache(self, num_steps: Optional[int] = None) -> None:
        if num_steps is not None:
            self.teacache_num_steps = int(num_steps)
        self.teacache_accumulated_diff = 0.0
        self.teacache_previous_indicator = None
        self.teacache_cached_video_residual = None
        self.teacache_cached_action_residual = None
        self.teacache_step_idx = 0
        self.teacache_num_cache_hits = 0
        self.teacache_num_cache_misses = 0

    def teacache_stats(self) -> Dict[str, float]:
        total = self.teacache_num_cache_hits + self.teacache_num_cache_misses
        return {
            "enabled": float(self.enable_teacache),
            "delta": float(self.teacache_delta),
            "steps": float(self.teacache_step_idx),
            "cache_hits": float(self.teacache_num_cache_hits),
            "cache_misses": float(self.teacache_num_cache_misses),
            "hit_rate": float(self.teacache_num_cache_hits / total) if total else 0.0,
        }

    def _teacache_active(self) -> bool:
        return bool(self.enable_teacache and not self.training and self.config.ae_num_layers > 0)

    def _teacache_indicator(
        self,
        video_tokens: torch.Tensor,
        video_adaln_params: torch.Tensor,
    ) -> torch.Tensor:
        first_wan_layer = self.compact_wan.video_model.wan_model.blocks[0]
        first_modulation = self._block_modulation(first_wan_layer, video_adaln_params)
        return (
            first_wan_layer.norm1(video_tokens).float()
            * (1 + first_modulation[1].squeeze(2))
            + first_modulation[0].squeeze(2)
        )

    def _teacache_relative_l1(
        self,
        indicator: torch.Tensor,
        previous_indicator: torch.Tensor,
    ) -> torch.Tensor:
        num = (indicator - previous_indicator).abs().float().sum()
        den = previous_indicator.abs().float().sum()
        if (
            self.teacache_sync_distributed
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            dist.all_reduce(num, op=dist.ReduceOp.SUM)
            dist.all_reduce(den, op=dist.ReduceOp.SUM)
        return num / den.clamp_min(1e-12)

    def _teacache_estimated_diff(self, rel_l1: torch.Tensor) -> float:
        value = float(rel_l1.detach().cpu())
        if self.teacache_poly_coefficients is None:
            return value
        estimate = 0.0
        for coef in self.teacache_poly_coefficients:
            estimate = estimate * value + coef
        return float(estimate)

    def _teacache_has_residuals(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> bool:
        return (
            self.teacache_cached_video_residual is not None
            and self.teacache_cached_action_residual is not None
            and self.teacache_cached_video_residual.shape == video_tokens.shape
            and self.teacache_cached_action_residual.shape == action_tokens.shape
        )

    def _teacache_should_calc(
        self,
        indicator: torch.Tensor,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> bool:
        previous_indicator = self.teacache_previous_indicator
        is_first_step = self.teacache_step_idx == 0
        is_last_step = (
            self.teacache_num_steps > 0
            and self.teacache_step_idx >= self.teacache_num_steps - 1
        )
        if (
            previous_indicator is None
            or previous_indicator.shape != indicator.shape
            or previous_indicator.device != indicator.device
        ):
            self.teacache_accumulated_diff = 0.0
            self.teacache_previous_indicator = indicator.detach()
            self.teacache_num_cache_misses += 1
            return True

        if is_first_step or (self.teacache_force_last_step and is_last_step):
            self.teacache_accumulated_diff = 0.0
            self.teacache_previous_indicator = indicator.detach()
            self.teacache_num_cache_misses += 1
            return True

        rel_l1 = self._teacache_relative_l1(indicator, previous_indicator)
        estimated_diff = self._teacache_estimated_diff(rel_l1)
        self.teacache_accumulated_diff += estimated_diff
        self.teacache_previous_indicator = indicator.detach()

        if (
            self.teacache_accumulated_diff <= self.teacache_delta
            and self._teacache_has_residuals(video_tokens, action_tokens)
        ):
            self.teacache_num_cache_hits += 1
            return False

        self.teacache_accumulated_diff = 0.0
        self.teacache_num_cache_misses += 1
        return True

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        text_embeddings = batch["text_embeddings"]
        noisy_actions = batch["noisy_actions"]
        action_t = batch["action_t"]
        initial_state = batch["initial_state"]
        video_t = batch["video_t"]
        if "future_latent" in batch:
            video_tokens, seq_lens, grid_sizes, freqs = self.compact_wan.prepare_multiscale_video_tokens(
                batch["condition_latent"],
                batch["future_latent"],
            )
        else:
            video_tokens, seq_lens, grid_sizes, freqs = self.compact_wan.prepare_video_tokens(batch["video_latent"])
        text_context = self.compact_wan.prepare_text_context(text_embeddings)
        video_head_time_emb, video_adaln_params = self._build_video_time_embeddings(
            video_t,
            video_tokens.shape[1],
        )

        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(noisy_actions.shape[0], -1, -1)
        else:
            registers = None

        state_tokens = initial_state.unsqueeze(1)
        action_tokens = self.action_expert.input_encoder(state_tokens, noisy_actions, registers)
        action_head_time_emb, action_adaln_params = self._build_action_time_embeddings(
            action_t,
            action_tokens.shape[1],
        )

        teacache_active = self._teacache_active()
        should_calc = True
        if teacache_active:
            indicator = self._teacache_indicator(video_tokens, video_adaln_params)
            should_calc = self._teacache_should_calc(indicator, video_tokens, action_tokens)

        if should_calc:
            video_origin = video_tokens
            action_origin = action_tokens
            for layer_idx in range(self.config.ae_num_layers):
                wan_layer = self.compact_wan.video_model.wan_model.blocks[layer_idx]
                action_block = self.action_expert.blocks[layer_idx]
                video_modulation = self._block_modulation(wan_layer, video_adaln_params)
                action_modulation = self._block_modulation(action_block, action_adaln_params)

                video_tokens, action_tokens = self._joint_attention(
                    video_tokens,
                    action_tokens,
                    video_modulation,
                    action_modulation,
                    layer_idx,
                    seq_lens,
                    grid_sizes,
                    freqs,
                )

                cross_dtype = self.compact_wan.video_model.precision
                with torch.autocast("cuda", dtype=cross_dtype, enabled=video_tokens.is_cuda):
                    cross_in = wan_layer.norm3(video_tokens)
                    cross_out = wan_layer.cross_attn(cross_in, text_context, None)
                video_tokens = video_tokens + cross_out
                video_ffn_in = (
                    wan_layer.norm2(video_tokens).float() * (1 + video_modulation[4].squeeze(2))
                    + video_modulation[3].squeeze(2)
                )
                video_ffn_weight = wan_layer.ffn[0].weight
                video_ffn = wan_layer.ffn(video_ffn_in.to(device=video_ffn_weight.device, dtype=video_ffn_weight.dtype))
                with torch.amp.autocast("cuda", dtype=torch.float32, enabled=video_tokens.is_cuda):
                    video_tokens = video_tokens + video_ffn * video_modulation[5].squeeze(2)

                action_ffn_in = (
                    action_block.norm2(action_tokens).float() * (1 + action_modulation[4].squeeze(2))
                    + action_modulation[3].squeeze(2)
                )
                action_ffn_weight = action_block.ffn[0].weight
                action_ffn = action_block.ffn(
                    action_ffn_in.to(device=action_ffn_weight.device, dtype=action_ffn_weight.dtype)
                )
                with torch.amp.autocast("cuda", dtype=torch.float32, enabled=action_tokens.is_cuda):
                    action_tokens = action_tokens + action_ffn * action_modulation[5].squeeze(2)

            if teacache_active:
                self.teacache_cached_video_residual = (video_tokens - video_origin).detach()
                self.teacache_cached_action_residual = (action_tokens - action_origin).detach()
        else:
            video_residual = self.teacache_cached_video_residual
            action_residual = self.teacache_cached_action_residual
            assert video_residual is not None and action_residual is not None
            video_tokens = video_tokens + video_residual.to(device=video_tokens.device, dtype=video_tokens.dtype)
            action_tokens = action_tokens + action_residual.to(device=action_tokens.device, dtype=action_tokens.dtype)

        if teacache_active:
            self.teacache_step_idx += 1

        if isinstance(grid_sizes, dict):
            video_pred = self.compact_wan.apply_multiscale_video_head(video_tokens, video_head_time_emb, grid_sizes)
        else:
            video_pred = self.compact_wan.apply_video_head(video_tokens, video_head_time_emb, grid_sizes)
        action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
        trim_len = action_pred_full.shape[1] - self.action_expert.config.num_registers
        action_pred = action_pred_full[:, 1:trim_len, :]
        return {
            "video_pred": video_pred,
            "action_pred": action_pred,
        }
