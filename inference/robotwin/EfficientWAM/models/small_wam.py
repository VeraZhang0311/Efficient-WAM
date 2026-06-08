from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.distributed as dist

from .action_expert import ActionExpert, ActionExpertConfig
from .compact_wan import CompactWANConfig, CompactWANModel
from ..third_party.wan.modules.attention import flash_attention
from ..third_party.wan.modules.model import sinusoidal_embedding_1d
from ..third_party.wan.modules.model import rope_apply


_LATENCY_PROFILE_METRICS = (
    "vgm_video",
    "vgm_prepare",
    "vgm_layer_setup",
    "joint_attn",
    "vgm_cross_attn",
    "vgm_ffn",
    "vgm_head",
    "action_prepare",
    "action_ffn",
    "action_head",
)


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
    """EfficientWAM model for video-action flow matching."""

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
        self.teacache_cached_video_cache: Optional[Dict[str, object]] = None
        self.teacache_step_idx = 0
        self.teacache_num_steps = 0
        self.teacache_poly_coefficients: Optional[tuple[float, ...]] = None
        self.teacache_sync_distributed = False
        self.teacache_force_last_step = False
        self.teacache_num_cache_hits = 0
        self.teacache_num_cache_misses = 0
        self.enable_action_teacache = False
        self.action_teacache_delta = 0.0
        self.action_teacache_accumulated_diff = 0.0
        self.action_teacache_previous_indicator: Optional[torch.Tensor] = None
        self.action_teacache_cached_residual: Optional[torch.Tensor] = None
        self.action_teacache_step_idx = 0
        self.action_teacache_force_last_step = False
        self.action_teacache_num_cache_hits = 0
        self.action_teacache_num_cache_misses = 0
        self._rope_freq_grid_cache: Dict[tuple[object, ...], torch.Tensor] = {}
        self.reset_latency_profile()

    def reset_latency_profile(self) -> None:
        self._latency_profile: Dict[str, object] = {}
        for metric in _LATENCY_PROFILE_METRICS:
            self._latency_profile[f"{metric}_ms_sum"] = 0.0
            self._latency_profile[f"{metric}_calls"] = 0
            self._latency_profile[f"{metric}_cuda_events"] = []

    def consume_latency_profile(self) -> Dict[str, object]:
        profile = self._latency_profile
        self.reset_latency_profile()
        return profile

    def _start_latency_event(self, metric: str, reference: torch.Tensor) -> tuple[str, object, object, Optional[float]]:
        if reference.is_cuda and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            return metric, start_event, end_event, None
        return metric, None, None, time.perf_counter()

    def _finish_latency_event(
        self,
        event: tuple[str, object, object, Optional[float]],
        *,
        count_call: bool = False,
    ) -> None:
        metric, start_event, end_event, wall_start = event
        if count_call:
            calls_key = f"{metric}_calls"
            self._latency_profile[calls_key] = int(self._latency_profile[calls_key]) + 1
        if start_event is not None and end_event is not None:
            end_event.record()
            events = self._latency_profile[f"{metric}_cuda_events"]
            if not isinstance(events, list):
                raise TypeError(f"{metric}_cuda_events must be a list")
            events.append((start_event, end_event))
        elif wall_start is not None:
            sum_key = f"{metric}_ms_sum"
            self._latency_profile[sum_key] = (
                float(self._latency_profile[sum_key])
                + (time.perf_counter() - wall_start) * 1000.0
            )

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

        joint_attn_event = self._start_latency_event("joint_attn", video_tokens)
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
        self._finish_latency_event(joint_attn_event, count_call=True)
        return video_tokens, action_tokens

    def _video_attention_kv_for_cache(
        self,
        video_tokens: torch.Tensor,
        video_modulation: tuple[torch.Tensor, ...],
        layer_idx: int,
        grid_sizes: torch.Tensor | Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wan_layer = self.compact_wan.video_model.wan_model.blocks[layer_idx]
        attn = wan_layer.self_attn
        video_mod = video_modulation
        norm_video = wan_layer.norm1(video_tokens).float() * (1 + video_mod[1].squeeze(2)) + video_mod[0].squeeze(2)
        batch, seq_len = norm_video.shape[:2]
        heads = attn.num_heads
        head_dim = attn.head_dim
        attention_dtype = self.compact_wan.video_model.precision
        with torch.autocast("cuda", dtype=attention_dtype, enabled=norm_video.is_cuda):
            video_k = attn.norm_k(attn.k(norm_video)).view(batch, seq_len, heads, head_dim)
            video_v = attn.v(norm_video).view(batch, seq_len, heads, head_dim)
            video_k = (
                self.compact_wan.apply_multiscale_rope(video_k, grid_sizes, freqs)
                if isinstance(grid_sizes, dict)
                else rope_apply(video_k, grid_sizes, freqs)
            )
        return video_k.detach(), video_v.detach()

    def _empty_video_cache(
        self,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor | Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> Dict[str, object]:
        return {
            "seq_lens": seq_lens.detach(),
            "grid_sizes": (
                {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in grid_sizes.items()
                }
                if isinstance(grid_sizes, dict)
                else grid_sizes.detach()
            ),
            "freqs": freqs.detach(),
            "video_k": [],
            "video_v": [],
        }

    def _append_video_cache_layer(
        self,
        cache: Dict[str, object],
        video_tokens: torch.Tensor,
        video_modulation: tuple[torch.Tensor, ...],
        layer_idx: int,
        grid_sizes: torch.Tensor | Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> None:
        video_k, video_v = self._video_attention_kv_for_cache(
            video_tokens,
            video_modulation,
            layer_idx,
            grid_sizes,
            freqs,
        )
        cache["video_k"].append(video_k)
        cache["video_v"].append(video_v)

    def configure_teacache(
        self,
        enabled: bool,
        delta: float = 0.0,
        num_steps: Optional[int] = None,
        poly_coefficients: Optional[Sequence[float]] = None,
        sync_distributed: bool = False,
        force_last_step: bool = False,
        action_enabled: bool = True,
        action_delta: Optional[float] = None,
        action_force_last_step: bool = False,
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
        self.enable_action_teacache = bool(enabled and action_enabled)
        self.action_teacache_delta = float(self.teacache_delta if action_delta is None else action_delta)
        if self.action_teacache_delta < 0:
            raise ValueError("action_teacache_delta must be non-negative")
        self.action_teacache_force_last_step = bool(action_force_last_step)
        self.reset_teacache(num_steps=num_steps)

    def reset_teacache(self, num_steps: Optional[int] = None) -> None:
        if num_steps is not None:
            self.teacache_num_steps = int(num_steps)
        self.teacache_accumulated_diff = 0.0
        self.teacache_previous_indicator = None
        self.teacache_cached_video_residual = None
        self.teacache_cached_action_residual = None
        self.teacache_cached_video_cache = None
        self.teacache_step_idx = 0
        self.teacache_num_cache_hits = 0
        self.teacache_num_cache_misses = 0
        self.reset_action_teacache(reset_stats=True)

    def reset_action_teacache(self, reset_stats: bool = False) -> None:
        self.action_teacache_accumulated_diff = 0.0
        self.action_teacache_previous_indicator = None
        self.action_teacache_cached_residual = None
        if reset_stats:
            self.action_teacache_step_idx = 0
            self.action_teacache_num_cache_hits = 0
            self.action_teacache_num_cache_misses = 0

    def teacache_stats(self) -> Dict[str, float]:
        joint_total = self.teacache_num_cache_hits + self.teacache_num_cache_misses
        action_total = self.action_teacache_num_cache_hits + self.action_teacache_num_cache_misses
        return {
            "enabled": float(self.enable_teacache),
            "delta": float(self.teacache_delta),
            "joint_steps": float(self.teacache_step_idx),
            "joint_cache_hits": float(self.teacache_num_cache_hits),
            "joint_cache_misses": float(self.teacache_num_cache_misses),
            "joint_hit_rate": float(self.teacache_num_cache_hits / joint_total) if joint_total else 0.0,
            "action_enabled": float(self.enable_action_teacache),
            "action_delta": float(self.action_teacache_delta),
            "action_steps": float(self.action_teacache_step_idx),
            "action_cache_hits": float(self.action_teacache_num_cache_hits),
            "action_cache_misses": float(self.action_teacache_num_cache_misses),
            "action_hit_rate": (
                float(self.action_teacache_num_cache_hits / action_total) if action_total else 0.0
            ),
        }

    def _teacache_active(self) -> bool:
        return bool(self.enable_teacache and not self.training and self.config.ae_num_layers > 0)

    def _action_teacache_active(self) -> bool:
        return bool(self.enable_action_teacache and not self.training and self.config.ae_num_layers > 0)

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
        require_video_cache: bool,
    ) -> bool:
        has_residuals = (
            self.teacache_cached_video_residual is not None
            and self.teacache_cached_action_residual is not None
            and self.teacache_cached_video_residual.shape == video_tokens.shape
            and self.teacache_cached_action_residual.shape == action_tokens.shape
        )
        if require_video_cache:
            return has_residuals and self.teacache_cached_video_cache is not None
        return has_residuals

    def _teacache_should_calc(
        self,
        indicator: torch.Tensor,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        require_video_cache: bool,
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
            and self._teacache_has_residuals(video_tokens, action_tokens, require_video_cache)
        ):
            self.teacache_num_cache_hits += 1
            return False

        self.teacache_accumulated_diff = 0.0
        self.teacache_num_cache_misses += 1
        return True

    def _teacache_copy_cached_video_cache(self) -> Optional[Dict[str, object]]:
        cached = self.teacache_cached_video_cache
        if cached is None:
            return None
        return {
            "seq_lens": cached["seq_lens"],
            "grid_sizes": (
                dict(cached["grid_sizes"])
                if isinstance(cached["grid_sizes"], dict)
                else cached["grid_sizes"]
            ),
            "freqs": cached["freqs"],
            "video_k": list(cached["video_k"]),
            "video_v": list(cached["video_v"]),
        }

    def _action_teacache_indicator(
        self,
        action_tokens: torch.Tensor,
        action_adaln_params: torch.Tensor,
    ) -> torch.Tensor:
        first_action_block = self.action_expert.blocks[0]
        first_modulation = self._block_modulation(first_action_block, action_adaln_params)
        return (
            first_action_block.norm1(action_tokens).float()
            * (1 + first_modulation[1].squeeze(2))
            + first_modulation[0].squeeze(2)
        )

    def _action_teacache_has_residual(self, action_tokens: torch.Tensor) -> bool:
        return (
            self.action_teacache_cached_residual is not None
            and self.action_teacache_cached_residual.shape == action_tokens.shape
        )

    def _action_teacache_should_calc(
        self,
        indicator: torch.Tensor,
        action_tokens: torch.Tensor,
        is_last_action_step: bool,
    ) -> bool:
        previous_indicator = self.action_teacache_previous_indicator
        if (
            previous_indicator is None
            or previous_indicator.shape != indicator.shape
            or previous_indicator.device != indicator.device
        ):
            self.action_teacache_accumulated_diff = 0.0
            self.action_teacache_previous_indicator = indicator.detach()
            self.action_teacache_num_cache_misses += 1
            return True

        if self.action_teacache_force_last_step and is_last_action_step:
            self.action_teacache_accumulated_diff = 0.0
            self.action_teacache_previous_indicator = indicator.detach()
            self.action_teacache_num_cache_misses += 1
            return True

        rel_l1 = self._teacache_relative_l1(indicator, previous_indicator)
        estimated_diff = self._teacache_estimated_diff(rel_l1)
        self.action_teacache_accumulated_diff += estimated_diff
        self.action_teacache_previous_indicator = indicator.detach()

        if (
            self.action_teacache_accumulated_diff <= self.action_teacache_delta
            and self._action_teacache_has_residual(action_tokens)
        ):
            self.action_teacache_num_cache_hits += 1
            return False

        self.action_teacache_accumulated_diff = 0.0
        self.action_teacache_num_cache_misses += 1
        return True

    def _action_attention_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_modulation: tuple[torch.Tensor, ...],
        layer_idx: int,
        video_cache: Dict[str, object],
    ) -> torch.Tensor:
        wan_layer = self.compact_wan.video_model.wan_model.blocks[layer_idx]
        action_block = self.action_expert.blocks[layer_idx]
        action_mod = action_modulation

        norm_action = (
            action_block.norm1(action_tokens).float() * (1 + action_mod[1].squeeze(2))
            + action_mod[0].squeeze(2)
        )
        action_q, action_k, action_v = self._action_qkv_for_wan(action_block, norm_action)

        video_k = video_cache["video_k"][layer_idx].to(device=action_q.device, dtype=action_q.dtype)
        video_v = video_cache["video_v"][layer_idx].to(device=action_v.device, dtype=action_v.dtype)
        joint_k = torch.cat([video_k, action_k], dim=1)
        joint_v = torch.cat([video_v, action_v], dim=1)
        video_seq_lens = video_cache["seq_lens"].to(device=action_q.device)
        joint_seq_lens = video_seq_lens + action_tokens.shape[1]

        attention_dtype = self.compact_wan.video_model.precision
        with torch.autocast("cuda", dtype=attention_dtype, enabled=action_q.is_cuda):
            action_out_h = flash_attention(
                q=action_q.to(dtype=attention_dtype),
                k=joint_k.to(dtype=attention_dtype),
                v=joint_v.to(dtype=attention_dtype),
                k_lens=joint_seq_lens,
                window_size=wan_layer.self_attn.window_size,
            )

        action_out_h = action_out_h.flatten(2).to(
            device=action_block.wan_action_o.weight.device,
            dtype=action_block.wan_action_o.weight.dtype,
        )
        action_out = action_block.wan_action_o(action_out_h)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=action_tokens.is_cuda):
            return action_tokens + action_out * action_mod[2].squeeze(2)

    def _build_action_tokens(
        self,
        noisy_actions: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(noisy_actions.shape[0], -1, -1)
        else:
            registers = None
        state_tokens = initial_state.unsqueeze(1)
        return self.action_expert.input_encoder(state_tokens, noisy_actions, registers)

    def forward_action_with_video_cache(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        video_cache = batch["video_cache"]
        noisy_actions = batch["noisy_actions"]
        action_t = batch["action_t"]
        initial_state = batch["initial_state"]
        is_last_action_step = bool(batch.get("is_last_action_step", False))

        action_tokens = self._build_action_tokens(noisy_actions, initial_state)
        action_head_time_emb, action_adaln_params = self._build_action_time_embeddings(
            action_t,
            action_tokens.shape[1],
        )

        action_teacache_active = self._action_teacache_active()
        should_calc = True
        if action_teacache_active:
            indicator = self._action_teacache_indicator(action_tokens, action_adaln_params)
            should_calc = self._action_teacache_should_calc(
                indicator,
                action_tokens,
                is_last_action_step=is_last_action_step,
            )

        if should_calc:
            action_origin = action_tokens
            for layer_idx in range(self.config.ae_num_layers):
                action_block = self.action_expert.blocks[layer_idx]
                action_modulation = self._block_modulation(action_block, action_adaln_params)
                action_tokens = self._action_attention_with_video_cache(
                    action_tokens,
                    action_modulation,
                    layer_idx,
                    video_cache,
                )

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

            if action_teacache_active:
                self.action_teacache_cached_residual = (action_tokens - action_origin).detach()
        else:
            action_residual = self.action_teacache_cached_residual
            assert action_residual is not None
            action_tokens = action_tokens + action_residual.to(device=action_tokens.device, dtype=action_tokens.dtype)

        if action_teacache_active:
            self.action_teacache_step_idx += 1

        action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
        trim_len = action_pred_full.shape[1] - self.action_expert.config.num_registers
        action_pred = action_pred_full[:, 1:trim_len, :]
        return {"action_pred": action_pred}

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        text_embeddings = batch["text_embeddings"]
        noisy_actions = batch["noisy_actions"]
        action_t = batch["action_t"]
        initial_state = batch["initial_state"]
        video_t = batch["video_t"]
        return_video_cache = bool(batch.get("return_video_cache", False))

        vgm_video_reference = batch["future_latent"] if "future_latent" in batch else batch["video_latent"]
        vgm_video_event = self._start_latency_event("vgm_video", vgm_video_reference)
        vgm_prepare_event = self._start_latency_event("vgm_prepare", vgm_video_reference)
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
        self._finish_latency_event(vgm_prepare_event, count_call=True)
        self._finish_latency_event(vgm_video_event)

        action_prepare_event = self._start_latency_event("action_prepare", noisy_actions)
        action_tokens = self._build_action_tokens(noisy_actions, initial_state)
        action_head_time_emb, action_adaln_params = self._build_action_time_embeddings(
            action_t,
            action_tokens.shape[1],
        )
        self._finish_latency_event(action_prepare_event, count_call=True)

        teacache_active = self._teacache_active()
        should_calc = True
        if teacache_active:
            indicator = self._teacache_indicator(video_tokens, video_adaln_params)
            should_calc = self._teacache_should_calc(
                indicator,
                video_tokens,
                action_tokens,
                require_video_cache=return_video_cache,
            )

        video_cache = None
        if should_calc:
            video_cache = self._empty_video_cache(seq_lens, grid_sizes, freqs) if return_video_cache else None
            video_origin = video_tokens
            action_origin = action_tokens
            for layer_idx in range(self.config.ae_num_layers):
                wan_layer = self.compact_wan.video_model.wan_model.blocks[layer_idx]
                action_block = self.action_expert.blocks[layer_idx]
                action_modulation = self._block_modulation(action_block, action_adaln_params)
                vgm_video_event = self._start_latency_event("vgm_video", video_tokens)
                vgm_layer_setup_event = self._start_latency_event("vgm_layer_setup", video_tokens)
                video_modulation = self._block_modulation(wan_layer, video_adaln_params)
                if video_cache is not None:
                    self._append_video_cache_layer(
                        video_cache,
                        video_tokens,
                        video_modulation,
                        layer_idx,
                        grid_sizes,
                        freqs,
                    )
                self._finish_latency_event(vgm_layer_setup_event, count_call=True)

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

                vgm_cross_attn_event = self._start_latency_event("vgm_cross_attn", video_tokens)
                cross_dtype = self.compact_wan.video_model.precision
                with torch.autocast("cuda", dtype=cross_dtype, enabled=video_tokens.is_cuda):
                    cross_in = wan_layer.norm3(video_tokens)
                    cross_out = wan_layer.cross_attn(cross_in, text_context, None)
                video_tokens = video_tokens + cross_out
                self._finish_latency_event(vgm_cross_attn_event, count_call=True)

                vgm_ffn_event = self._start_latency_event("vgm_ffn", video_tokens)
                video_ffn_in = (
                    wan_layer.norm2(video_tokens).float() * (1 + video_modulation[4].squeeze(2))
                    + video_modulation[3].squeeze(2)
                )
                video_ffn_weight = wan_layer.ffn[0].weight
                video_ffn = wan_layer.ffn(video_ffn_in.to(device=video_ffn_weight.device, dtype=video_ffn_weight.dtype))
                with torch.amp.autocast("cuda", dtype=torch.float32, enabled=video_tokens.is_cuda):
                    video_tokens = video_tokens + video_ffn * video_modulation[5].squeeze(2)
                self._finish_latency_event(vgm_ffn_event, count_call=True)
                self._finish_latency_event(vgm_video_event)

                action_ffn_event = self._start_latency_event("action_ffn", action_tokens)
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
                self._finish_latency_event(action_ffn_event, count_call=True)

            if teacache_active:
                self.teacache_cached_video_residual = (video_tokens - video_origin).detach()
                self.teacache_cached_action_residual = (action_tokens - action_origin).detach()
                self.teacache_cached_video_cache = video_cache
        else:
            video_residual = self.teacache_cached_video_residual
            action_residual = self.teacache_cached_action_residual
            assert video_residual is not None and action_residual is not None
            vgm_video_event = self._start_latency_event("vgm_video", video_tokens)
            video_tokens = video_tokens + video_residual.to(device=video_tokens.device, dtype=video_tokens.dtype)
            video_cache = self._teacache_copy_cached_video_cache() if return_video_cache else None
            self._finish_latency_event(vgm_video_event)
            action_tokens = action_tokens + action_residual.to(device=action_tokens.device, dtype=action_tokens.dtype)

        if teacache_active:
            self.teacache_step_idx += 1

        vgm_video_event = self._start_latency_event("vgm_video", video_tokens)
        vgm_head_event = self._start_latency_event("vgm_head", video_tokens)
        if isinstance(grid_sizes, dict):
            video_pred = self.compact_wan.apply_multiscale_video_head(video_tokens, video_head_time_emb, grid_sizes)
        else:
            video_pred = self.compact_wan.apply_video_head(video_tokens, video_head_time_emb, grid_sizes)
        self._finish_latency_event(vgm_head_event, count_call=True)
        self._finish_latency_event(vgm_video_event)

        action_head_event = self._start_latency_event("action_head", action_tokens)
        action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
        trim_len = action_pred_full.shape[1] - self.action_expert.config.num_registers
        action_pred = action_pred_full[:, 1:trim_len, :]
        self._finish_latency_event(action_head_event, count_call=True)
        return {
            "video_pred": video_pred,
            "action_pred": action_pred,
            **({"video_cache": video_cache} if video_cache is not None else {}),
        }
