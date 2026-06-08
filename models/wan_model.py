# WAN Video Diffusion Model
# Provides VAE encoding/decoding and feature extraction for diffusion-pipe I2V training

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Sequence
import logging
import sys
import os
import json
from pathlib import Path

from third_party.wan.modules.attention import flash_attention
from third_party.wan.modules.model import WanModel, rope_apply, sinusoidal_embedding_1d
from third_party.wan.modules.vae2_2 import Wan2_2_VAE

# Optional safetensors support
try:
    from safetensors.torch import load_file as safe_load_file  # type: ignore
except Exception:  # pragma: no cover
    safe_load_file = None

logger = logging.getLogger(__name__)


_WAN_WEIGHT_FILENAMES = (
    "diffusion_pytorch_model.safetensors",
    "model.safetensors",
    "pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    "model.bin",
    "model.pt",
    "wan_model.pt",
    "dit.pt",
)
_WAN_SHARD_INDEX_FILENAMES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
    "pytorch_model.safetensors.index.json",
    "diffusion_pytorch_model.bin.index.json",
    "pytorch_model.bin.index.json",
)
_WAN_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt"}
_NON_DIT_WEIGHT_NAME_PARTS = (
    "vae",
    "t5",
    "umt5",
    "text_encoder",
    "tokenizer",
    "optimizer",
    "scheduler",
)


def _load_wan_arch_config(config_path: str) -> Dict[str, Any]:
    config_json_path = os.path.join(config_path, "config.json")
    if not os.path.exists(config_json_path):
        raise FileNotFoundError(f"WAN config.json not found at {config_json_path}")
    with open(config_json_path, "r") as f:
        return json.load(f)


def _extract_model_state_dict(loaded: Any) -> Dict[str, torch.Tensor]:
    if isinstance(loaded, dict) and ("state_dict" in loaded or "model" in loaded):
        loaded = loaded.get("state_dict", loaded.get("model"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected WAN checkpoint to contain a state dict, got {type(loaded).__name__}")
    return loaded


def _load_wan_state_dict_file(path: Path) -> Dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        if safe_load_file is None:
            raise RuntimeError("safetensors not available. Please 'pip install safetensors'.")
        return safe_load_file(str(path), device="cpu")
    if suffix in {".bin", ".pt"}:
        return _extract_model_state_dict(torch.load(str(path), map_location="cpu"))
    raise ValueError(f"Unsupported WAN checkpoint file type: {path}")


def _load_wan_sharded_state_dict(index_path: Path) -> Dict[str, torch.Tensor]:
    with index_path.open("r") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Shard index missing non-empty weight_map: {index_path}")

    state_dict: Dict[str, torch.Tensor] = {}
    shard_names = sorted(set(str(name) for name in weight_map.values()))
    for shard_name in shard_names:
        shard_path = index_path.parent / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"WAN checkpoint shard listed in {index_path} not found: {shard_path}")
        state_dict.update(_load_wan_state_dict_file(shard_path))
    return state_dict


def _is_probable_wan_weight_file(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in _WAN_WEIGHT_SUFFIXES and not any(
        part in name for part in _NON_DIT_WEIGHT_NAME_PARTS
    )


def _find_wan_weight_file(checkpoint_dir: Path) -> Optional[Path]:
    for filename in _WAN_WEIGHT_FILENAMES:
        path = checkpoint_dir / filename
        if path.exists():
            return path

    candidates = [
        path
        for path in checkpoint_dir.iterdir()
        if path.is_file() and _is_probable_wan_weight_file(path)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _find_wan_shard_index(checkpoint_dir: Path) -> Optional[Path]:
    for filename in _WAN_SHARD_INDEX_FILENAMES:
        path = checkpoint_dir / filename
        if path.exists():
            return path
    return None


def _find_wan_shard_files(checkpoint_dir: Path) -> List[Path]:
    for prefix in ("diffusion_pytorch_model", "pytorch_model", "model"):
        for suffix in (".safetensors", ".bin"):
            shards = sorted(checkpoint_dir.glob(f"{prefix}-*-of-*{suffix}"))
            if shards:
                return shards
    return []


def _select_evenly_spaced_indices(total: int, keep: int) -> List[int]:
    if keep > total:
        raise ValueError(f"Cannot keep {keep} indices from total {total}")
    if keep == total:
        return list(range(total))
    steps = torch.linspace(0, total - 1, steps=keep)
    indices = torch.round(steps).to(torch.long).tolist()
    deduped: List[int] = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    cursor = 0
    while len(deduped) < keep:
        if cursor not in seen:
            deduped.append(cursor)
            seen.add(cursor)
        cursor += 1
    return sorted(deduped)


def _select_head_group_indices(total_heads: int, keep_heads: int, head_dim: int) -> List[int]:
    head_indices = _select_evenly_spaced_indices(total_heads, keep_heads)
    hidden_indices: List[int] = []
    for head_idx in head_indices:
        start = head_idx * head_dim
        hidden_indices.extend(range(start, start + head_dim))
    return hidden_indices


def _select_ffn_indices(total_ffn_dim: int, keep_ffn_dim: int) -> List[int]:
    return _select_evenly_spaced_indices(total_ffn_dim, keep_ffn_dim)


def _slice_vector(tensor: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    idx = torch.as_tensor(indices, dtype=torch.long)
    return tensor.index_select(0, idx).clone()


def _slice_linear(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    row_indices: Optional[Sequence[int]] = None,
    col_indices: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    out = weight
    if row_indices is not None:
        out = out.index_select(0, torch.as_tensor(row_indices, dtype=torch.long))
    if col_indices is not None:
        out = out.index_select(1, torch.as_tensor(col_indices, dtype=torch.long))
    result = {"weight": out.clone()}
    if bias is not None:
        if row_indices is not None:
            result["bias"] = bias.index_select(0, torch.as_tensor(row_indices, dtype=torch.long)).clone()
        else:
            result["bias"] = bias.clone()
    return result


def _slice_conv3d_out_channels(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_indices: Sequence[int],
) -> Dict[str, torch.Tensor]:
    idx = torch.as_tensor(out_indices, dtype=torch.long)
    result = {"weight": weight.index_select(0, idx).clone()}
    if bias is not None:
        result["bias"] = bias.index_select(0, idx).clone()
    return result


def _build_time_projection_row_indices(hidden_indices: Sequence[int], teacher_dim: int, groups: int = 6) -> List[int]:
    row_indices: List[int] = []
    for group_idx in range(groups):
        offset = group_idx * teacher_dim
        row_indices.extend([offset + idx for idx in hidden_indices])
    return row_indices


def _copy_non_block_state(
    teacher_state: Dict[str, torch.Tensor],
    student_state: Dict[str, torch.Tensor],
    hidden_indices: Sequence[int],
    ffn_indices: Sequence[int],
    teacher_dim: int,
) -> None:
    patch = _slice_conv3d_out_channels(
        teacher_state["patch_embedding.weight"],
        teacher_state.get("patch_embedding.bias"),
        hidden_indices,
    )
    student_state["patch_embedding.weight"] = patch["weight"]
    if "bias" in patch:
        student_state["patch_embedding.bias"] = patch["bias"]

    text0 = _slice_linear(
        teacher_state["text_embedding.0.weight"],
        teacher_state.get("text_embedding.0.bias"),
        row_indices=hidden_indices,
    )
    student_state["text_embedding.0.weight"] = text0["weight"]
    student_state["text_embedding.0.bias"] = text0["bias"]
    text2 = _slice_linear(
        teacher_state["text_embedding.2.weight"],
        teacher_state.get("text_embedding.2.bias"),
        row_indices=hidden_indices,
        col_indices=hidden_indices,
    )
    student_state["text_embedding.2.weight"] = text2["weight"]
    student_state["text_embedding.2.bias"] = text2["bias"]

    time0 = _slice_linear(
        teacher_state["time_embedding.0.weight"],
        teacher_state.get("time_embedding.0.bias"),
        row_indices=hidden_indices,
    )
    student_state["time_embedding.0.weight"] = time0["weight"]
    student_state["time_embedding.0.bias"] = time0["bias"]
    time2 = _slice_linear(
        teacher_state["time_embedding.2.weight"],
        teacher_state.get("time_embedding.2.bias"),
        row_indices=hidden_indices,
        col_indices=hidden_indices,
    )
    student_state["time_embedding.2.weight"] = time2["weight"]
    student_state["time_embedding.2.bias"] = time2["bias"]

    row_indices = _build_time_projection_row_indices(hidden_indices, teacher_dim, groups=6)
    time_proj = _slice_linear(
        teacher_state["time_projection.1.weight"],
        teacher_state.get("time_projection.1.bias"),
        row_indices=row_indices,
        col_indices=hidden_indices,
    )
    student_state["time_projection.1.weight"] = time_proj["weight"]
    student_state["time_projection.1.bias"] = time_proj["bias"]

    head = _slice_linear(
        teacher_state["head.head.weight"],
        teacher_state.get("head.head.bias"),
        col_indices=hidden_indices,
    )
    student_state["head.head.weight"] = head["weight"]
    student_state["head.head.bias"] = head["bias"]
    student_state["head.modulation"] = teacher_state["head.modulation"][:, :, hidden_indices].clone()


def _copy_block_state(
    teacher_state: Dict[str, torch.Tensor],
    student_state: Dict[str, torch.Tensor],
    teacher_idx: int,
    student_idx: int,
    hidden_indices: Sequence[int],
    ffn_indices: Sequence[int],
) -> None:
    teacher_prefix = f"blocks.{teacher_idx}."
    student_prefix = f"blocks.{student_idx}."

    def t(name: str) -> str:
        return teacher_prefix + name

    def s(name: str) -> str:
        return student_prefix + name

    # Norms
    for norm_name in [
        "self_attn.norm_q.weight",
        "self_attn.norm_k.weight",
        "cross_attn.norm_q.weight",
        "cross_attn.norm_k.weight",
        "norm3.weight",
        "norm3.bias",
    ]:
        if t(norm_name) in teacher_state:
            student_state[s(norm_name)] = _slice_vector(teacher_state[t(norm_name)], hidden_indices)

    # Attention projections
    for attn_name in [
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
    ]:
        sliced = _slice_linear(
            teacher_state[t(f"{attn_name}.weight")],
            teacher_state.get(t(f"{attn_name}.bias")),
            row_indices=hidden_indices,
            col_indices=hidden_indices,
        )
        student_state[s(f"{attn_name}.weight")] = sliced["weight"]
        if "bias" in sliced:
            student_state[s(f"{attn_name}.bias")] = sliced["bias"]

    # FFN
    ffn0 = _slice_linear(
        teacher_state[t("ffn.0.weight")],
        teacher_state.get(t("ffn.0.bias")),
        row_indices=ffn_indices,
        col_indices=hidden_indices,
    )
    student_state[s("ffn.0.weight")] = ffn0["weight"]
    student_state[s("ffn.0.bias")] = ffn0["bias"]
    ffn2 = _slice_linear(
        teacher_state[t("ffn.2.weight")],
        teacher_state.get(t("ffn.2.bias")),
        row_indices=hidden_indices,
        col_indices=ffn_indices,
    )
    student_state[s("ffn.2.weight")] = ffn2["weight"]
    student_state[s("ffn.2.bias")] = ffn2["bias"]

    # Modulation
    student_state[s("modulation")] = teacher_state[t("modulation")][:, :, hidden_indices].clone()


def _build_structured_sliced_wan_state_dict(
    teacher_state: Dict[str, torch.Tensor],
    teacher_model_config: Dict[str, Any],
    student_model_config: Dict[str, Any],
    teacher_layer_mapping: Sequence[int],
) -> Dict[str, torch.Tensor]:
    teacher_dim = int(teacher_model_config["dim"])
    teacher_heads = int(teacher_model_config["num_heads"])
    teacher_head_dim = teacher_dim // teacher_heads
    teacher_ffn_dim = int(teacher_model_config["ffn_dim"])

    student_dim = int(student_model_config["dim"])
    student_heads = int(student_model_config["num_heads"])
    student_ffn_dim = int(student_model_config["ffn_dim"])
    student_layers = int(student_model_config["num_layers"])

    if len(teacher_layer_mapping) != student_layers:
        raise ValueError(
            f"Layer mapping length {len(teacher_layer_mapping)} must match student num_layers {student_layers}"
        )

    hidden_indices = _select_head_group_indices(teacher_heads, student_heads, teacher_head_dim)
    ffn_indices = _select_ffn_indices(teacher_ffn_dim, student_ffn_dim)

    student_state: Dict[str, torch.Tensor] = {}
    _copy_non_block_state(
        teacher_state=teacher_state,
        student_state=student_state,
        hidden_indices=hidden_indices,
        ffn_indices=ffn_indices,
        teacher_dim=teacher_dim,
    )

    zero_based_mapping = [layer - 1 for layer in teacher_layer_mapping]
    for student_idx, teacher_idx in enumerate(zero_based_mapping):
        _copy_block_state(
            teacher_state=teacher_state,
            student_state=student_state,
            teacher_idx=teacher_idx,
            student_idx=student_idx,
            hidden_indices=hidden_indices,
            ffn_indices=ffn_indices,
        )

    return student_state

def _strip_known_prefixes_for_wan(sd: Dict[str, torch.Tensor], target_model: nn.Module) -> Dict[str, torch.Tensor]:
    """Strip only the 'dit.' prefix from checkpoint keys if present."""
    if not isinstance(sd, dict):
        return sd
    if not any(k.startswith('dit.') for k in sd.keys()):
        return sd
    mapped = { (k[4:] if k.startswith('dit.') else k): v for k, v in sd.items() }
    logger.info("Stripped 'dit.' prefix from checkpoint keys")
    return mapped


def _load_wan_state_dict_from_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load a raw WAN state dict from a supported checkpoint path."""
    path = Path(checkpoint_path)
    if path.is_dir():
        shard_index = _find_wan_shard_index(path)
        if shard_index is not None:
            logger.info("Loading WAN weights directly from shard index %s", shard_index)
            wan_state_dict = _load_wan_sharded_state_dict(shard_index)
        else:
            shard_files = _find_wan_shard_files(path)
            if shard_files:
                logger.info("Loading WAN weights directly from %d shard files in %s", len(shard_files), path)
                wan_state_dict = {}
                for shard_file in shard_files:
                    wan_state_dict.update(_load_wan_state_dict_file(shard_file))
            else:
                weight_file = _find_wan_weight_file(path)
                if weight_file is not None:
                    logger.info("Loading WAN weights directly from %s", weight_file)
                    wan_state_dict = _load_wan_state_dict_file(weight_file)
                else:
                    logger.info(
                        "No direct WAN weight file found in %s; falling back to WanModel.from_pretrained",
                        checkpoint_path,
                    )
                    loaded_model = WanModel.from_pretrained(checkpoint_path)
                    wan_state_dict = loaded_model.state_dict()
    elif path.suffix.lower() in _WAN_WEIGHT_SUFFIXES:
        wan_state_dict = _load_wan_state_dict_file(path)
    else:
        loaded_model = WanModel.from_pretrained(checkpoint_path)
        wan_state_dict = loaded_model.state_dict()

    try:
        wan_state_dict = _strip_known_prefixes_for_wan(wan_state_dict, None)
    except Exception:
        pass
    return wan_state_dict


def _build_pruned_wan_state_dict(
    full_state_dict: Dict[str, torch.Tensor],
    keep_layer_indices: List[int]
) -> Dict[str, torch.Tensor]:
    """Map teacher WAN block weights onto a shallower student WAN."""
    keep_layer_indices = list(keep_layer_indices)
    block_prefix = "blocks."
    pruned_state_dict: Dict[str, torch.Tensor] = {}

    # Copy non-block weights verbatim.
    for key, value in full_state_dict.items():
        if not key.startswith(block_prefix):
            pruned_state_dict[key] = value

    # Remap surviving blocks to contiguous student indices.
    for student_idx, teacher_idx in enumerate(keep_layer_indices):
        teacher_prefix = f"{block_prefix}{teacher_idx}."
        student_prefix = f"{block_prefix}{student_idx}."
        found_any = False
        for key, value in full_state_dict.items():
            if key.startswith(teacher_prefix):
                found_any = True
                pruned_state_dict[student_prefix + key[len(teacher_prefix):]] = value
        if not found_any:
            raise KeyError(f"Teacher WAN block {teacher_idx} not found in checkpoint state dict")

    return pruned_state_dict

class WanVideoModel(nn.Module):
    """
    WAN Video Diffusion Model wrapper for TI2V Teacher Forcing training.
    Provides VAE encoding/decoding and feature extraction for joint video-action training.
    Uses Teacher Forcing approach for I2V conditioning (DiffSynth-Studio style).
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        vae_path: Optional[str],
        device: str = "cuda",
        precision: str = "bfloat16",
        load_vae: bool = True,
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[precision]
        
        # Initialize WAN model
        self.wan_model = WanModel(**model_config)
        self.wan_model.to(device=self.device, dtype=self.precision)

        # Initialize VAE only for pixel<->latent paths. Latent-only teacher
        # forward passes do not need it and can skip a large checkpoint load.
        self.vae = None
        if load_vae:
            if vae_path is None:
                raise ValueError("vae_path is required when load_vae=True")
            self.vae = Wan2_2_VAE(vae_pth=vae_path, device=self.device)
        else:
            logger.info("Skipping WAN VAE load for latent-only model usage")
        
        logger.info(f"WAN Video Model initialized with {sum(p.numel() for p in self.wan_model.parameters()):,} parameters")
    
    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode video pixels to latent space.
        
        Args:
            video_pixels: Video in pixel space [B, C, T, H, W], range [-1, 1]
            
        Returns:
            Video latents [B, C', T', H', W']
        """
        with torch.no_grad():
            if self.vae is None:
                raise RuntimeError("WanVideoModel was initialized with load_vae=False; encode_video is unavailable")
            return self.vae.encode(video_pixels)
    
    def decode_video(self, video_latents: torch.Tensor) -> torch.Tensor:
        """
        Decode video latents to pixel space.
        
        Args:
            video_latents: Video latents [B, C, T, H, W]
            
        Returns:
            Video pixels [B, C', T', H', W'], range [-1, 1]
        """
        with torch.no_grad():
            if self.vae is None:
                raise RuntimeError("WanVideoModel was initialized with load_vae=False; decode_video is unavailable")
            video_pixels = []
            for i in range(video_latents.shape[0]):
                pixels = self.vae.decode([video_latents[i]])[0]
                video_pixels.append(pixels)
            result = torch.stack(video_pixels, dim=0)
            return result

    def prepare_video_tokens(
        self,
        video_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patchify WAN latents into token form for layer-wise MoT execution."""
        if video_latent.ndim != 5:
            raise ValueError(
                f"Expected 5D tensor [B, C, f, h, w], got {video_latent.ndim}D with shape {video_latent.shape}"
            )
        if video_latent.shape[1] != 48:
            raise ValueError(f"Expected 48 channels for WAN 2.2 latents, got {video_latent.shape[1]}")

        device = self.wan_model.patch_embedding.weight.device
        dtype = self.precision
        video_latent = video_latent.to(device=device, dtype=dtype)
        if self.wan_model.freqs.device != device:
            self.wan_model.freqs = self.wan_model.freqs.to(device)

        autocast_enabled = device.type == "cuda"
        with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            patched = self.wan_model.patch_embedding(video_latent)
            grid_sizes = torch.stack(
                [
                    torch.tensor(patched[idx].shape[1:], dtype=torch.long, device=device)
                    for idx in range(patched.shape[0])
                ]
            )
            tokens = patched.flatten(2).transpose(1, 2)
            seq_lens = torch.full(
                (tokens.shape[0],),
                tokens.shape[1],
                dtype=torch.long,
                device=device,
            )
        return tokens, seq_lens, grid_sizes, self.wan_model.freqs

    @staticmethod
    def _validate_wan_latent(name: str, latent: torch.Tensor) -> None:
        if latent.ndim != 5:
            raise ValueError(f"Expected {name} latent [B, C, f, h, w], got {latent.ndim}D {latent.shape}")
        if latent.shape[1] != 48:
            raise ValueError(f"Expected 48 channels for {name} WAN 2.2 latents, got {latent.shape[1]}")

    def _patch_video_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_wan_latent("video", latent)
        device = self.wan_model.patch_embedding.weight.device
        latent = latent.to(device=device, dtype=self.precision)
        with torch.autocast("cuda", dtype=self.precision, enabled=device.type == "cuda"):
            patched = self.wan_model.patch_embedding(latent)
            grid_sizes = torch.stack(
                [
                    torch.tensor(patched[idx].shape[1:], dtype=torch.long, device=device)
                    for idx in range(patched.shape[0])
                ]
            )
            tokens = patched.flatten(2).transpose(1, 2)
        return tokens, grid_sizes

    def prepare_multiscale_video_tokens(
        self,
        condition_latent: torch.Tensor,
        future_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor | int], torch.Tensor]:
        """Patchify high-resolution condition and low-resolution future latents."""
        self._validate_wan_latent("condition", condition_latent)
        self._validate_wan_latent("future", future_latent)
        if condition_latent.shape[0] != future_latent.shape[0]:
            raise ValueError(
                "Multiscale condition and future batch sizes differ: "
                f"{condition_latent.shape[0]} != {future_latent.shape[0]}"
            )
        device = self.wan_model.patch_embedding.weight.device
        if self.wan_model.freqs.device != device:
            self.wan_model.freqs = self.wan_model.freqs.to(device)
        condition_tokens, condition_grid_sizes = self._patch_video_latent(condition_latent)
        future_tokens, future_grid_sizes = self._patch_video_latent(future_latent)
        tokens = torch.cat([condition_tokens, future_tokens], dim=1)
        seq_lens = torch.full(
            (tokens.shape[0],),
            tokens.shape[1],
            dtype=torch.long,
            device=device,
        )
        layout: Dict[str, torch.Tensor | int] = {
            "condition_grid_sizes": condition_grid_sizes,
            "future_grid_sizes": future_grid_sizes,
            "condition_seq_len": int(condition_tokens.shape[1]),
            "future_seq_len": int(future_tokens.shape[1]),
        }
        patch_f, patch_h, patch_w = self.wan_model.patch_size
        layout["condition_grid_shape"] = (
            int(condition_latent.shape[2] // patch_f),
            int(condition_latent.shape[3] // patch_h),
            int(condition_latent.shape[4] // patch_w),
        )
        layout["future_grid_shape"] = (
            int(future_latent.shape[2] // patch_f),
            int(future_latent.shape[3] // patch_h),
            int(future_latent.shape[4] // patch_w),
        )
        return tokens, seq_lens, layout, self.wan_model.freqs

    @staticmethod
    def multiscale_layout_grid_sizes(
        layout: Dict[str, torch.Tensor | int],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        condition_grid_sizes = layout["condition_grid_sizes"]
        future_grid_sizes = layout["future_grid_sizes"]
        condition_seq_len = int(layout["condition_seq_len"])
        if not isinstance(condition_grid_sizes, torch.Tensor) or not isinstance(future_grid_sizes, torch.Tensor):
            raise TypeError("Multiscale video layout must carry tensor grid sizes")
        return condition_grid_sizes, future_grid_sizes, condition_seq_len

    def apply_multiscale_rope(
        self,
        heads: torch.Tensor,
        layout: Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        condition_grid_sizes, future_grid_sizes, condition_seq_len = self.multiscale_layout_grid_sizes(layout)
        condition_heads = rope_apply(heads[:, :condition_seq_len], condition_grid_sizes, freqs)
        future_heads = rope_apply(heads[:, condition_seq_len:], future_grid_sizes, freqs)
        return torch.cat([condition_heads, future_heads], dim=1)

    def apply_video_self_attention(
        self,
        self_attn: nn.Module,
        video_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor | Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(grid_sizes, dict):
            return self_attn(video_tokens, seq_lens, grid_sizes, freqs)

        batch, seq_len = video_tokens.shape[:2]
        heads = int(self_attn.num_heads)
        head_dim = int(self_attn.head_dim)
        q = self_attn.norm_q(self_attn.q(video_tokens)).view(batch, seq_len, heads, head_dim)
        k = self_attn.norm_k(self_attn.k(video_tokens)).view(batch, seq_len, heads, head_dim)
        v = self_attn.v(video_tokens).view(batch, seq_len, heads, head_dim)
        attended = flash_attention(
            q=self.apply_multiscale_rope(q, grid_sizes, freqs),
            k=self.apply_multiscale_rope(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self_attn.window_size,
        )
        return self_attn.o(attended.flatten(2))

    def prepare_text_context(self, text_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Pad/truncate T5 embeddings and project them into WAN hidden space."""
        device = self.wan_model.patch_embedding.weight.device
        dtype = self.precision
        padded_embeddings = []
        for emb in text_embeddings:
            emb = emb.to(device=device, dtype=dtype)
            if emb.size(0) > self.wan_model.text_len:
                emb = emb[: self.wan_model.text_len]
            elif emb.size(0) < self.wan_model.text_len:
                pad = emb.new_zeros(self.wan_model.text_len - emb.size(0), emb.size(1))
                emb = torch.cat([emb, pad], dim=0)
            padded_embeddings.append(emb)

        autocast_enabled = device.type == "cuda"
        with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            return self.wan_model.text_embedding(torch.stack(padded_embeddings, dim=0))

    def prepare_time_embeddings(
        self,
        timestep: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build WAN head and AdaLN time embeddings for an already-tokenized sequence."""
        device = self.wan_model.patch_embedding.weight.device
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(1).expand(timestep.size(0), seq_len)
        timestep = timestep.to(device=device)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=device.type == "cuda"):
            batch = timestep.size(0)
            flat_t = timestep.flatten()
            head_time_emb = self.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.wan_model.freq_dim, flat_t)
                .unflatten(0, (batch, seq_len))
                .float()
                .to(device)
            )
            adaln = self.wan_model.time_projection(head_time_emb).unflatten(2, (6, self.wan_model.dim))
        return head_time_emb, adaln

    def apply_video_head(
        self,
        video_tokens: torch.Tensor,
        video_time_emb: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Apply WAN output head and unpatchify tokens back to latent velocity."""
        head_weight = self.wan_model.head.head.weight
        video_tokens = video_tokens.to(device=head_weight.device, dtype=head_weight.dtype)
        with torch.autocast("cuda", dtype=self.precision, enabled=head_weight.device.type == "cuda"):
            output = self.wan_model.head(video_tokens, video_time_emb)
            output = self.wan_model.unpatchify(output, grid_sizes)
        return torch.stack([item.float() for item in output], dim=0)

    def apply_multiscale_video_head(
        self,
        video_tokens: torch.Tensor,
        video_time_emb: torch.Tensor,
        layout: Dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        """Unpatchify only future velocity tokens for a multiscale video."""
        _, future_grid_sizes, condition_seq_len = self.multiscale_layout_grid_sizes(layout)
        future_tokens = video_tokens[:, condition_seq_len:]
        future_time_emb = video_time_emb[:, condition_seq_len:]
        head_weight = self.wan_model.head.head.weight
        future_tokens = future_tokens.to(device=head_weight.device, dtype=head_weight.dtype)
        with torch.autocast("cuda", dtype=self.precision, enabled=head_weight.device.type == "cuda"):
            output = self.wan_model.head(future_tokens, future_time_emb)
            output = self.wan_model.unpatchify(output, future_grid_sizes)
        return torch.stack([item.float() for item in output], dim=0)

    def get_multiscale_layer_features(
        self,
        condition_latent: torch.Tensor,
        future_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        layer_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Extract WAN features from split condition/future latent streams."""
        if layer_indices is None:
            requested_layer_indices = list(range(len(self.wan_model.blocks)))
        else:
            requested_layer_indices = list(layer_indices)
            if requested_layer_indices and min(requested_layer_indices) >= 1:
                requested_layer_indices = [idx - 1 for idx in requested_layer_indices]

        x, seq_lens, layout, freqs = self.prepare_multiscale_video_tokens(
            condition_latent,
            future_latent,
        )
        e, e0 = self.prepare_time_embeddings(timestep, int(x.shape[1]))
        context = self.prepare_text_context(text_embeddings)
        device = self.wan_model.patch_embedding.weight.device
        layer_features = []
        with torch.autocast("cuda", dtype=self.precision, enabled=device.type == "cuda"):
            for i, block in enumerate(self.wan_model.blocks):
                with torch.amp.autocast("cuda", dtype=torch.float32, enabled=x.is_cuda):
                    modulation = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
                norm_video = block.norm1(x).float() * (1 + modulation[1].squeeze(2)) + modulation[0].squeeze(2)
                self_out = self.apply_video_self_attention(
                    block.self_attn,
                    norm_video,
                    seq_lens,
                    layout,
                    freqs,
                )
                with torch.amp.autocast("cuda", dtype=torch.float32, enabled=x.is_cuda):
                    x = x + self_out * modulation[2].squeeze(2)
                x = x + block.cross_attn(block.norm3(x), context, None)
                ffn_in = block.norm2(x).float() * (1 + modulation[4].squeeze(2)) + modulation[3].squeeze(2)
                ffn_weight = block.ffn[0].weight
                ffn_out = block.ffn(ffn_in.to(device=ffn_weight.device, dtype=ffn_weight.dtype))
                with torch.amp.autocast("cuda", dtype=torch.float32, enabled=x.is_cuda):
                    x = x + ffn_out * modulation[5].squeeze(2)
                if i in requested_layer_indices:
                    layer_features.append(x)
        layer_features.append(self.apply_multiscale_video_head(x, e, layout))
        return layer_features
    
    def get_layer_features(
        self,
        video_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        layer_indices: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Extract intermediate layer features for cross-attention injection.
        
        Args:
            video_latent: Video latent tensors [B, C, T, H, W]
            timestep: Diffusion timesteps [B]
            text_embeddings: List of text embeddings
            layer_indices: Which layers to extract (None = all layers)
            
        Returns:
            List of feature tensors from specified layers
        """
        if layer_indices is None:
            requested_layer_indices = list(range(len(self.wan_model.blocks)))
        else:
            requested_layer_indices = list(layer_indices)
            if requested_layer_indices and min(requested_layer_indices) >= 1:
                requested_layer_indices = [idx - 1 for idx in requested_layer_indices]
        
        # Expect 5D batch input: [B, C, f, h, w] - standard WAN input (48 channels)
        if video_latent.ndim != 5:
            raise ValueError(f"Expected 5D tensor [B, C, f, h, w], got {video_latent.ndim}D with shape {video_latent.shape}")
        
        # Ensure input has correct channel count for WAN 2.2 (48 channels)
        expected_channels = 48
        if video_latent.shape[1] != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels for WAN 2.2, got {video_latent.shape[1]} channels")
        
        x, seq_lens, grid_sizes, freqs = self.prepare_video_tokens(video_latent)
        seq_len = int(x.shape[1])
        e, e0 = self.prepare_time_embeddings(timestep, seq_len)
        context = self.prepare_text_context(text_embeddings)

        device = self.wan_model.patch_embedding.weight.device
        autocast_enabled = device.type == "cuda"
        with torch.autocast("cuda", dtype=self.precision, enabled=autocast_enabled):
            # Forward through specified layers
            layer_features = []
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                context_lens=None
            )

            for i, block in enumerate(self.wan_model.blocks):
                x = block(x, **kwargs)
                if i in requested_layer_indices:
                    layer_features.append(x)

            # Apply head and unpatchify to get final output (like forward method)
            x = self.wan_model.head(x, e)
            x = self.wan_model.unpatchify(x, grid_sizes)
        final_output = torch.stack([u.float() for u in x], dim=0)
        
        # Add final output as last element
        layer_features.append(final_output)
        
        return layer_features

    @classmethod
    def from_config(
        cls,
        config_path: str,
        vae_path: Optional[str],
        device: str = "cuda",
        precision: str = "bfloat16",
        load_vae: bool = True,
    ) -> 'WanVideoModel':
        """
        Initialize WAN model architecture, optionally with VAE, without WAN weights.
        Useful when model weights will be loaded from a higher-level checkpoint.
        """
        # Load WAN model config
        config_json_path = os.path.join(config_path, 'config.json')
        if not os.path.exists(config_json_path):
            raise FileNotFoundError(f"WAN config.json not found at {config_json_path}")
        with open(config_json_path, 'r') as f:
            model_config = json.load(f)
        # Create model without loading WAN weights
        model = cls(
            model_config=model_config,
            vae_path=vae_path,
            device=device,
            precision=precision,
            load_vae=load_vae,
        )
        logger.info("Initialized WAN model from config only (no WAN weights loaded)")
        return model

    @classmethod
    def from_compact_config(
        cls,
        config_path: str,
        vae_path: Optional[str],
        student_model_config: Dict[str, Any],
        device: str = "cuda",
        precision: str = "bfloat16",
        load_vae: bool = True,
    ) -> "WanVideoModel":
        """Initialize compact WAN architecture, optionally with VAE, without loading WAN weights."""
        teacher_model_config = _load_wan_arch_config(config_path)
        merged_student_config = dict(teacher_model_config)
        merged_student_config.update(student_model_config)
        merged_student_config["num_layers"] = int(student_model_config["num_layers"])

        model = cls(
            model_config=merged_student_config,
            vae_path=vae_path,
            device=device,
            precision=precision,
            load_vae=load_vae,
        )
        logger.info("Initialized compact WAN model from config only (no WAN weights loaded)")
        return model

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        vae_path: Optional[str],
        config_path: Optional[str] = None,
        device: str = "cuda",
        precision: str = "bfloat16",
        load_vae: bool = True,
    ) -> 'WanVideoModel':
        """
        Load pretrained WAN model.
        
        Args:
            checkpoint_path: Path to WAN checkpoint (.pt file or directory)
            vae_path: Path to VAE checkpoint
            config_path: Path to config directory (optional, defaults to checkpoint_path)
            device: Device to load model on
            precision: Model precision
            
        Returns:
            WanVideoModel instance
        """
        # Load WAN model config
        if config_path is None:
            config_path = checkpoint_path
        
        config_json_path = os.path.join(config_path, 'config.json')
        if os.path.exists(config_json_path):
            with open(config_json_path, 'r') as f:
                model_config = json.load(f)
        
        # Create model
        model = cls(
            model_config=model_config,
            vae_path=vae_path,
            device=device,
            precision=precision,
            load_vae=load_vae,
        )
        
        # Load WAN weights - support directory and file formats
        try:
            logger.info(f"Loading WAN weights from {checkpoint_path}")
            wan_state_dict = _load_wan_state_dict_from_checkpoint(checkpoint_path)
            model.wan_model.load_state_dict(wan_state_dict, strict=True)
            logger.info("Successfully loaded WAN weights from checkpoint")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load WAN checkpoint from {checkpoint_path}") from e
        
        return model

    @classmethod
    def from_pretrained_pruned(
        cls,
        checkpoint_path: str,
        vae_path: Optional[str],
        keep_layer_indices: List[int],
        config_path: Optional[str] = None,
        device: str = "cuda",
        precision: str = "bfloat16",
        load_vae: bool = True,
    ) -> 'WanVideoModel':
        """Load a pruned WAN by keeping a subset of teacher layers."""
        if config_path is None:
            config_path = checkpoint_path

        config_json_path = os.path.join(config_path, 'config.json')
        if not os.path.exists(config_json_path):
            raise FileNotFoundError(f"WAN config.json not found at {config_json_path}")
        with open(config_json_path, 'r') as f:
            model_config = json.load(f)

        keep_layer_indices = list(keep_layer_indices)
        model_config['num_layers'] = len(keep_layer_indices)

        model = cls(
            model_config=model_config,
            vae_path=vae_path,
            device=device,
            precision=precision,
            load_vae=load_vae,
        )

        try:
            logger.info(
                "Loading pruned WAN from %s with keep_layer_indices=%s",
                checkpoint_path,
                keep_layer_indices,
            )
            full_state_dict = _load_wan_state_dict_from_checkpoint(checkpoint_path)
            pruned_state_dict = _build_pruned_wan_state_dict(full_state_dict, keep_layer_indices)
            model.wan_model.load_state_dict(pruned_state_dict, strict=True)
            logger.info(
                "Successfully loaded pruned WAN weights with %d student layers from %d teacher layers",
                len(keep_layer_indices),
                len({int(k.split('.')[1]) for k in full_state_dict if k.startswith('blocks.')}),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pruned WAN checkpoint from {checkpoint_path} "
                f"with keep_layer_indices={keep_layer_indices}"
            ) from e

        return model

    @classmethod
    def from_pretrained_compact(
        cls,
        checkpoint_path: str,
        vae_path: Optional[str],
        student_model_config: Dict[str, Any],
        teacher_layer_mapping: Sequence[int],
        config_path: Optional[str] = None,
        device: str = "cuda",
        precision: str = "bfloat16",
        load_vae: bool = True,
    ) -> "WanVideoModel":
        """Load a compact WAN initialized by structured slicing from the teacher checkpoint."""
        if config_path is None:
            config_path = checkpoint_path

        teacher_model_config = _load_wan_arch_config(config_path)
        model = cls.from_compact_config(
            config_path=config_path,
            vae_path=vae_path,
            student_model_config=student_model_config,
            device=device,
            precision=precision,
            load_vae=load_vae,
        )
        merged_student_config = dict(teacher_model_config)
        merged_student_config.update(student_model_config)
        merged_student_config["num_layers"] = int(student_model_config["num_layers"])

        try:
            logger.info(
                "Loading compact WAN from %s with structured slicing and teacher_layer_mapping=%s",
                checkpoint_path,
                list(teacher_layer_mapping),
            )
            teacher_state = _load_wan_state_dict_from_checkpoint(checkpoint_path)
            compact_state = _build_structured_sliced_wan_state_dict(
                teacher_state=teacher_state,
                teacher_model_config=teacher_model_config,
                student_model_config=merged_student_config,
                teacher_layer_mapping=teacher_layer_mapping,
            )
            model.wan_model.load_state_dict(compact_state, strict=True)
            logger.info("Successfully loaded compact WAN via structured slicing")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load compact WAN checkpoint from {checkpoint_path} "
                f"with teacher_layer_mapping={list(teacher_layer_mapping)}"
            ) from e

        return model
