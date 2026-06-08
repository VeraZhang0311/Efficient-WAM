from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from .action_normalization import RoboTwinQposNormalizer, normalize_action_normalization_config
from .third_party.wan.modules.t5 import T5EncoderModel
from .models.compact_wan import CompactWANConfig, CompactWANModel
from .models.small_wam import SmallWAMActionConfig, SmallWAMActionModel


@dataclass
class EfficientWAMRuntime:
    model: SmallWAMActionModel
    t5_encoder: T5EncoderModel
    action_normalizer: RoboTwinQposNormalizer
    device: str
    num_inference_steps: int
    num_video_inference_steps: int
    video_refresh_steps: tuple[int, ...]
    video_stop_cosine_threshold: float | None
    action_skip_cosine_threshold: float | None
    chunk_size: int
    num_video_frames: int
    video_size: tuple[int, int]
    scene_prefix: str
    save_predicted_video: bool
    predicted_video_fps: int
    predicted_video_every_n_chunks: int
    predicted_video_max_chunks_per_episode: int | None
    predicted_video_dirname: str
    predicted_video_include_condition_frame: bool


COMPACT_WAN_RUNTIME_FILES = {
    "vae_path": "Wan2.2_VAE.pth",
    "text_checkpoint_path": "models_t5_umt5-xxl-enc-bf16.pth",
    "tokenizer_path": "google/umt5-xxl",
}

COMPACT_WAN_COMPAT_KEYS = (
    "precision",
    "dim",
    "ffn_dim",
    "num_heads",
    "num_layers",
    "head_dim",
    "future_video_size",
)


def load_deploy_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r") as f:
        return normalize_deploy_config(yaml.safe_load(f) or {})


def _with_wan_runtime_paths(config: Dict[str, Any], wan_root: str) -> Dict[str, Any]:
    merged = dict(config)
    merged_model = dict(merged.get("model", {}))
    merged_compact = dict(merged_model.get("compact_wan", {}))
    merged_text = dict(merged.get("text_encoder", {}))
    wan_root_path = Path(str(wan_root))

    merged["wan_path"] = str(wan_root)
    merged_compact["checkpoint_path"] = str(wan_root)
    merged_compact["config_path"] = str(wan_root)
    merged_compact["vae_path"] = str(wan_root_path / COMPACT_WAN_RUNTIME_FILES["vae_path"])
    merged_text["checkpoint_path"] = str(wan_root_path / COMPACT_WAN_RUNTIME_FILES["text_checkpoint_path"])
    merged_text["tokenizer_path"] = str(wan_root_path / COMPACT_WAN_RUNTIME_FILES["tokenizer_path"])

    merged_model["compact_wan"] = merged_compact
    merged["model"] = merged_model
    merged["text_encoder"] = merged_text
    return merged


def normalize_deploy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config)
    checkpoint_path = merged.get("checkpoint_path") or merged.get("ckpt_setting")
    if checkpoint_path:
        merged["checkpoint_path"] = str(checkpoint_path)

    wan_root = merged.get("wan_path")
    if wan_root:
        merged = _with_wan_runtime_paths(merged, str(wan_root))
    return merged


def apply_runtime_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = normalize_deploy_config(config)
    merged_model = dict(merged.get("model", {}))
    merged_common = dict(merged.get("common", {}))
    merged_inference = dict(merged.get("inference", {}))

    checkpoint_path = overrides.get("ckpt_setting") or overrides.get("checkpoint_path")
    if checkpoint_path:
        merged["checkpoint_path"] = str(checkpoint_path)

    wan_root = overrides.get("wan_path")
    if wan_root:
        merged = _with_wan_runtime_paths(merged, str(wan_root))
        merged_model = dict(merged.get("model", {}))

    num_inference_steps = _override_value(overrides, "num_inference_steps", "inference.num_inference_steps")
    num_video_inference_steps = _override_value(
        overrides,
        "num_video_inference_steps",
        "video_inference_steps",
        "vgm_num_inference_steps",
        "inference.num_video_inference_steps",
        "inference.video_inference_steps",
        "inference.vgm_num_inference_steps",
    )
    video_refresh_steps = _override_value(
        overrides,
        "video_refresh_steps",
        "vgm_refresh_steps",
        "video_step_indices",
        "inference.video_refresh_steps",
        "inference.vgm_refresh_steps",
        "inference.video_step_indices",
    )
    video_stop_cosine_threshold = _override_value(
        overrides,
        "video_stop_cosine_threshold",
        "vgm_stop_cosine_threshold",
        "inference.video_stop_cosine_threshold",
        "inference.vgm_stop_cosine_threshold",
    )
    action_skip_cosine_threshold = _override_value(
        overrides,
        "action_skip_cosine_threshold",
        "action_stop_cosine_threshold",
        "inference.action_skip_cosine_threshold",
        "inference.action_stop_cosine_threshold",
    )
    save_predicted_video = _override_value(overrides, "save_predicted_video", "inference.save_predicted_video")
    predicted_video_fps = _override_value(overrides, "predicted_video_fps", "inference.predicted_video_fps")
    predicted_video_every_n_chunks = _override_value(
        overrides,
        "predicted_video_every_n_chunks",
        "inference.predicted_video_every_n_chunks",
    )
    predicted_video_max_chunks_per_episode = _override_value(
        overrides,
        "predicted_video_max_chunks_per_episode",
        "inference.predicted_video_max_chunks_per_episode",
    )
    predicted_video_include_condition_frame = _override_value(
        overrides,
        "predicted_video_include_condition_frame",
        "inference.predicted_video_include_condition_frame",
    )

    if num_inference_steps is not None:
        merged_inference["num_inference_steps"] = int(num_inference_steps)
    if num_video_inference_steps is not None:
        merged_inference["num_video_inference_steps"] = int(num_video_inference_steps)
    if video_refresh_steps is not None:
        merged_inference["video_refresh_steps"] = list(
            _parse_step_indices(video_refresh_steps, field_name="video_refresh_steps")
        )
    if video_stop_cosine_threshold is not None:
        merged_inference["video_stop_cosine_threshold"] = _parse_optional_float(
            video_stop_cosine_threshold,
            field_name="video_stop_cosine_threshold",
        )
    if action_skip_cosine_threshold is not None:
        merged_inference["action_skip_cosine_threshold"] = _parse_optional_float(
            action_skip_cosine_threshold,
            field_name="action_skip_cosine_threshold",
        )
    if save_predicted_video is not None:
        merged_inference["save_predicted_video"] = bool(save_predicted_video)
    if predicted_video_fps is not None:
        merged_inference["predicted_video_fps"] = int(predicted_video_fps)
    if predicted_video_every_n_chunks is not None:
        merged_inference["predicted_video_every_n_chunks"] = int(predicted_video_every_n_chunks)
    if predicted_video_max_chunks_per_episode is not None:
        merged_inference["predicted_video_max_chunks_per_episode"] = int(
            predicted_video_max_chunks_per_episode
        )
    if predicted_video_include_condition_frame is not None:
        merged_inference["predicted_video_include_condition_frame"] = bool(predicted_video_include_condition_frame)
    teacache_enabled = _override_value(
        overrides,
        "teacache_enabled",
        "enable_teacache",
        "inference.teacache.enabled",
        "inference.enable_teacache",
    )
    teacache_delta = _override_value(
        overrides,
        "teacache_delta",
        "inference.teacache.delta",
        "inference.teacache_delta",
    )
    teacache_force_last_step = _override_value(
        overrides,
        "teacache_force_last_step",
        "inference.teacache.force_last_step",
    )
    teacache_action_enabled = _override_value(
        overrides,
        "teacache_action_enabled",
        "inference.teacache.action_enabled",
    )
    teacache_action_delta = _override_value(
        overrides,
        "teacache_action_delta",
        "inference.teacache.action_delta",
    )
    if any(
        value is not None
        for value in (
            teacache_enabled,
            teacache_delta,
            teacache_force_last_step,
            teacache_action_enabled,
            teacache_action_delta,
        )
    ):
        teacache_cfg = dict(merged_inference.get("teacache", {}))
        if teacache_enabled is not None:
            teacache_cfg["enabled"] = teacache_enabled
        if teacache_delta is not None:
            teacache_cfg["delta"] = teacache_delta
        if teacache_force_last_step is not None:
            teacache_cfg["force_last_step"] = teacache_force_last_step
        if teacache_action_enabled is not None:
            teacache_cfg["action_enabled"] = teacache_action_enabled
        if teacache_action_delta is not None:
            teacache_cfg["action_delta"] = teacache_action_delta
        merged_inference["teacache"] = teacache_cfg

    merged["model"] = merged_model
    merged["common"] = merged_common
    merged["inference"] = merged_inference
    return merged


def _select_keys(config: Dict[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
    return {key: config.get(key) for key in keys if key in config}


def _override_value(overrides: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in overrides:
            return overrides[key]
    return None


def _validate_model_config(config: Dict[str, Any], checkpoint_config: Dict[str, Any]) -> None:
    current_model_cfg = config["model"]
    checkpoint_model_cfg = checkpoint_config.get("model")
    if checkpoint_model_cfg is None:
        raise RuntimeError("EfficientWAM checkpoint is missing exported model config metadata")

    expected_compact = _select_keys(current_model_cfg["compact_wan"], COMPACT_WAN_COMPAT_KEYS)
    expected = {
        "compact_wan": expected_compact,
        "action_expert": current_model_cfg["action_expert"],
    }
    actual = {
        "compact_wan": _select_keys(checkpoint_model_cfg.get("compact_wan") or {}, tuple(expected_compact)),
        "action_expert": checkpoint_model_cfg.get("action_expert"),
    }
    mismatches = []
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual[key]!r}")
    if mismatches:
        raise RuntimeError("Deploy config does not match EfficientWAM checkpoint: " + "; ".join(mismatches))


def _load_checkpoint_payload(checkpoint_path: str) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported EfficientWAM checkpoint format at {checkpoint_path}")
    return payload


def _checkpoint_config(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    checkpoint_config = payload.get("config")
    return checkpoint_config if isinstance(checkpoint_config, dict) else None


def _action_normalization_config(config: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(config, dict):
        return None
    if "action_normalization" in config:
        value = config.get("action_normalization")
        return value if isinstance(value, dict) else None
    dataset_cfg = config.get("dataset")
    if isinstance(dataset_cfg, dict):
        value = dataset_cfg.get("action_normalization")
        return value if isinstance(value, dict) else None
    return None


def _merge_action_normalization_config(
    runtime_config: Dict[str, Any],
    checkpoint_config: Dict[str, Any] | None,
) -> Dict[str, Any]:
    checkpoint_norm = _action_normalization_config(checkpoint_config)
    runtime_norm = _action_normalization_config(runtime_config)
    if checkpoint_norm is None and runtime_norm is None:
        return normalize_action_normalization_config(None)
    if checkpoint_norm is None:
        return normalize_action_normalization_config(runtime_norm)
    merged = normalize_action_normalization_config(checkpoint_norm)
    if runtime_norm is not None:
        runtime_norm = dict(runtime_norm)
        if "enabled" in runtime_norm:
            merged["enabled"] = runtime_norm["enabled"]
        if "stats_path" in runtime_norm:
            merged["stats_path"] = runtime_norm["stats_path"]
        if "stats" in runtime_norm:
            merged["stats"] = runtime_norm["stats"]
        if "type" in runtime_norm:
            merged["type"] = runtime_norm["type"]
        merged = normalize_action_normalization_config(merged)
    return merged


def _build_action_normalizer(config: Dict[str, Any]) -> RoboTwinQposNormalizer:
    checkpoint_path = config.get("checkpoint_path") or config.get("model", {}).get("checkpoint_path")
    checkpoint_config = None
    if checkpoint_path:
        payload = _load_checkpoint_payload(str(checkpoint_path))
        checkpoint_config = _checkpoint_config(payload)
    normalization_config = _merge_action_normalization_config(config, checkpoint_config)
    return RoboTwinQposNormalizer.from_config(normalization_config)


def _checkpoint_model_value(
    config: Dict[str, Any],
    checkpoint_config: Dict[str, Any] | None,
    key: str,
    default: Any,
) -> Any:
    model_cfg = config.get("model", {})
    if key in model_cfg:
        return model_cfg[key]
    if checkpoint_config is not None:
        checkpoint_model_cfg = checkpoint_config.get("model", {})
        if key in checkpoint_model_cfg:
            return checkpoint_model_cfg[key]
    return default


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"", "0", "false", "no", "n", "off", "none", "null", "~"}:
            return False
    raise ValueError(f"{field_name} must be a boolean value, got {value!r}")


def _torch_compile_config(config: Dict[str, Any]) -> Dict[str, Any]:
    inference_cfg = config.get("inference", {})
    compile_cfg = inference_cfg.get("torch_compile", {})
    field_name = "inference.torch_compile"
    if isinstance(compile_cfg, (bool, str, int, float)) or compile_cfg is None:
        compile_cfg = {"enabled": compile_cfg}
    elif not isinstance(compile_cfg, dict):
        raise ValueError(f"{field_name} must be a mapping or boolean value, got {compile_cfg!r}")

    return {
        "enabled": _parse_bool(compile_cfg.get("enabled", False), field_name=f"{field_name}.enabled"),
        "action_cache": _parse_bool(
            compile_cfg.get("action_cache", True),
            field_name=f"{field_name}.action_cache",
        ),
        "mode": str(compile_cfg.get("mode", "reduce-overhead")),
        "fullgraph": _parse_bool(
            compile_cfg.get("fullgraph", False),
            field_name=f"{field_name}.fullgraph",
        ),
    }


def _parse_optional_float_list(value: Any, *, field_name: str) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"none", "null", "~"}:
            return None
        loaded = yaml.safe_load(text)
        value = loaded if loaded is not None else text
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    raise ValueError(f"{field_name} must be a list of floats, got {value!r}")


def _parse_optional_cache_delta(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "~"}:
            return None
        value = text
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative float or null, got {value!r}") from exc
    if parsed < 0.0:
        raise ValueError(f"{field_name} must be non-negative, got {parsed}")
    return parsed


def _teacache_config(config: Dict[str, Any]) -> Dict[str, Any]:
    inference_cfg = config.get("inference", {})
    raw = inference_cfg.get("teacache", inference_cfg.get("tea_cache", {}))
    field_name = "inference.teacache"
    if isinstance(raw, (bool, str, int, float)) or raw is None:
        raw = {"enabled": raw}
    elif not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a mapping or boolean value, got {raw!r}")
    return {
        "enabled": _parse_bool(raw.get("enabled", False), field_name=f"{field_name}.enabled"),
        "delta": float(raw.get("delta", 0.0)),
        "poly_coefficients": _parse_optional_float_list(
            raw.get("poly_coefficients"),
            field_name=f"{field_name}.poly_coefficients",
        ),
        "sync_distributed": _parse_bool(
            raw.get("sync_distributed", False),
            field_name=f"{field_name}.sync_distributed",
        ),
        "force_last_step": _parse_bool(
            raw.get("force_last_step", False),
            field_name=f"{field_name}.force_last_step",
        ),
        "action_enabled": _parse_bool(
            raw.get("action_enabled", True),
            field_name=f"{field_name}.action_enabled",
        ),
        "action_delta": _parse_optional_cache_delta(
            raw.get("action_delta"),
            field_name=f"{field_name}.action_delta",
        ),
        "action_force_last_step": _parse_bool(
            raw.get("action_force_last_step", False),
            field_name=f"{field_name}.action_force_last_step",
        ),
    }


def _compile_model_for_inference(model: SmallWAMActionModel, config: Dict[str, Any]) -> None:
    compile_cfg = _torch_compile_config(config)
    if not compile_cfg["enabled"]:
        return
    if not compile_cfg["action_cache"]:
        return
    if getattr(model, "enable_action_teacache", False):
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("inference.torch_compile requires a PyTorch build with torch.compile")

    model.forward_action_with_video_cache = torch.compile(
        model.forward_action_with_video_cache,
        mode=compile_cfg["mode"],
        fullgraph=compile_cfg["fullgraph"],
    )


def _load_model_state(
    model: SmallWAMActionModel,
    checkpoint_path: str,
    config: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported EfficientWAM checkpoint format at {checkpoint_path}")
    checkpoint_config = _checkpoint_config(payload)
    if checkpoint_config is None:
        raise RuntimeError(f"EfficientWAM checkpoint {checkpoint_path} is missing exported config metadata")
    _validate_model_config(config, checkpoint_config)
    model.load_state_dict(state_dict, strict=True)


def build_model_from_config(config: Dict[str, Any], device: str = "cuda") -> SmallWAMActionModel:
    model_cfg = config["model"]
    checkpoint_path = config.get("checkpoint_path") or model_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("EfficientWAM inference requires checkpoint_path to point to an exported EfficientWAM checkpoint")
    payload = _load_checkpoint_payload(str(checkpoint_path))
    checkpoint_config = _checkpoint_config(payload)
    wan_cfg = model_cfg["compact_wan"]
    checkpoint_wan_cfg = checkpoint_config.get("model", {}).get("compact_wan", {}) if checkpoint_config else {}
    future_video_size = wan_cfg.get("future_video_size", checkpoint_wan_cfg.get("future_video_size"))
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
    compact_wan = CompactWANModel.from_teacher_checkpoint(compact_cfg, device=device)
    efficient_wam_cfg = SmallWAMActionConfig(
        compact_wan=compact_wan.config,
        action_dim=int(model_cfg["action_expert"]["action_dim"]),
        state_dim=int(model_cfg["action_expert"]["state_dim"]),
        chunk_size=int(model_cfg["action_expert"]["chunk_size"]),
        ae_dim=int(model_cfg["action_expert"]["dim"]),
        ae_ffn_dim=int(model_cfg["action_expert"]["ffn_dim"]),
        ae_num_layers=int(model_cfg["action_expert"]["num_layers"]),
        wan_frozen=True,
    )
    model = SmallWAMActionModel(efficient_wam_cfg, compact_wan)
    _load_model_state(model, str(checkpoint_path), config, payload)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    teacache_cfg = _teacache_config(config)
    model.configure_teacache(
        enabled=teacache_cfg["enabled"],
        delta=teacache_cfg["delta"],
        num_steps=int(config.get("inference", {}).get("num_video_inference_steps", 0)),
        poly_coefficients=teacache_cfg["poly_coefficients"],
        sync_distributed=teacache_cfg["sync_distributed"],
        force_last_step=teacache_cfg["force_last_step"],
        action_enabled=teacache_cfg["action_enabled"],
        action_delta=teacache_cfg["action_delta"],
        action_force_last_step=teacache_cfg["action_force_last_step"],
    )
    _compile_model_for_inference(model, config)
    return model


def _build_t5_encoder(config: Dict[str, Any], device: str) -> T5EncoderModel:
    model_cfg = config["model"]
    wan_root = model_cfg["compact_wan"]["checkpoint_path"]
    text_cfg = config.get("text_encoder", {})
    checkpoint_path = text_cfg.get(
        "checkpoint_path",
        str(Path(wan_root) / "models_t5_umt5-xxl-enc-bf16.pth"),
    )
    tokenizer_path = text_cfg.get(
        "tokenizer_path",
        str(Path(wan_root) / "google" / "umt5-xxl"),
    )
    return T5EncoderModel(
        text_len=int(text_cfg.get("text_len", 512)),
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
    )


def _parse_step_indices(value: Any, *, field_name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return tuple()
        loaded = yaml.safe_load(text)
        if isinstance(loaded, list):
            value = loaded
        elif isinstance(loaded, int):
            value = [loaded]
        else:
            text = text.strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            value = [part for part in text.replace(",", " ").split() if part]
    elif isinstance(value, int):
        value = [value]
    elif not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of integer action step indices, got {value!r}")

    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only integer action step indices: {value!r}") from exc


def _parse_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "~", "false", "off"}:
            return None
        value = text
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float in [-1, 1] or null, got {value!r}") from exc
    if parsed < -1.0 or parsed > 1.0:
        raise ValueError(f"{field_name} must be in [-1, 1], got {parsed}")
    return parsed


def _explicit_video_refresh_steps(inference_cfg: Dict[str, Any]) -> tuple[int, ...] | None:
    for key in ("video_refresh_steps", "vgm_refresh_steps", "video_step_indices"):
        if key in inference_cfg and inference_cfg[key] is not None:
            return _parse_step_indices(inference_cfg[key], field_name=key)
    return None


def _evenly_spaced_video_refresh_steps(action_steps: int, video_steps: int) -> tuple[int, ...]:
    video_steps = max(1, min(int(video_steps), action_steps))
    if video_steps >= action_steps:
        return tuple(range(action_steps))
    if video_steps == 1:
        return (0,)
    last_action_idx = action_steps - 1
    last_video_idx = video_steps - 1
    return tuple(
        (video_idx * last_action_idx + last_video_idx // 2) // last_video_idx
        for video_idx in range(video_steps)
    )


def _validate_video_refresh_steps(steps: tuple[int, ...], action_steps: int) -> tuple[int, ...]:
    if not steps:
        raise ValueError("inference.video_refresh_steps must not be empty")
    if len(set(steps)) != len(steps):
        raise ValueError(f"inference.video_refresh_steps contains duplicate indices: {list(steps)!r}")
    if tuple(sorted(steps)) != steps:
        raise ValueError(f"inference.video_refresh_steps must be sorted ascending: {list(steps)!r}")
    if steps[0] != 0:
        raise ValueError("inference.video_refresh_steps must start with 0 because the first step builds the cache")
    invalid = [step for step in steps if step < 0 or step >= action_steps]
    if invalid:
        raise ValueError(
            "inference.video_refresh_steps indices must be in "
            f"[0, {action_steps - 1}], got {invalid!r}"
        )
    return steps


def build_runtime_from_config(config: Dict[str, Any], device: str = "cuda") -> EfficientWAMRuntime:
    inference_cfg = config.get("inference", {})
    common_cfg = config.get("common", {})
    max_chunks = inference_cfg.get("predicted_video_max_chunks_per_episode")
    num_inference_steps = int(inference_cfg.get("num_inference_steps", 10))
    num_inference_steps = max(1, num_inference_steps)
    explicit_refresh_steps = _explicit_video_refresh_steps(inference_cfg)
    if explicit_refresh_steps is None:
        num_video_inference_steps = int(
            inference_cfg.get(
                "num_video_inference_steps",
                inference_cfg.get(
                    "video_inference_steps",
                    inference_cfg.get("vgm_num_inference_steps", num_inference_steps),
                ),
            )
        )
        video_refresh_steps = _evenly_spaced_video_refresh_steps(num_inference_steps, num_video_inference_steps)
    else:
        video_refresh_steps = explicit_refresh_steps
    video_refresh_steps = _validate_video_refresh_steps(video_refresh_steps, num_inference_steps)
    num_video_inference_steps = len(video_refresh_steps)
    video_stop_cosine_threshold = _parse_optional_float(
        inference_cfg.get("video_stop_cosine_threshold", inference_cfg.get("vgm_stop_cosine_threshold")),
        field_name="video_stop_cosine_threshold",
    )
    action_skip_cosine_threshold = _parse_optional_float(
        inference_cfg.get("action_skip_cosine_threshold", inference_cfg.get("action_stop_cosine_threshold")),
        field_name="action_skip_cosine_threshold",
    )
    return EfficientWAMRuntime(
        model=build_model_from_config(config, device=device),
        t5_encoder=_build_t5_encoder(config, device=device),
        action_normalizer=_build_action_normalizer(config),
        device=device,
        num_inference_steps=num_inference_steps,
        num_video_inference_steps=num_video_inference_steps,
        video_refresh_steps=video_refresh_steps,
        video_stop_cosine_threshold=video_stop_cosine_threshold,
        action_skip_cosine_threshold=action_skip_cosine_threshold,
        chunk_size=int(config["model"]["action_expert"]["chunk_size"]),
        num_video_frames=int(common_cfg.get("num_video_frames", 8)),
        video_size=(
            int(common_cfg.get("video_height", 384)),
            int(common_cfg.get("video_width", 320)),
        ),
        scene_prefix=str(
            inference_cfg.get(
                "scene_prefix",
                "The whole scene is in a realistic, industrial art style with three views: "
                "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                "The aloha robot is currently performing the following task: ",
            )
        ),
        save_predicted_video=bool(inference_cfg.get("save_predicted_video", False)),
        predicted_video_fps=int(inference_cfg.get("predicted_video_fps", 4)),
        predicted_video_every_n_chunks=max(1, int(inference_cfg.get("predicted_video_every_n_chunks", 1))),
        predicted_video_max_chunks_per_episode=(
            int(max_chunks) if max_chunks is not None else None
        ),
        predicted_video_dirname=str(inference_cfg.get("predicted_video_dirname", "efficient_wam_predicted_video")),
        predicted_video_include_condition_frame=bool(
            inference_cfg.get("predicted_video_include_condition_frame", True)
        ),
    )
