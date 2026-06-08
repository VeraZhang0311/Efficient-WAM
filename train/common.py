from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


logger = logging.getLogger(__name__)


_ANSI_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "gray": "90",
}


def color_logs_enabled() -> bool:
    flag = os.environ.get("EFFICIENT_WAM_COLOR_LOGS", os.environ.get("COLOR_LOGS", "")).lower()
    if flag in {"0", "false", "no", "off", "never"} or "NO_COLOR" in os.environ:
        return False
    if flag in {"1", "true", "yes", "on", "always"} or os.environ.get("FORCE_COLOR"):
        return True
    return os.environ.get("TERM", "") != "dumb"


def color_text(text: str, color: str | None = None, *, bold: bool = False, dim: bool = False) -> str:
    if not color_logs_enabled():
        return text
    codes = []
    if bold:
        codes.append(_ANSI_CODES["bold"])
    if dim:
        codes.append(_ANSI_CODES["dim"])
    if color:
        codes.append(_ANSI_CODES[color])
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def setup_logging(log_level: str = "INFO", rank: int = 0) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
        force=True,
    )


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping at config root: {path}")
    return cfg


def build_arg_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON")
    parser.add_argument("--run_name", type=str, default=None, help="Override run name")
    parser.add_argument("--project_name", type=str, default=None, help="Override tracker project name")
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all", "none"],
        help="Logging backend",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Accelerate checkpoint directory to resume")
    parser.add_argument(
        "--action_init_checkpoint",
        type=str,
        default=None,
        help="Load action_expert.* weights from an exported EfficientWAM checkpoint before training",
    )
    parser.add_argument(
        "--efficient_wam_init_checkpoint",
        type=str,
        default=None,
        help="Load full EfficientWAM weights from an exported checkpoint before training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=None,
        help="Override per-device train batch size",
    )
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader worker count")
    parser.add_argument(
        "--pin_memory",
        type=str,
        default=None,
        choices=["true", "false", "1", "0", "yes", "no"],
        help="Override dataloader pin_memory",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Accepted for launcher compatibility")
    return parser


def parse_bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes"}


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.run_name is not None:
        config["name"] = args.run_name
    if args.project_name is not None:
        config["project_name"] = args.project_name
    if args.report_to is not None:
        config["report_to"] = args.report_to
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.resume_from is not None:
        config["resume_from"] = args.resume_from
    if args.action_init_checkpoint is not None:
        config.setdefault("model", {})["action_init_checkpoint"] = args.action_init_checkpoint
    if args.efficient_wam_init_checkpoint is not None:
        config.setdefault("model", {})["efficient_wam_init_checkpoint"] = args.efficient_wam_init_checkpoint
    if args.gradient_accumulation_steps is not None:
        config["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    if args.per_device_batch_size is not None:
        config["per_device_batch_size"] = int(args.per_device_batch_size)
    if args.num_workers is not None:
        config["num_workers"] = int(args.num_workers)
    if args.pin_memory is not None:
        config["pin_memory"] = parse_bool(args.pin_memory)
    return config


def normalize_report_to(report_to: Optional[str | Iterable[str]]) -> list[str]:
    if report_to is None:
        return ["tensorboard"]
    if isinstance(report_to, str):
        if report_to == "all":
            return ["wandb", "tensorboard"]
        if report_to == "none":
            return []
        return [report_to]
    return list(report_to)


def serialize_tracker_config(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if value is None:
        return "null"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(
            {str(k): serialize_tracker_config(v) for k, v in value.items()},
            ensure_ascii=True,
            sort_keys=True,
        )
    if isinstance(value, (list, tuple)):
        return json.dumps([serialize_tracker_config(item) for item in value], ensure_ascii=True)
    return str(value)


def flatten_tracker_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_tracker_config(value, name))
        else:
            flat[name] = serialize_tracker_config(value)
    return flat


def get_tracker_init_config(config: Dict[str, Any]) -> Dict[str, Any]:
    tracker_config = {str(key): serialize_tracker_config(value) for key, value in config.items()}
    tracker_config.update(flatten_tracker_config(config))
    return tracker_config


def build_accelerator(args: argparse.Namespace, config: Dict[str, Any]):
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration

    log_with = normalize_report_to(config.get("report_to", "tensorboard"))
    run_dir = get_run_dir(config)
    log_dir = get_log_dir(config)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed else None,
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        mixed_precision="bf16",
        log_with=log_with,
        project_dir=str(run_dir),
        project_config=ProjectConfiguration(
            project_dir=str(run_dir),
            logging_dir=str(log_dir),
            total_limit=int(config.get("checkpoint_total_limit", 10)),
        ),
    )
    return accelerator


def get_run_dir(config: Dict[str, Any]) -> Path:
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    return checkpoint_dir / str(config.get("name", config.get("system", "efficient_wam_run")))


def get_export_dir(config: Dict[str, Any]) -> Path:
    return get_run_dir(config) / "exports"


def get_log_dir(config: Dict[str, Any]) -> Path:
    return get_run_dir(config) / "logs"


def init_experiment_trackers(
    accelerator,
    config: Dict[str, Any],
    default_name: str,
    default_project: str | None = None,
) -> None:
    report_to = normalize_report_to(config.get("report_to", "tensorboard"))
    if not report_to:
        return
    run_name = str(config.get("name", default_name))
    project_name = str(config.get("project_name", default_project or default_name))
    tracker_config = get_tracker_init_config(
        {
            **config,
            "tracker_project_name": project_name,
            "tracker_run_name": run_name,
        }
    )
    if report_to == ["tensorboard"]:
        accelerator.init_trackers(project_name, config=tracker_config)
        return
    init_kwargs: Dict[str, Any] = {}
    if "wandb" in report_to:
        init_kwargs["wandb"] = {"name": run_name}
    accelerator.init_trackers(project_name, config=tracker_config, init_kwargs=init_kwargs)


def get_per_device_batch_size(config: Dict[str, Any]) -> int:
    if "per_device_batch_size" in config:
        return max(1, int(config["per_device_batch_size"]))
    return max(1, int(config.get("global_batch_size", 1)))


def get_effective_global_batch_size(config: Dict[str, Any], world_size: int = 1) -> float:
    if "global_batch_size" in config:
        return float(config["global_batch_size"])
    return float(
        get_per_device_batch_size(config)
        * max(1, int(world_size))
        * max(1, int(config.get("gradient_accumulation_steps", 1)))
    )


def add_speed_metrics(
    metrics: Dict[str, float],
    global_step: int,
    last_log_step: int,
    last_log_time: float,
    samples_per_step: float | None = None,
) -> tuple[int, float]:
    now = time.monotonic()
    step_delta = max(1, int(global_step) - int(last_log_step))
    elapsed = max(now - float(last_log_time), 1e-9)
    steps_per_sec = float(step_delta) / elapsed
    metrics["speed/steps_per_sec"] = steps_per_sec
    metrics["speed/sec_per_step"] = elapsed / float(step_delta)
    if samples_per_step is not None:
        metrics["speed/samples_per_sec"] = steps_per_sec * float(samples_per_step)
    return int(global_step), now


def get_optimizer_config(config: Dict[str, Any]) -> Dict[str, float]:
    learning_rate = float(config.get("learning_rate", 1e-4))
    weight_decay = float(config.get("weight_decay", 1e-2))
    return {
        "learning_rate": learning_rate,
        "action_learning_rate": float(config.get("action_learning_rate", learning_rate)),
        "video_learning_rate": float(config.get("video_learning_rate", learning_rate)),
        "weight_decay": weight_decay,
        "action_weight_decay": float(config.get("action_weight_decay", weight_decay)),
        "video_weight_decay": float(config.get("video_weight_decay", weight_decay)),
        "min_lr_ratio": float(config.get("min_lr_ratio", 0.1)),
    }


def get_learning_rate_metrics(optimizer, prefix: str = "lr") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    seen_names: Dict[str, int] = {}
    single_group = len(optimizer.param_groups) == 1
    for idx, group in enumerate(optimizer.param_groups):
        lr = float(group.get("lr", 0.0))
        name = str(group.get("name") or ("model" if single_group else f"group_{idx}"))
        count = seen_names.get(name, 0)
        seen_names[name] = count + 1
        if count:
            name = f"{name}_{count}"
        metrics[f"{prefix}/{name}"] = lr
    return metrics


def get_epoch_progress_metrics(
    global_step: int,
    dataloader,
    gradient_accumulation_steps: int = 1,
    world_size: int = 1,
) -> Dict[str, float]:
    try:
        local_micro_batches_per_epoch = len(dataloader)
    except TypeError:
        return {}

    if local_micro_batches_per_epoch <= 0:
        return {}

    grad_accum = max(1, int(gradient_accumulation_steps))
    optimizer_steps_per_epoch = max(1, math.ceil(local_micro_batches_per_epoch / grad_accum))
    current_epoch_step = 0 if global_step <= 0 else ((global_step - 1) % optimizer_steps_per_epoch) + 1
    fractional_epoch = float(global_step) / float(optimizer_steps_per_epoch)

    return {
        "progress/epoch": fractional_epoch,
        "progress/epoch_index": float(math.floor(fractional_epoch)),
        "progress/epoch_step": float(current_epoch_step),
        "progress/steps_per_epoch": float(optimizer_steps_per_epoch),
        "progress/local_micro_batches_per_epoch": float(local_micro_batches_per_epoch),
        "progress/gradient_accumulation_steps": float(grad_accum),
        "progress/world_size": float(max(1, int(world_size))),
    }


def format_progress_for_log(global_step: int, max_steps: int, metrics: Dict[str, float]) -> str:
    total_steps = max(1, int(max_steps))
    pct = 100.0 * float(global_step) / float(total_steps)
    progress = f"step={global_step}/{total_steps} ({pct:.1f}%)"

    epoch = metrics.get("progress/epoch")
    if epoch is None:
        return progress

    epoch_step = metrics.get("progress/epoch_step")
    steps_per_epoch = metrics.get("progress/steps_per_epoch")
    if epoch_step is not None and steps_per_epoch:
        return f"{progress} epoch={epoch:.3f} epoch_step={int(epoch_step)}/{int(steps_per_epoch)}"
    return f"{progress} epoch={epoch:.3f}"


def _format_metric_value(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return str(numeric)
    abs_value = abs(numeric)
    if abs_value != 0.0 and (abs_value < 1e-3 or abs_value >= 1e4):
        return f"{numeric:.3e}"
    if abs_value < 10:
        return f"{numeric:.4g}"
    if abs_value < 100:
        return f"{numeric:.3f}"
    return f"{numeric:.1f}"


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


class LogTracker:
    def __init__(self, stage: str, max_steps: int, loss_precision: int = 3):
        self.stage = stage
        self.max_steps = max_steps
        self.loss_precision = int(loss_precision)
        self.ema_loss = None
        self.ema_beta = 0.98
        self.ema_samp_sec = None

    def _format_loss(self, value: float) -> str:
        return f"{value:.{self.loss_precision}f}"

    def log_step(
        self,
        global_step: int,
        metrics: Dict[str, float],
        *,
        loss_keys: Iterable[tuple[str, str]],
        lr_keys: Iterable[tuple[str, str]] = (),
        grad_key: str = "grad/norm",
        extra_keys: Iterable[tuple[str, str]] = (),
    ) -> list[str]:
        lines_to_log = []
        
        current_loss = metrics.get("loss/total")
        if current_loss is not None:
            if self.ema_loss is None:
                self.ema_loss = current_loss
            else:
                self.ema_loss = self.ema_beta * self.ema_loss + (1 - self.ema_beta) * current_loss

        samp_sec = metrics.get("speed/samples_per_sec")
        step_sec = metrics.get("speed/steps_per_sec")
        if samp_sec is not None:
            if self.ema_samp_sec is None:
                self.ema_samp_sec = samp_sec
            else:
                self.ema_samp_sec = 0.9 * self.ema_samp_sec + 0.1 * samp_sec

        remaining_steps = self.max_steps - global_step
        eta_seconds = remaining_steps / step_sec if step_sec and step_sec > 0 else 0
        eta_str = format_time(eta_seconds)

        grad_norm = metrics.get(grad_key)
        if current_loss is not None and (math.isnan(current_loss) or math.isinf(current_loss)):
            lines_to_log.append(color_text(f"WARNING step={global_step} loss={current_loss}, stop training or check input batch", "red", bold=True))
        if grad_norm is not None:
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                lines_to_log.append(color_text(f"WARNING step={global_step} grad_norm={grad_norm}, possible instability", "red", bold=True))
            elif grad_norm > 10:
                lines_to_log.append(color_text(f"WARNING step={global_step} grad_norm={grad_norm:.1f}, possible instability", "red", bold=True))
                
        if samp_sec is not None and self.ema_samp_sec is not None and samp_sec < 0.5 * self.ema_samp_sec:
            lines_to_log.append(color_text(f"WARNING step={global_step} throughput significantly lower than average", "yellow", bold=True))

        epoch_num = metrics.get("progress/epoch", 0.0)
        pct = 100.0 * global_step / max(1, self.max_steps)

        stage_lower = self.stage.lower()
        if stage_lower.startswith("stage") and stage_lower[5:].isdigit():
            short_stage = f"S{stage_lower[5:]}"
        else:
            short_stage = self.stage
        
        def dim_text(text: str) -> str:
            return color_text(text, dim=True)

        s_stage = dim_text(short_stage)
        s_step = dim_text(f"{global_step:06d}/{self.max_steps:06d}")
        s_pct = dim_text(f"{pct:.2f}%")
        s_epoch = f"{color_text('ep', 'cyan', bold=True)} {color_text(f'{epoch_num:.3f}', 'cyan')}"
        
        part_prefix = f"{s_stage} {s_step}  {s_pct}  {s_epoch}"
        
        loss_val = self._format_loss(current_loss) if current_loss is not None else "?"
        s_loss = f"{color_text('loss', 'green', bold=True)} {color_text(loss_val, 'green')}"
        s_ema = dim_text(f"ema {self._format_loss(self.ema_loss)}") if self.ema_loss is not None else ""
        part_loss = f"{s_loss}  {s_ema}".strip()
        
        loss_comps = []
        for label, key in loss_keys:
            if label != "total" and key in metrics:
                loss_comps.append(dim_text(f"{label} {self._format_loss(metrics[key])}"))
        part_comps = "  ".join(loss_comps)
        
        lr_vals = []
        for label, key in lr_keys:
            if key in metrics:
                lr_vals.append(color_text(f"{metrics[key]:.1e}", "magenta"))
        s_lr = f"{color_text('lr', 'magenta', bold=True)} {','.join(lr_vals)}" if lr_vals else ""
        s_grad = dim_text(f"grad {grad_norm:.2f}") if grad_norm is not None else ""
        if grad_norm is not None and grad_norm > 10:
            s_grad = color_text(f"grad {grad_norm:.2f}", "red")
        part_opt = f"{s_lr}  {s_grad}".strip()
        
        speed_vals = []
        if step_sec is not None:
            speed_vals.append(f"{step_sec:.2f} step/s")
        s_speed = dim_text("  ".join(speed_vals)) if speed_vals else ""
        s_eta = dim_text(f"ETA {eta_str}")
        part_speed = f"{s_speed}  {s_eta}".strip()
        
        delimiter = dim_text(" | ")
        parts = [p for p in [part_prefix, part_loss, part_comps, part_opt, part_speed] if p]
        single_line = delimiter.join(parts)
        
        lines_to_log.append(single_line)
        
        if global_step % 500 == 0:
            detailed = []
            detailed.append("─" * 56)
            stage_num = short_stage[-1] if short_stage[-1].isdigit() else "X"
            detailed.append(f"Stage{stage_num} @ step {global_step} / {self.max_steps}   epoch {epoch_num:.3f}   ETA {eta_str}")
            detailed.append("\nLoss")
            for label, key in loss_keys:
                if key in metrics:
                    detailed.append(f"  {label:<9} {self._format_loss(metrics[key])}")
            detailed.append("\nOptimization")
            for label, key in lr_keys:
                if key in metrics:
                    detailed.append(f"  lr_{label:<6} {metrics[key]:.2e}")
            if grad_norm is not None:
                detailed.append(f"  grad_norm {grad_norm:.2f}")
            if step_sec is not None:
                detailed.append(f"  step_rate {step_sec:.2f} steps/s")
            
            if extra_keys:
                has_extra = any(key in metrics for _, key in extra_keys)
                if has_extra:
                    detailed.append("\nSigma")
                    for label, key in extra_keys:
                        if key in metrics:
                            detailed.append(f"  {label:<9} {metrics[key]:.3f}")
            detailed.append("─" * 56)
            lines_to_log.append("\n".join(detailed))
            
        return lines_to_log


def reduce_metrics_for_log(accelerator, metrics: Dict[str, float], reduction: str = "mean") -> Dict[str, float]:
    """Average numeric metrics across ranks before console/tracker logging.

    Per-step losses are computed on each rank's local shard first; this reducer
    turns them into global means when each rank uses the same per-device batch
    size, which is how these distributed DataLoaders are configured.
    """
    if getattr(accelerator, "num_processes", 1) <= 1:
        return dict(metrics)

    numeric_items = [
        (key, float(value))
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    ]
    if not numeric_items:
        return dict(metrics)

    import torch

    keys = [key for key, _ in numeric_items]
    values = torch.tensor([value for _, value in numeric_items], device=accelerator.device, dtype=torch.float32)
    reduced_values = accelerator.reduce(values, reduction=reduction).detach().cpu().tolist()

    reduced_metrics = dict(metrics)
    for key, value in zip(keys, reduced_values):
        reduced_metrics[key] = float(value)
    return reduced_metrics


def get_dataloader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    num_workers = int(config.get("num_workers", 0))
    dataloader_config: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(config.get("pin_memory", True)),
    }
    if num_workers > 0:
        dataloader_config["persistent_workers"] = bool(config.get("persistent_workers", True))
        dataloader_config["prefetch_factor"] = max(1, int(config.get("prefetch_factor", 2)))
    return dataloader_config


def make_dataset_sampler(dataset):
    if hasattr(dataset, "make_sampler"):
        return dataset.make_sampler()
    return None


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1
