from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .third_party.wan.utils.fm import FlowMatchScheduler

from .model_loader import EfficientWAMRuntime


_LATENCY_PROFILE_METRICS = (
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


def _new_latency_meter() -> Dict[str, float]:
    meter = {
        "calls": 0,
        "t5_ms_sum": 0.0,
        "world_ms_sum": 0.0,
        "vgm_forward_ms_sum": 0.0,
        "vgm_forward_calls": 0,
        "total_ms_sum": 0.0,
        "per_action_ms_sum": 0.0,
    }
    for metric in _LATENCY_PROFILE_METRICS:
        meter[f"{metric}_ms_sum"] = 0.0
        meter[f"{metric}_calls"] = 0
    return meter


def _new_forward_timer() -> Dict[str, Any]:
    timer = {
        "calls": 0,
        "vgm_video_ms_sum": 0.0,
        "vgm_video_cuda_events": [],
    }
    for metric in _LATENCY_PROFILE_METRICS:
        timer[f"{metric}_ms_sum"] = 0.0
        timer[f"{metric}_calls"] = 0
        timer[f"{metric}_cuda_events"] = []
    return timer


def _new_similarity_meter() -> Dict[str, Any]:
    return {
        "chunks": 0,
        "action_steps": 0,
        "action_full_steps": 0,
        "action_cache_steps": 0,
        "video_steps": 0,
        "action_cos_count": 0,
        "action_cos_sum": 0.0,
        "action_full_cos_count": 0,
        "action_full_cos_sum": 0.0,
        "action_cache_cos_count": 0,
        "action_cache_cos_sum": 0.0,
        "video_cos_count": 0,
        "video_cos_sum": 0.0,
        "cache_age_count": 0,
        "cache_age_sum": 0.0,
        "by_step": {},
    }


@dataclass
class EfficientWAMRunner:
    runtime: EfficientWAMRuntime
    observation_history: List[Dict[str, Any]] = field(default_factory=list)
    action_buffer: List[Any] = field(default_factory=list)
    current_instruction: str = ""
    cached_text_embeddings: Optional[List[torch.Tensor]] = None
    action_scheduler: FlowMatchScheduler = field(init=False)
    video_scheduler: FlowMatchScheduler = field(init=False)
    predicted_video_root: Optional[Path] = None
    predicted_video_episode_idx: Optional[int] = None
    predicted_video_chunk_idx: int = 0
    predicted_video_saved_chunks: int = 0
    predicted_video_process: Optional[subprocess.Popen] = None
    predicted_video_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.latency_warmup_calls = 1
        self.action_scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.video_scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.action_scheduler.set_timesteps(num_inference_steps=self.runtime.num_inference_steps, training=False)
        self.video_scheduler.set_timesteps(num_inference_steps=self.runtime.num_video_inference_steps, training=False)
        self.reset_task_latency_stats()
        self.reset_episode_latency_stats()
        self.reset_task_similarity_stats()
        self.reset_episode_similarity_stats()
        atexit.register(self.close_prediction_video)
        atexit.register(self.print_task_similarity_summary)

    def reset(self) -> None:
        self.close_prediction_video()
        self.observation_history.clear()
        self.action_buffer.clear()
        self.cached_text_embeddings = None
        self.current_instruction = ""
        self.action_scheduler.set_timesteps(num_inference_steps=self.runtime.num_inference_steps, training=False)
        self.video_scheduler.set_timesteps(num_inference_steps=self.runtime.num_video_inference_steps, training=False)
        self.reset_episode_latency_stats()
        self.reset_episode_similarity_stats()
        self.predicted_video_chunk_idx = 0
        self.predicted_video_saved_chunks = 0

    @property
    def model(self):
        return self.runtime.model

    @property
    def device(self) -> str:
        return self.runtime.device

    def set_instruction(self, instruction: str) -> None:
        instruction = instruction or ""
        if instruction != self.current_instruction:
            self.current_instruction = instruction
            self.cached_text_embeddings = None

    def set_prediction_video_context(self, root: Optional[str], episode_idx: Optional[int]) -> None:
        if not self.runtime.save_predicted_video or root is None:
            self.close_prediction_video()
            self.predicted_video_root = None
            return
        next_root = Path(root) / self.runtime.predicted_video_dirname
        if next_root != self.predicted_video_root or episode_idx != self.predicted_video_episode_idx:
            self.close_prediction_video()
            self.predicted_video_root = next_root
            self.predicted_video_episode_idx = episode_idx
            self.predicted_video_chunk_idx = 0
            self.predicted_video_saved_chunks = 0

    def _get_text_embeddings(self) -> tuple[List[torch.Tensor], float]:
        if self.cached_text_embeddings is not None:
            return self.cached_text_embeddings, 0.0
        instruction = f"{self.runtime.scene_prefix}{self.current_instruction}"
        self._cuda_synchronize()
        t5_start = time.perf_counter()
        t5_out = self.runtime.t5_encoder([instruction], self.device)
        self._cuda_synchronize()
        t5_encode_ms = (time.perf_counter() - t5_start) * 1000.0
        if isinstance(t5_out, torch.Tensor):
            if t5_out.dim() == 3:
                self.cached_text_embeddings = [t5_out[0].to(self.device, dtype=torch.bfloat16)]
            else:
                raise ValueError(f"Unexpected T5 tensor shape: {tuple(t5_out.shape)}")
        elif isinstance(t5_out, list):
            self.cached_text_embeddings = [
                emb.to(self.device, dtype=torch.bfloat16) for emb in t5_out
            ]
        else:
            raise ValueError("Unexpected T5 encoder output format")
        return self.cached_text_embeddings, t5_encode_ms

    def _initialize_video_latent(
        self,
        first_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_frame = first_frame.to(self.device, dtype=self.model.compact_wan.video_model.precision)
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        with torch.inference_mode():
            condition_latent = self.model.compact_wan.encode_video(first_frame_norm)
            if self.model.compact_wan.is_multiscale:
                future_size = self.model.compact_wan.config.future_video_size
                if future_size is None:
                    raise RuntimeError("Multiscale compact WAN is missing future_video_size")
                channels = int(condition_latent.shape[1])
                height, width = self._latent_spatial_size(future_size)
                future_latent = torch.randn(
                    (1, channels, self.runtime.num_video_frames // 4, height, width),
                    device=self.device,
                    dtype=self.model.compact_wan.video_model.precision,
                )
                return future_latent, condition_latent
        _, channels, _, height, width = condition_latent.shape
        total_latent_frames = 1 + self.runtime.num_video_frames // 4
        video_latent = torch.randn(
            (1, channels, total_latent_frames, height, width),
            device=self.device,
            dtype=self.model.compact_wan.video_model.precision,
        )
        video_latent[:, :, 0:1] = condition_latent
        return video_latent, condition_latent

    @staticmethod
    def _latent_spatial_size(pixel_size: tuple[int, int]) -> tuple[int, int]:
        vae_spatial_downsample = 16
        height, width = (int(value) for value in pixel_size)
        if height % vae_spatial_downsample != 0 or width % vae_spatial_downsample != 0:
            raise ValueError(f"future_video_size must be divisible by {vae_spatial_downsample}, got {pixel_size}")
        return height // vae_spatial_downsample, width // vae_spatial_downsample

    def _timed_vgm_forward(self, batch: Dict[str, Any], timer: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if hasattr(self.model, "reset_latency_profile"):
            self.model.reset_latency_profile()
        outputs = self.model(batch)
        timer["calls"] += 1
        if hasattr(self.model, "consume_latency_profile"):
            profile = self.model.consume_latency_profile()
            timer["vgm_video_ms_sum"] += float(profile.get("vgm_video_ms_sum", 0.0))
            timer["vgm_video_cuda_events"].extend(profile.get("vgm_video_cuda_events", []))
            for metric in _LATENCY_PROFILE_METRICS:
                timer[f"{metric}_ms_sum"] += float(profile.get(f"{metric}_ms_sum", 0.0))
                timer[f"{metric}_calls"] += int(profile.get(f"{metric}_calls", 0))
                timer[f"{metric}_cuda_events"].extend(profile.get(f"{metric}_cuda_events", []))
        return outputs

    @staticmethod
    def _finish_forward_timer(timer: Dict[str, Any]) -> tuple[float, int, Dict[str, float], Dict[str, int]]:
        ms_sum = float(timer["vgm_video_ms_sum"])
        for start_event, end_event in timer["vgm_video_cuda_events"]:
            ms_sum += float(start_event.elapsed_time(end_event))
        profile_ms_sums: Dict[str, float] = {}
        profile_calls: Dict[str, int] = {}
        for metric in _LATENCY_PROFILE_METRICS:
            metric_ms_sum = float(timer[f"{metric}_ms_sum"])
            for start_event, end_event in timer[f"{metric}_cuda_events"]:
                metric_ms_sum += float(start_event.elapsed_time(end_event))
            profile_ms_sums[metric] = metric_ms_sum
            profile_calls[metric] = int(timer[f"{metric}_calls"])
        return ms_sum, int(timer["calls"]), profile_ms_sums, profile_calls

    def _video_refresh_schedule(self) -> Dict[int, int]:
        action_steps = max(1, int(self.runtime.num_inference_steps))
        return {
            int(action_step): video_step
            for video_step, action_step in enumerate(self.runtime.video_refresh_steps)
            if 0 <= int(action_step) < action_steps
        }

    def _sample_action_chunk(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        bool,
        float,
        float,
        float,
        int,
        Dict[str, float],
        Dict[str, int],
        List[Dict[str, Any]],
    ]:
        video_dtype = self.model.compact_wan.video_model.precision
        action_dtype = next(self.model.action_expert.parameters()).dtype
        first_frame = first_frame.to(self.device, dtype=video_dtype)
        state = state.to(self.device, dtype=action_dtype)
        state = self.runtime.action_normalizer.normalize(state)
        text_embeddings, t5_encode_ms = self._get_text_embeddings()

        video_latent, condition_latent = self._initialize_video_latent(first_frame)
        noisy_actions = torch.randn(
            (1, self.runtime.chunk_size, self.model.config.action_dim),
            device=self.device,
            dtype=action_dtype,
        )

        action_timesteps = self.action_scheduler.timesteps.to(device=self.device, dtype=action_dtype)
        video_timesteps = self.video_scheduler.timesteps.to(device=self.device, dtype=video_dtype)
        video_refresh_schedule = self._video_refresh_schedule()
        if hasattr(self.model, "reset_teacache"):
            self.model.reset_teacache(num_steps=max(1, len(video_refresh_schedule)))
        use_video_cache = (
            len(video_refresh_schedule) < self.runtime.num_inference_steps
            or self.runtime.video_stop_cosine_threshold is not None
        )
        video_cache = None
        last_video_step_idx = 0
        last_video_refresh_action_step = 0
        video_branch_stopped = False
        action_reuse_active = False
        cached_action_pred = None
        previous_action_pred_for_control = None
        previous_video_pred_for_control = None
        similarity_events: List[Dict[str, Any]] = [{"kind": "chunk"}]
        vgm_forward_timer = _new_forward_timer()
        self._cuda_synchronize()
        world_start = time.perf_counter()
        with torch.inference_mode():
            for step_idx in range(self.runtime.num_inference_steps):
                current_action_t = action_timesteps[step_idx].expand(1)
                video_step_idx = video_refresh_schedule.get(step_idx)
                do_video_refresh = video_step_idx is not None and not video_branch_stopped
                reused_action = False
                if do_video_refresh:
                    action_reuse_active = False
                    if hasattr(self.model, "reset_action_teacache"):
                        self.model.reset_action_teacache()

                if action_reuse_active and not do_video_refresh and cached_action_pred is not None:
                    outputs = {"action_pred": cached_action_pred}
                    reused_action = True
                elif do_video_refresh or video_cache is None:
                    if video_step_idx is None:
                        video_step_idx = last_video_step_idx
                    last_video_step_idx = video_step_idx
                    last_video_refresh_action_step = step_idx
                    current_video_t = video_timesteps[video_step_idx].expand(1)
                    batch = {
                        "video_t": current_video_t,
                        "initial_state": state,
                        "noisy_actions": noisy_actions,
                        "action_t": current_action_t,
                        "text_embeddings": text_embeddings,
                    }
                    if self.model.compact_wan.is_multiscale:
                        batch["condition_latent"] = condition_latent
                        batch["future_latent"] = video_latent
                    else:
                        batch["video_latent"] = video_latent
                    if use_video_cache:
                        batch["return_video_cache"] = True
                    outputs = self._timed_vgm_forward(batch, vgm_forward_timer)
                    video_cache = outputs.get("video_cache") if use_video_cache else None
                    current_video_pred = self._video_velocity_for_similarity(outputs["video_pred"])
                    if (
                        self.runtime.video_stop_cosine_threshold is not None
                        and previous_video_pred_for_control is not None
                    ):
                        video_cos = self._velocity_cosine(current_video_pred, previous_video_pred_for_control)
                        if video_cos >= self.runtime.video_stop_cosine_threshold:
                            video_branch_stopped = True
                    previous_video_pred_for_control = current_video_pred.detach()
                    similarity_events.append(
                        {
                            "kind": "video",
                            "pred": outputs["video_pred"].detach(),
                            "action_step": step_idx,
                        }
                    )
                    video_latent = self.video_scheduler.step(
                        outputs["video_pred"],
                        current_video_t,
                        video_latent,
                    )
                    if not self.model.compact_wan.is_multiscale:
                        video_latent[:, :, 0:1] = condition_latent
                else:
                    outputs = self.model.forward_action_with_video_cache(
                        {
                            "video_cache": video_cache,
                            "initial_state": state,
                            "noisy_actions": noisy_actions,
                            "action_t": current_action_t,
                            "is_last_action_step": step_idx + 1 >= self.runtime.num_inference_steps,
                        }
                    )

                is_cache_step = not do_video_refresh and video_cache is not None
                current_action_pred = outputs["action_pred"].detach()
                if (
                    self.runtime.action_skip_cosine_threshold is not None
                    and previous_action_pred_for_control is not None
                    and is_cache_step
                    and not reused_action
                ):
                    action_cos = self._velocity_cosine(current_action_pred, previous_action_pred_for_control)
                    if action_cos >= self.runtime.action_skip_cosine_threshold:
                        action_reuse_active = True
                        cached_action_pred = current_action_pred
                previous_action_pred_for_control = current_action_pred
                similarity_events.append(
                    {
                        "kind": "action",
                        "pred": current_action_pred,
                        "action_step": step_idx,
                        "is_cache_step": is_cache_step,
                        "cache_age": step_idx - last_video_refresh_action_step,
                    }
                )
                noisy_actions = self.action_scheduler.step(
                    outputs["action_pred"],
                    current_action_t,
                    noisy_actions,
                )
        self._cuda_synchronize()
        world_model_ms = (time.perf_counter() - world_start) * 1000.0
        (
            vgm_forward_ms_sum,
            vgm_forward_calls,
            profile_ms_sums,
            profile_calls,
        ) = self._finish_forward_timer(vgm_forward_timer)
        actions = self.runtime.action_normalizer.denormalize(noisy_actions)
        predicted_video_has_condition_frame = not self.model.compact_wan.is_multiscale
        return (
            actions,
            video_latent,
            predicted_video_has_condition_frame,
            t5_encode_ms,
            world_model_ms,
            vgm_forward_ms_sum,
            vgm_forward_calls,
            profile_ms_sums,
            profile_calls,
            similarity_events,
        )

    def step(self, processed_observation: Dict[str, Any]) -> Any:
        self.observation_history.append(processed_observation)
        self._cuda_synchronize()
        total_start = time.perf_counter()
        (
            actions,
            predicted_video_latent,
            predicted_video_has_condition_frame,
            t5_encode_ms,
            world_model_ms,
            vgm_forward_ms_sum,
            vgm_forward_calls,
            profile_ms_sums,
            profile_calls,
            similarity_events,
        ) = self._sample_action_chunk(
            first_frame=processed_observation["first_frame"],
            state=processed_observation["state"],
        )
        self._cuda_synchronize()
        total_get_action_ms = (time.perf_counter() - total_start) * 1000.0
        actions_np = actions.squeeze(0).detach().cpu().numpy()
        self.action_buffer.extend(actions_np)
        self._record_latency(
            t5_encode_ms=t5_encode_ms,
            world_model_ms=world_model_ms,
            vgm_forward_ms_sum=vgm_forward_ms_sum,
            vgm_forward_calls=vgm_forward_calls,
            profile_ms_sums=profile_ms_sums,
            profile_calls=profile_calls,
            total_get_action_ms=total_get_action_ms,
            num_actions=len(actions_np),
        )
        self._record_similarity_events(similarity_events)
        self._maybe_save_predicted_video(
            predicted_video_latent,
            has_condition_frame=predicted_video_has_condition_frame,
        )
        return actions_np

    def _maybe_save_predicted_video(self, video_latent: torch.Tensor, *, has_condition_frame: bool) -> None:
        if not self.runtime.save_predicted_video or self.predicted_video_root is None:
            return

        chunk_idx = self.predicted_video_chunk_idx
        self.predicted_video_chunk_idx += 1
        if chunk_idx % self.runtime.predicted_video_every_n_chunks != 0:
            return
        max_chunks = self.runtime.predicted_video_max_chunks_per_episode
        if max_chunks is not None and self.predicted_video_saved_chunks >= max_chunks:
            return

        episode_idx = self.predicted_video_episode_idx
        frames = self._decode_predicted_video(video_latent, has_condition_frame=has_condition_frame)
        output_path = self.predicted_video_root / (
            f"episode{episode_idx}.mp4" if episode_idx is not None else "episode_unknown.mp4"
        )
        self._append_predicted_video_frames(frames, output_path)
        self.predicted_video_saved_chunks += 1

    def _decode_predicted_video(self, video_latent: torch.Tensor, *, has_condition_frame: bool) -> torch.Tensor:
        with torch.inference_mode():
            pixels = self.model.compact_wan.decode_video(video_latent)
        if has_condition_frame and not self.runtime.predicted_video_include_condition_frame and pixels.shape[2] > 1:
            pixels = pixels[:, :, 1:]
        frames = ((pixels[0].clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
        return frames.permute(1, 2, 3, 0).contiguous().cpu()

    def _append_predicted_video_frames(self, frames: torch.Tensor, output_path: Path) -> None:
        if frames.numel() == 0:
            return
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected RGB frames [T, H, W, 3], got {tuple(frames.shape)}")
        if self.predicted_video_process is None:
            self._open_predicted_video_writer(output_path, frames)
        if self.predicted_video_process is None or self.predicted_video_process.stdin is None:
            raise RuntimeError("Predicted video writer is not open")
        self.predicted_video_process.stdin.write(frames.numpy().tobytes())

    def _open_predicted_video_writer(self, output_path: Path, frames: torch.Tensor) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = int(frames.shape[1]), int(frames.shape[2])
        self.predicted_video_path = output_path
        self.predicted_video_process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(max(1, int(self.runtime.predicted_video_fps))),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
        )
        if self.predicted_video_process.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin for predicted video")

    def close_prediction_video(self) -> None:
        process = self.predicted_video_process
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
        output_path = self.predicted_video_path
        self.predicted_video_process = None
        self.predicted_video_path = None
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while writing predicted video: {output_path}")

    def _cuda_synchronize(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset_episode_latency_stats(self) -> None:
        self._episode_latency = _new_latency_meter()
        self._episode_warmup_remaining = self.latency_warmup_calls

    def reset_task_latency_stats(self) -> None:
        self._task_latency = _new_latency_meter()

    def reset_episode_similarity_stats(self) -> None:
        self._episode_similarity = _new_similarity_meter()

    def reset_task_similarity_stats(self) -> None:
        self._task_similarity = _new_similarity_meter()

    @staticmethod
    def _velocity_cosine(current: torch.Tensor, previous: torch.Tensor) -> float:
        current_flat = current.detach().float().flatten(1)
        previous_flat = previous.detach().float().flatten(1).to(device=current_flat.device)
        cos = F.cosine_similarity(current_flat, previous_flat, dim=1, eps=1e-8)
        return float(cos.mean().item())

    @staticmethod
    def _video_velocity_for_similarity(video_pred: torch.Tensor) -> torch.Tensor:
        if video_pred.ndim == 5 and video_pred.shape[2] > 1:
            return video_pred[:, :, 1:]
        return video_pred

    def _record_similarity_chunk_start(self) -> None:
        self._episode_similarity["chunks"] += 1
        self._task_similarity["chunks"] += 1

    def _record_video_similarity(
        self,
        video_pred: torch.Tensor,
        previous_video_pred: Optional[torch.Tensor],
        *,
        action_step: int,
    ) -> torch.Tensor:
        current = self._video_velocity_for_similarity(video_pred)
        self._episode_similarity["video_steps"] += 1
        self._task_similarity["video_steps"] += 1
        if previous_video_pred is not None:
            cos = self._velocity_cosine(current, previous_video_pred)
            self._episode_similarity["video_cos_count"] += 1
            self._episode_similarity["video_cos_sum"] += cos
            self._task_similarity["video_cos_count"] += 1
            self._task_similarity["video_cos_sum"] += cos
            self._record_step_cosine("video", int(action_step), cos)
        return current.detach()

    def _record_action_similarity(
        self,
        action_pred: torch.Tensor,
        previous_action_pred: Optional[torch.Tensor],
        *,
        action_step: int,
        is_cache_step: bool,
        cache_age: int,
    ) -> torch.Tensor:
        current = action_pred.detach()
        branch_prefix = "action_cache" if is_cache_step else "action_full"
        self._episode_similarity["action_steps"] += 1
        self._task_similarity["action_steps"] += 1
        self._episode_similarity[f"{branch_prefix}_steps"] += 1
        self._task_similarity[f"{branch_prefix}_steps"] += 1
        if is_cache_step:
            self._episode_similarity["cache_age_count"] += 1
            self._episode_similarity["cache_age_sum"] += float(cache_age)
            self._task_similarity["cache_age_count"] += 1
            self._task_similarity["cache_age_sum"] += float(cache_age)

        if previous_action_pred is not None:
            cos = self._velocity_cosine(current, previous_action_pred)
            self._episode_similarity["action_cos_count"] += 1
            self._episode_similarity["action_cos_sum"] += cos
            self._episode_similarity[f"{branch_prefix}_cos_count"] += 1
            self._episode_similarity[f"{branch_prefix}_cos_sum"] += cos
            self._task_similarity["action_cos_count"] += 1
            self._task_similarity["action_cos_sum"] += cos
            self._task_similarity[f"{branch_prefix}_cos_count"] += 1
            self._task_similarity[f"{branch_prefix}_cos_sum"] += cos
            self._record_step_cosine("action_all", int(action_step), cos)
            self._record_step_cosine(branch_prefix, int(action_step), cos)
        return current

    @staticmethod
    def _update_step_cosine(meter: Dict[str, Any], metric: str, step: int, cos: float) -> None:
        by_step = meter["by_step"]
        metric_bucket = by_step.setdefault(metric, {})
        values = metric_bucket.setdefault(step, {"count": 0, "sum": 0.0})
        values["count"] += 1
        values["sum"] += cos

    def _record_step_cosine(self, metric: str, step: int, cos: float) -> None:
        self._update_step_cosine(self._episode_similarity, metric, step, cos)
        self._update_step_cosine(self._task_similarity, metric, step, cos)

    def _record_similarity_events(self, events: List[Dict[str, Any]]) -> None:
        previous_action_pred = None
        previous_video_pred = None
        for event in events:
            kind = event.get("kind")
            if kind == "chunk":
                self._record_similarity_chunk_start()
            elif kind == "video":
                previous_video_pred = self._record_video_similarity(
                    event["pred"],
                    previous_video_pred,
                    action_step=int(event["action_step"]),
                )
            elif kind == "action":
                previous_action_pred = self._record_action_similarity(
                    event["pred"],
                    previous_action_pred,
                    action_step=int(event["action_step"]),
                    is_cache_step=bool(event["is_cache_step"]),
                    cache_age=int(event["cache_age"]),
                )

    def _record_latency(
        self,
        t5_encode_ms: float,
        world_model_ms: float,
        vgm_forward_ms_sum: float,
        vgm_forward_calls: int,
        profile_ms_sums: Dict[str, float],
        profile_calls: Dict[str, int],
        total_get_action_ms: float,
        num_actions: int,
    ) -> None:
        if self._episode_warmup_remaining > 0:
            self._episode_warmup_remaining -= 1
            return
        per_action_ms = total_get_action_ms / max(1, num_actions)
        self._update_latency_meter(
            self._episode_latency,
            t5_encode_ms,
            world_model_ms,
            vgm_forward_ms_sum,
            vgm_forward_calls,
            profile_ms_sums,
            profile_calls,
            total_get_action_ms,
            per_action_ms,
        )
        self._update_latency_meter(
            self._task_latency,
            t5_encode_ms,
            world_model_ms,
            vgm_forward_ms_sum,
            vgm_forward_calls,
            profile_ms_sums,
            profile_calls,
            total_get_action_ms,
            per_action_ms,
        )

    @staticmethod
    def _update_latency_meter(
        meter: Dict[str, float],
        t5_encode_ms: float,
        world_model_ms: float,
        vgm_forward_ms_sum: float,
        vgm_forward_calls: int,
        profile_ms_sums: Dict[str, float],
        profile_calls: Dict[str, int],
        total_get_action_ms: float,
        per_action_ms: float,
    ) -> None:
        meter["calls"] += 1
        meter["t5_ms_sum"] += t5_encode_ms
        meter["world_ms_sum"] += world_model_ms
        meter["vgm_forward_ms_sum"] += vgm_forward_ms_sum
        meter["vgm_forward_calls"] += vgm_forward_calls
        for metric in _LATENCY_PROFILE_METRICS:
            meter[f"{metric}_ms_sum"] += float(profile_ms_sums.get(metric, 0.0))
            meter[f"{metric}_calls"] += int(profile_calls.get(metric, 0))
        meter["total_ms_sum"] += total_get_action_ms
        meter["per_action_ms_sum"] += per_action_ms

    @staticmethod
    def _summarize_latency_meter(meter: Dict[str, float]) -> Optional[Dict[str, float]]:
        calls = int(meter["calls"])
        if calls <= 0:
            return None
        vgm_forward_calls = int(meter["vgm_forward_calls"])
        summary = {
            "calls": calls,
            "avg_t5": meter["t5_ms_sum"] / calls,
            "avg_world": meter["world_ms_sum"] / calls,
            "vgm_forward_calls": vgm_forward_calls,
            "avg_vgm_forward": (
                meter["vgm_forward_ms_sum"] / vgm_forward_calls
                if vgm_forward_calls > 0
                else 0.0
            ),
            "avg_total": meter["total_ms_sum"] / calls,
            "avg_per_action": meter["per_action_ms_sum"] / calls,
        }
        for metric in _LATENCY_PROFILE_METRICS:
            metric_calls = int(meter[f"{metric}_calls"])
            metric_ms_sum = float(meter[f"{metric}_ms_sum"])
            summary[f"{metric}_calls"] = metric_calls
            summary[f"avg_{metric}"] = metric_ms_sum / metric_calls if metric_calls > 0 else 0.0
            summary[f"avg_{metric}_per_vgm"] = (
                metric_ms_sum / vgm_forward_calls if vgm_forward_calls > 0 else 0.0
            )
        return summary

    def get_episode_latency_summary(self) -> Optional[Dict[str, float]]:
        return self._summarize_latency_meter(self._episode_latency)

    def get_task_latency_summary(self) -> Optional[Dict[str, float]]:
        return self._summarize_latency_meter(self._task_latency)

    @staticmethod
    def _format_latency_summary(label: str, summary: Dict[str, float]) -> str:
        return (
            f"Latency Summary [{label}] "
            f"avg_t5={summary['avg_t5']:.2f}ms, "
            f"avg_world={summary['avg_world']:.2f}ms, "
            f"avg_vgm_forward={summary['avg_vgm_forward']:.2f}ms, "
            f"vgm_forward_calls={int(summary['vgm_forward_calls'])}, "
            f"avg_vgm_prepare={summary['avg_vgm_prepare_per_vgm']:.2f}ms, "
            f"avg_vgm_layer_setup={summary['avg_vgm_layer_setup_per_vgm']:.2f}ms, "
            f"avg_joint_attn={summary['avg_joint_attn']:.2f}ms, "
            f"avg_joint_attn_per_vgm={summary['avg_joint_attn_per_vgm']:.2f}ms, "
            f"joint_attn_calls={int(summary['joint_attn_calls'])}, "
            f"avg_vgm_cross_attn={summary['avg_vgm_cross_attn_per_vgm']:.2f}ms, "
            f"avg_vgm_ffn={summary['avg_vgm_ffn_per_vgm']:.2f}ms, "
            f"avg_vgm_head={summary['avg_vgm_head_per_vgm']:.2f}ms, "
            f"avg_action_prepare={summary['avg_action_prepare_per_vgm']:.2f}ms, "
            f"avg_action_ffn={summary['avg_action_ffn_per_vgm']:.2f}ms, "
            f"avg_action_head={summary['avg_action_head_per_vgm']:.2f}ms, "
            f"avg_total={summary['avg_total']:.2f}ms, "
            f"avg_per_action={summary['avg_per_action']:.2f}ms"
        )

    def print_episode_latency_summary(self, episode_idx: Optional[int] = None) -> None:
        summary = self.get_episode_latency_summary()
        if summary is not None:
            label = f"episode {episode_idx}" if episode_idx is not None else "episode"
            print(self._format_latency_summary(label, summary), flush=True)

    @staticmethod
    def _average(meter: Dict[str, float], sum_key: str, count_key: str) -> Optional[float]:
        count = int(meter[count_key])
        if count <= 0:
            return None
        return meter[sum_key] / count

    @classmethod
    def _summarize_similarity_meter(cls, meter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        chunks = int(meter["chunks"])
        if chunks <= 0:
            return None
        by_step = {}
        for metric, metric_bucket in meter["by_step"].items():
            by_step[metric] = {
                int(step): {
                    "count": int(values["count"]),
                    "avg_cos": values["sum"] / max(1, int(values["count"])),
                }
                for step, values in metric_bucket.items()
                if int(values["count"]) > 0
            }
        return {
            "chunks": chunks,
            "action_steps": int(meter["action_steps"]),
            "action_full_steps": int(meter["action_full_steps"]),
            "action_cache_steps": int(meter["action_cache_steps"]),
            "video_steps": int(meter["video_steps"]),
            "avg_action_cos": cls._average(meter, "action_cos_sum", "action_cos_count"),
            "avg_action_full_cos": cls._average(meter, "action_full_cos_sum", "action_full_cos_count"),
            "avg_action_cache_cos": cls._average(meter, "action_cache_cos_sum", "action_cache_cos_count"),
            "avg_video_cos": cls._average(meter, "video_cos_sum", "video_cos_count"),
            "avg_cache_age": cls._average(meter, "cache_age_sum", "cache_age_count"),
            "by_step": by_step,
        }

    @staticmethod
    def _format_optional(value: Optional[float]) -> str:
        return "nan" if value is None else f"{value:.6f}"

    @classmethod
    def _format_similarity_summary(cls, label: str, summary: Dict[str, Any]) -> str:
        return (
            f"EfficientWAM {label} velocity cosine summary: "
            f"chunks={summary['chunks']} "
            f"action_steps={summary['action_steps']} "
            f"action_full_steps={summary['action_full_steps']} "
            f"action_cache_steps={summary['action_cache_steps']} "
            f"video_steps={summary['video_steps']} "
            f"avg_action_cos={cls._format_optional(summary['avg_action_cos'])} "
            f"avg_action_full_cos={cls._format_optional(summary['avg_action_full_cos'])} "
            f"avg_action_cache_cos={cls._format_optional(summary['avg_action_cache_cos'])} "
            f"avg_video_cos={cls._format_optional(summary['avg_video_cos'])} "
            f"avg_cache_age={cls._format_optional(summary['avg_cache_age'])}"
        )

    @staticmethod
    def _format_similarity_step_lines(label: str, summary: Dict[str, Any]) -> List[str]:
        lines = []
        by_step = summary.get("by_step", {})
        for metric in sorted(by_step):
            for step in sorted(by_step[metric]):
                values = by_step[metric][step]
                lines.append(
                    f"EfficientWAM {label} velocity cosine step: "
                    f"metric={metric} "
                    f"step={step} "
                    f"count={values['count']} "
                    f"avg_cos={values['avg_cos']:.6f}"
                )
        return lines

    def get_episode_similarity_summary(self) -> Optional[Dict[str, Any]]:
        return self._summarize_similarity_meter(self._episode_similarity)

    def get_task_similarity_summary(self) -> Optional[Dict[str, Any]]:
        return self._summarize_similarity_meter(self._task_similarity)

    def print_episode_similarity_summary(self) -> None:
        summary = self.get_episode_similarity_summary()
        if summary is not None:
            for line in self._format_similarity_step_lines("episode", summary):
                print(line, flush=True)

    def print_task_similarity_summary(self) -> None:
        summary = self.get_task_similarity_summary()
        if summary is not None:
            for line in self._format_similarity_step_lines("task", summary):
                print(line, flush=True)
