from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import torch

from .model_loader import load_real_policy
from .preprocess import preprocess_images, preprocess_state
from .robot_adapter import RobotAdapter


@dataclass
class RealRuntimeConfig:
    num_inference_steps: int
    video_refresh_steps: Sequence[int]
    execute_steps_per_chunk: int
    action_step_seconds: float


class EfficientWAMRealRunner:
    """Hardware-agnostic real-robot runner template.

    The template exposes the Efficient-WAM-RT inference interfaces: compact expert
    loading, multiscale future latents through `future_video_size`, and asymmetric
    video refresh through `video_refresh_steps`. Platform-specific state/action
    conversion should be implemented inside the provided RobotAdapter.
    """

    def __init__(self, config: Dict[str, Any], adapter: RobotAdapter):
        self.config = config
        self.adapter = adapter
        self.model, self.device, self.dtype = load_real_policy(config)
        common_cfg = config.get("common", {})
        infer_cfg = config.get("inference", {})
        self.camera_names = list(common_cfg.get("camera_names", ["left_wrist", "right_wrist", "head"]))
        self.height = int(common_cfg.get("video_height", 384))
        self.width = int(common_cfg.get("video_width", 320))
        self.runtime = RealRuntimeConfig(
            num_inference_steps=int(infer_cfg.get("num_inference_steps", 5)),
            video_refresh_steps=tuple(int(step) for step in infer_cfg.get("video_refresh_steps", [0, 1])),
            execute_steps_per_chunk=int(infer_cfg.get("execute_steps_per_chunk", 5)),
            action_step_seconds=float(infer_cfg.get("action_step_seconds", 0.3)),
        )

    @torch.no_grad()
    def predict_action_chunk(self) -> np.ndarray:
        observation = self.adapter.get_observation()
        image_tensor = preprocess_images(
            observation.images,
            camera_names=self.camera_names,
            height=self.height,
            width=self.width,
        ).to(self.device, dtype=self.dtype)
        state = preprocess_state(observation.state, device=self.device, dtype=self.dtype)
        raise NotImplementedError(
            "Connect text embedding, VAE latent preparation, and the shared denoising loop "
            "for your robot platform. See inference/robotwin/EfficientWAM/runner.py for "
            "the complete reference implementation."
        )

    def run_episode(self, *, max_chunks: int | None = None) -> None:
        self.adapter.reset()
        chunk_idx = 0
        while max_chunks is None or chunk_idx < max_chunks:
            action_chunk = self.predict_action_chunk()
            for action in action_chunk[: self.runtime.execute_steps_per_chunk]:
                self.adapter.execute_action(action, duration=self.runtime.action_step_seconds)
            chunk_idx += 1
