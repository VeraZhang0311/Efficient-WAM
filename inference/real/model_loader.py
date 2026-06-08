from __future__ import annotations

from typing import Any, Dict

import torch

from inference.robotwin.EfficientWAM.model_loader import build_model_from_config


def load_real_policy(config: Dict[str, Any]):
    device = str(config.get("device", "cuda"))
    model = build_model_from_config(config, device=device)
    dtype = model.compact_wan.video_model.precision
    return model, torch.device(device), dtype
