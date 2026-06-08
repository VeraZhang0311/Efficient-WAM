from __future__ import annotations

from typing import Dict, Iterable

import cv2
import numpy as np
import torch


def resize_rgb(image: np.ndarray, *, height: int, width: int) -> torch.Tensor:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image with shape HxWx3, got {image.shape}")
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor


def preprocess_images(
    images: Dict[str, np.ndarray],
    *,
    camera_names: Iterable[str],
    height: int,
    width: int,
) -> torch.Tensor:
    tensors = []
    for name in camera_names:
        if name not in images:
            raise KeyError(f"Missing camera view {name!r}")
        tensors.append(resize_rgb(images[name], height=height, width=width))
    return torch.stack(tensors, dim=0)


def preprocess_state(state: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(state, device=device, dtype=dtype).unsqueeze(0)
