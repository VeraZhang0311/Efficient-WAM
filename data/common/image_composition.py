from __future__ import annotations

from typing import Mapping, Sequence

import cv2
import numpy as np
import torch

from data.utils.image_utils import resize_with_padding


def _to_hwc_uint8(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def convert_color(image: np.ndarray | torch.Tensor, color_order: str) -> np.ndarray:
    """Return an RGB HWC uint8 image from runtime camera input."""

    arr = _to_hwc_uint8(image)
    order = color_order.lower()
    if order == "rgb":
        return arr
    if order == "bgr":
        return arr[..., ::-1].copy()
    raise ValueError(f"Unsupported color_order={color_order!r}; expected 'rgb' or 'bgr'")


def compose_t_layout_rgb(
    head_rgb: np.ndarray | torch.Tensor,
    left_rgb: np.ndarray | torch.Tensor,
    right_rgb: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Compose head, left wrist, and right wrist RGB images into one T-layout frame."""

    head = _to_hwc_uint8(head_rgb)
    left = _to_hwc_uint8(left_rgb)
    right = _to_hwc_uint8(right_rgb)
    head_h, head_w = head.shape[:2]
    half_w = max(1, head_w // 2)

    def _resize_wrist(img: np.ndarray) -> np.ndarray:
        scale = half_w / float(img.shape[1])
        new_h = max(1, int(round(img.shape[0] * scale)))
        return cv2.resize(img, (half_w, new_h), interpolation=cv2.INTER_AREA)

    left_resized = _resize_wrist(left)
    right_resized = _resize_wrist(right)
    bottom_h = max(left_resized.shape[0], right_resized.shape[0])
    bottom = np.zeros((bottom_h, head_w, 3), dtype=np.uint8)
    bottom[0:left_resized.shape[0], 0:half_w] = left_resized
    bottom[0:right_resized.shape[0], half_w:half_w + right_resized.shape[1]] = right_resized

    if head_h <= 0:
        raise ValueError("Head image has invalid height")
    return np.concatenate([head, bottom], axis=0)


def preprocess_composed_rgb(
    head_rgb: np.ndarray | torch.Tensor,
    left_rgb: np.ndarray | torch.Tensor,
    right_rgb: np.ndarray | torch.Tensor,
    image_size_hw: Sequence[int],
) -> torch.Tensor:
    if len(image_size_hw) != 2:
        raise ValueError(f"image_size_hw must contain [height, width], got {image_size_hw}")
    composed = compose_t_layout_rgb(head_rgb, left_rgb, right_rgb)
    resized = resize_with_padding(composed, (int(image_size_hw[0]), int(image_size_hw[1])))
    return torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0


def preprocess_image_dict(
    image_dict: Mapping[str, np.ndarray | torch.Tensor],
    image_size_hw: Sequence[int],
    keys: Sequence[str] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
) -> torch.Tensor:
    missing = [key for key in keys if key not in image_dict]
    if missing:
        raise KeyError(f"Missing image keys for T-layout composition: {missing}")
    return preprocess_composed_rgb(
        image_dict[keys[0]],
        image_dict[keys[1]],
        image_dict[keys[2]],
        image_size_hw,
    )
