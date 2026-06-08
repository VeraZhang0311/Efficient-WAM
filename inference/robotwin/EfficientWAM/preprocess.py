from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch

from .utils.image_utils import resize_with_padding


def _extract_image(observation: Dict[str, Any]) -> np.ndarray:
    if "observation" in observation:
        obs_data = observation["observation"]
        if "head_camera" in obs_data and "left_camera" in obs_data and "right_camera" in obs_data:
            head_img = obs_data["head_camera"]["rgb"]
            left_img = obs_data["left_camera"]["rgb"]
            right_img = obs_data["right_camera"]["rgb"]
            left_img_resized = cv2.resize(left_img, (160, 120))
            right_img_resized = cv2.resize(right_img, (160, 120))
            bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
            return np.concatenate([head_img, bottom_row], axis=0)
        raise ValueError("Missing camera data in RoboTwin observation")
    if "head_camera" in observation:
        return observation["head_camera"]
    if "image" in observation:
        return observation["image"]
    raise ValueError("No visual observation found in RoboTwin observation")


def _extract_state(observation: Dict[str, Any]) -> np.ndarray:
    if "joint_action" in observation and "vector" in observation["joint_action"]:
        return observation["joint_action"]["vector"]
    raise ValueError("No joint_action.vector found in RoboTwin observation")


def preprocess_robotwin_observation(
    observation: Dict[str, Any],
    target_size: Tuple[int, int] = (384, 320),
) -> Dict[str, Any]:
    """Convert RoboTwin observation into EfficientWAM-ready tensors."""
    image = _extract_image(observation)
    if image.shape[:2] != target_size:
        image = resize_with_padding(image, target_size)
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    state = _extract_state(observation)
    if isinstance(state, np.ndarray):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    else:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

    return {
        "first_frame": image_tensor,
        "state": state_tensor,
        "raw_observation": observation,
    }
