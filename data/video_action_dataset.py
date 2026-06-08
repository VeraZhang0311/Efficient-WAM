from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class VideoActionSample:
    episode_id: str
    task_id: str
    split: str
    condition_frame_idx: int
    video_indices: List[int]
    text_ref: str
    first_frame: torch.Tensor
    video_frames: torch.Tensor
    initial_state: torch.Tensor
    action_sequence: torch.Tensor
    text_embedding: Optional[torch.Tensor] = None


class VideoActionDataset(Dataset):
    """Video-action dataset wrapper used by Stage 2 and Stage 3."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[index]
        if sample is None:
            raise RuntimeError("Underlying video-action dataset returned None")
        return sample


def video_action_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated: Dict[str, torch.Tensor] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if key in {"language_embedding", "text_embeddings"}:
            collated["text_embeddings"] = values
        elif key in {
            "episode_name",
            "task_name",
            "video_path",
            "qpos_path",
            "lang_path",
            "video_indices",
            "action_indices",
            "source_path",
            "condition_frame_idx",
        }:
            collated[key] = values
        elif isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated
