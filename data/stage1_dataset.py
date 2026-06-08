from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset


class Stage1DistillDataset(Dataset):
    """Stage 1 wrapper for datasets that return EfficientWAM sample dictionaries."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[index]
        if sample is None:
            raise RuntimeError("Underlying Stage 1 dataset returned None")
        return sample


def stage1_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated: Dict[str, torch.Tensor] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if key in {"text_embeddings", "language_embedding"}:
            collated[key] = values
        elif key in {
            "episode_id",
            "task_id",
            "split",
            "text_ref",
            "video_indices",
            "lang_path",
            "episode_name",
            "task_name",
            "video_path",
            "qpos_path",
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
