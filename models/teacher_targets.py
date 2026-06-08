from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import torch


@dataclass
class TeacherTargets:
    """Unified teacher supervision payload for Stage 1 distillation."""

    hidden_targets: Dict[int, torch.Tensor]
    motion_targets: Dict[int, torch.Tensor]
    video_pred: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, object]] = None


class TeacherTargetProvider(Protocol):
    """Common interface for Stage 1 teacher target providers."""

    def get_teacher_targets(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        include_hidden: bool = True,
        include_motion: bool = True,
        include_video_pred: bool = True,
    ) -> TeacherTargets:
        """Return teacher targets for a Stage 1 batch."""
