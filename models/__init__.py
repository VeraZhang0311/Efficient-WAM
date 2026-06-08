"""Core model package for EfficientWAM."""

from .compact_wan import CompactWANConfig, CompactWANModel
from .small_wam import SmallWAMActionConfig, SmallWAMActionModel
from .teacher_targets import TeacherTargets, TeacherTargetProvider

__all__ = [
    "CompactWANConfig",
    "CompactWANModel",
    "SmallWAMActionConfig",
    "SmallWAMActionModel",
    "TeacherTargetProvider",
    "TeacherTargets",
]
