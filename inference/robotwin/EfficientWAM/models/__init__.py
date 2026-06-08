"""Local model package for copyable EfficientWAM RoboTwin deployment."""

from .compact_wan import CompactWANConfig, CompactWANModel
from .small_wam import SmallWAMActionConfig, SmallWAMActionModel

__all__ = [
    "CompactWANConfig",
    "CompactWANModel",
    "SmallWAMActionConfig",
    "SmallWAMActionModel",
]
