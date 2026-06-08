from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class RobotObservation:
    images: Dict[str, np.ndarray]
    state: np.ndarray
    instruction: str


class RobotAdapter(ABC):
    """Hardware interface expected by the real-robot inference template."""

    @abstractmethod
    def get_observation(self) -> RobotObservation:
        """Return the latest RGB images, robot state, and language instruction."""

    @abstractmethod
    def execute_action(self, action: np.ndarray, *, duration: float) -> None:
        """Send one low-level action to the robot controller."""

    def reset(self) -> None:
        """Optional reset hook for a new episode."""


class PlaceholderRobotAdapter(RobotAdapter):
    """Replace this class with the adapter for your robot platform."""

    def get_observation(self) -> RobotObservation:
        raise NotImplementedError("Implement get_observation() for your robot platform.")

    def execute_action(self, action: np.ndarray, *, duration: float) -> None:
        raise NotImplementedError("Implement execute_action() for your robot platform.")
