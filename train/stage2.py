"""Stage 2 EfficientWAM frozen-video action training entry."""

from __future__ import annotations

from train.video_action_trainer import run_video_action_training


def main() -> None:
    run_video_action_training(
        default_config="configs/robotwin/stage2_action.yaml",
        description="Stage 2 EfficientWAM frozen-video action training",
        default_project="stage2",
    )


if __name__ == "__main__":
    main()
