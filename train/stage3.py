"""Stage 3 EfficientWAM joint video-action refinement entry."""

from __future__ import annotations

from train.video_action_trainer import run_video_action_training


def main() -> None:
    run_video_action_training(
        default_config="configs/robotwin/stage3_joint.yaml",
        description="Stage 3 EfficientWAM joint video-action refinement",
        default_project="stage3",
    )


if __name__ == "__main__":
    main()
