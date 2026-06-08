"""Build the final EfficientWAM RoboTwin train dataset in one pass."""

from __future__ import annotations

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root.resolve()) not in sys.path:
    sys.path.insert(0, str(project_root.resolve()))

from data.robotwin2.train_dataset_packer import _build_arg_parser, run


def main() -> None:
    run(_build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
