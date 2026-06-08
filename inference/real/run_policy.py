from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .robot_adapter import PlaceholderRobotAdapter
from .runner import EfficientWAMRealRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="EfficientWAM real-robot inference template")
    parser.add_argument("--config", type=str, default="inference/real/deploy_policy.yml")
    parser.add_argument("--max_chunks", type=int, default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    runner = EfficientWAMRealRunner(config, PlaceholderRobotAdapter())
    runner.run_episode(max_chunks=args.max_chunks)


if __name__ == "__main__":
    main()
