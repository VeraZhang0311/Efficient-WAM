from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch

from .model_loader import apply_runtime_overrides, build_runtime_from_config, load_deploy_config
from .preprocess import preprocess_robotwin_observation
from .runner import EfficientWAMRunner

logger = logging.getLogger(__name__)


def build_runner(config_path: str, device: str = "cuda") -> EfficientWAMRunner:
    config = load_deploy_config(config_path)
    runtime = build_runtime_from_config(config, device=device)
    return EfficientWAMRunner(runtime=runtime)


def build_runner_from_args(usr_args: Dict[str, Any], device: str = "cuda") -> EfficientWAMRunner:
    config_path = usr_args.get(
        "config_path",
        str(Path(__file__).resolve().parent / "deploy_policy.yml"),
    )
    config = load_deploy_config(config_path)
    config = apply_runtime_overrides(config, usr_args)
    runtime = build_runtime_from_config(config, device=device)
    return EfficientWAMRunner(runtime=runtime)


def get_model(usr_args: Dict[str, Any]) -> EfficientWAMRunner:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return build_runner_from_args(usr_args=usr_args, device=device)


def eval(TASK_ENV, model: EfficientWAMRunner, observation: Dict[str, Any]) -> None:
    processed = preprocess_robotwin_observation(observation, target_size=model.runtime.video_size)
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.set_prediction_video_context(
        root=getattr(TASK_ENV, "eval_video_path", None),
        episode_idx=getattr(TASK_ENV, "test_num", None),
    )
    actions = model.step(processed)
    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")


def reset_model(model: EfficientWAMRunner) -> None:
    model.print_episode_latency_summary(episode_idx=model.predicted_video_episode_idx)
    model.print_episode_similarity_summary()
    model.close_prediction_video()
    model.reset()
    logger.info("EfficientWAM runner reset completed")


def main() -> None:
    parser = argparse.ArgumentParser(description="EfficientWAM canonical RoboTwin inference entry")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "deploy_policy.yml"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    runner = build_runner(args.config, device=args.device)
    sample = preprocess_robotwin_observation(
        {
            "observation": {
                "head_camera": {"rgb": torch.zeros(240, 320, 3, dtype=torch.uint8).numpy()},
                "left_camera": {"rgb": torch.zeros(240, 320, 3, dtype=torch.uint8).numpy()},
                "right_camera": {"rgb": torch.zeros(240, 320, 3, dtype=torch.uint8).numpy()},
            },
            "joint_action": {"vector": torch.zeros(runner.runtime.model.config.state_dim).numpy()},
        },
        target_size=runner.runtime.video_size,
    )
    runner.set_instruction("dummy instruction")
    runner.step(sample)


if __name__ == "__main__":
    main()
