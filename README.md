# Efficient-WAM

Efficient-WAM is a lightweight World-Action Model for efficient robot control. It reduces the cost of future imagination while preserving action performance through a compact video expert, multiscale future latents, and asymmetric video-action denoising.

[![Project Page](https://img.shields.io/badge/Project%20Page-Efficient--WAM-2ea44f?style=for-the-badge)](https://efficientwam.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.10040-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.10040)
[![MODELS](https://img.shields.io/badge/MODELS-Hugging%20Face-f5c542?style=for-the-badge)](https://huggingface.co/jiajun0613/Efficient-WAM_RoboTwin)
[![License](https://img.shields.io/badge/License-Apache--2.0-4f7cff?style=for-the-badge)](LICENSE)

## Overview

Efficient-WAM is built around an action-centric view of future prediction: the video branch should preserve control-relevant geometry, motion, and contact cues rather than photorealistic detail. The system combines three efficiency components:

1. Compact video expert distilled from WAN-2.2-5B.
2. Multiscale video-latent layout with low-resolution future latents.
3. Asymmetric video-action denoising for fast inference.

## Repository Structure

```text
Efficient-WAM/
├── configs/              # Training configs for RoboTwin experiments
├── data/                 # Dataset loading, preprocessing, and packing utilities
├── envs/                 # Conda environment files
├── inference/
│   ├── robotwin/         # RoboTwin evaluation policy package
│   └── real/             # Hardware-agnostic real-robot inference template
├── models/               # Efficient-WAM model components
├── scripts/              # Data preparation and training scripts
├── third_party/wan/      # Local WAN-2.2 interface used by Efficient-WAM
├── train/                # Stage-1/2/3 training entry points
└── utils/                # Shared utilities
```

## Installation

Create the training environment:

```bash
conda env create -f envs/train.yaml
conda activate EfficientWAM
```

Create the RoboTwin evaluation environment:

```bash
conda env create -f envs/robotwin_eval.yaml
conda activate RoboTwin
```

## Data Preparation

Download the RoboTwin 2.0 dataset and convert it into the Efficient-WAM training format:

```bash
bash scripts/robotwin/download_robotwin_dataset.sh
bash scripts/robotwin/convert_robotwin_dataset.sh
python scripts/robotwin/build_train_dataset.py --config configs/robotwin/train_dataset.yaml
python scripts/robotwin/prepare_stage1_pca.py --config configs/robotwin/stage1_pca_prep.yaml
```

Before running these commands, update the placeholder paths in the config files, including dataset roots, WAN-2.2 paths, and output directories.

## Training

Efficient-WAM uses a three-stage training recipe.

Stage 1: compact video expert distillation.

```bash
STAGE=stage1 bash scripts/train.sh
```

Stage 2: frozen-video action training.

```bash
STAGE=stage2 bash scripts/train.sh
```

Stage 3: joint video-action refinement.

```bash
STAGE=stage3 bash scripts/train.sh
```

The default configs are:

```text
configs/robotwin/stage1_video_distill.yaml
configs/robotwin/stage2_action.yaml
configs/robotwin/stage3_joint.yaml
```

## RoboTwin Evaluation

The RoboTwin policy package is provided in:

```text
inference/robotwin/EfficientWAM/
```

Copy or link this folder into the RoboTwin benchmark policy directory, then update `deploy_policy.yml` with the exported checkpoint path and WAN-2.2 path. The package supports multiscale future latents through `future_video_size` and asymmetric denoising through `video_refresh_steps`.
When using a virtual environment, leave `conda_env` empty and set `python_executable` to that environment's absolute Python path. The evaluation script checks for `sapien` before starting tasks.

Example:

```bash
cd inference/robotwin/EfficientWAM
bash eval.sh --config deploy_policy.yml --all
```

The actual rollout recording (`episodeN.mp4`) uses `eval_video_mode` in the
deployment config. Use `default` for RoboTwin's original 10 FPS recording with
one frame per policy action. Use `smooth` to capture intermediate simulator
frames at `eval_video_fps` (25 by default). `eval_video_fps` does not change the
`default` mode. The model's predicted video has a separate
`inference.predicted_video_fps` setting.

For RoboTwin `stable_2.0` at commit `13c3c47`, apply the included integration
patch once after cloning RoboTwin:

```bash
git -C /path/to/RoboTwin apply /path/to/Efficient-WAM/inference/robotwin/EfficientWAM/robotwin_eval_video.patch
```

## Real-Robot Inference Template

The real-robot template is provided in:

```text
inference/real/
```

It is hardware-agnostic and does not include any platform-specific SDK. To deploy on a new robot, implement `RobotAdapter` in `robot_adapter.py`, connect platform-specific observation preprocessing and action execution, and update `deploy_policy.yml`.

The template exposes the same Efficient-WAM-RT interfaces used in the paper:

- compact video expert checkpoint loading;
- low-resolution future latents via `future_video_size`;
- asymmetric video refresh via `video_refresh_steps`.

## Citation

If you find this project useful, please cite:

```bibtex
@article{li2026efficientwam,
  title   = {Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination},
  author  = {Li, Jiajun and Guo, Tiecheng and Ye, Yifan and Zhang, Rongyu and Chi, Xiaowei and Sun, Qianpu and Li, Ying and Lou, Yunfan and Huang, Yan and Lu, Zhihe and Guo, Meng and Zhang, Shanghang},
  journal = {arXiv preprint arXiv:2606.10040},
  year    = {2026},
  eprint  = {2606.10040},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url     = {https://arxiv.org/abs/2606.10040}
}
```

## Acknowledgements

This codebase builds on ideas and components from WAN-2.2, RoboTwin 2.0, and prior World-Action Model systems. Third-party code and assets retain their original licenses.

## License

This project is released under the Apache License 2.0. Third-party code and assets, including components adapted from WAN-2.2, retain their original licenses.
