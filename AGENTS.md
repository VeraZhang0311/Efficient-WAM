# Agent working notes

## Pod layout

- `/root/Efficient-WAM` is the code workspace. Keep source edits, Git work, and small local configuration here.
- `/workspace` is the persistent network volume. Store checkpoints under `/workspace/checkpoints`, WAN base model files under `/workspace/base_models`, RoboTwin assets under `/workspace/datasets`, and evaluation output under `/workspace/results`.
- Do not commit downloaded models, datasets, virtual environments, local deployment configuration, or generated results.

## Current RoboTwin smoke test

- GPU: one NVIDIA A40. Python environment: `/root/Efficient-WAM/.venv` (Python 3.10).
- Compatible RoboTwin checkout: `/root/Efficient-WAM/.external/RoboTwin`, pinned to `stable_2.0` commit `13c3c47`. Its `policy/EfficientWAM` directory links to this repository's `inference/robotwin/EfficientWAM` package.
- The local deployment file is `inference/robotwin/EfficientWAM/deploy_policy.local.yml`. It uses the **Efficient-WAM-RT** checkpoint and action statistics in `/workspace/checkpoints/Efficient-WAM-RT`, plus the WAN VAE, T5, tokenizer, and `config.json` in `/workspace/base_models/Wan2.2-TI2V-5B`.
- The RT checkpoint requires `future_video_size: [192, 160]` and `video_refresh_steps: [0, 1]`. The full-resolution Efficient-WAM checkpoint requires its matching settings; do not interchange the two configurations.
- For an exported checkpoint, inference constructs the compact WAN architecture and loads the checkpoint strictly. It does not need the full WAN teacher diffusion weights. Training still uses the teacher initialization path.

## Run and inspect the one-episode check

The Pod's NVIDIA Vulkan ICD points to `libGLX_nvidia.so.0`, which failed in this headless container. Generate a temporary ICD that uses the installed EGL library before running RoboTwin:

```bash
python - <<'PY'
import json
with open('/etc/vulkan/icd.d/nvidia_icd.json') as source:
    icd = json.load(source)
icd['ICD']['library_path'] = 'libEGL_nvidia.so.0'
with open('/tmp/efficientwam-nvidia-egl-icd.json', 'w') as target:
    json.dump(icd, target)
PY

cd /root/Efficient-WAM
PATH="$PWD/.venv/bin:$PATH" \
VK_ICD_FILENAMES=/tmp/efficientwam-nvidia-egl-icd.json \
EFFICIENT_WAM_LOG_ROOT=/workspace/results/efficientwam \
bash inference/robotwin/EfficientWAM/eval.sh \
  --config inference/robotwin/EfficientWAM/deploy_policy.local.yml \
  --task adjust_bottle --episode-num 1
```

In the local RoboTwin `stable_2.0` checkout, `script/eval_policy.py` originally hard-coded `test_num = 100` and ignored the override. The current checkout changes that line to `test_num = int(usr_args.get("test_num", usr_args.get("episode_num", 100)))`. Reapply that change if the checkout is recreated; otherwise `--episode-num 1` will run 100 episodes.

The verified smoke run on 2026-09-17 exited with code 0 and reported `adjust_bottle` success **1/1**. Its files are:

- Summary, CSV, and log: `/workspace/results/efficientwam/logs_single_20260917_150442/`
- RoboTwin result and videos: `/workspace/results/robotwin/adjust_bottle/EfficientWAM/demo_clean/workspace/checkpoints/Efficient-WAM-RT/Efficient-WAM-RT_stage3.pt/2026-09-17 15:05:25/`

Inspect them with:

```bash
cat /workspace/results/efficientwam/logs_single_20260917_150442/evaluation_summary.txt
cat /workspace/results/efficientwam/logs_single_20260917_150442/task_success_rates.csv
less /workspace/results/efficientwam/logs_single_20260917_150442/adjust_bottle.log
find /workspace/results/robotwin -type f \( -name '*.mp4' -o -name '_result.txt' \) | sort
```

This one-episode result checks that the environment and evaluation path work. It is not a full RoboTwin benchmark score. Ask for the desired task list and episode count before starting a larger GPU evaluation.
