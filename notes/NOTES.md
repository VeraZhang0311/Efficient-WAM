# Efficient-WAM：从代码到 RoboTwin 评测

这份说明对应当前 Runpod Pod 的 **Efficient-WAM-RT stage3** 推理环境。代码在 `/root/Efficient-WAM`，模型、RoboTwin 资产和评测结果在持久卷 `/workspace`。下面的命令都在这个 Pod 内执行。

## 先看懂各目录

| 路径 | 用途 |
| --- | --- |
| `configs/robotwin/`、`train/stage1.py`、`stage2.py`、`stage3.py` | 三阶段训练配置和入口；只跑已导出的 checkpoint 时不用启动训练。 |
| `data/` | 训练数据读取、转换和动作统计。 |
| `models/` | 模型组件：精简的视频生成分支、动作分支及两者的组合。 |
| `inference/robotwin/EfficientWAM/` | RoboTwin 策略实现和评测入口。`eval.sh` 安排任务；`deploy_policy.py` 对接 RoboTwin；`preprocess.py` 处理相机和关节观测；`model_loader.py` 加载 checkpoint、WAN VAE 和 T5；`runner.py` 预测动作。 |
| `.external/RoboTwin/` | 固定到 `stable_2.0` 的仿真器 checkout。`policy/EfficientWAM` 链接到本项目的推理包。 |
| `.venv/` | Python 3.10 环境。 |
| `/workspace/checkpoints/Efficient-WAM-RT/` | RT stage3 checkpoint 和动作归一化统计。 |
| `/workspace/base_models/Wan2.2-TI2V-5B/` | 推理所需的 WAN VAE、T5、tokenizer 和配置。 |
| `/workspace/datasets/RoboTwin/assets/` | 仿真场景和机器人资产。 |
| `/workspace/results/` | 日志、汇总表和视频。 |

推理流程可以读成：**RoboTwin 提供相机图像、关节状态和任务指令 → Efficient-WAM 预测一段动作 → RoboTwin 执行动作 → 保存成功率和视频**。当前 RT checkpoint 使用 `future_video_size: [192, 160]` 和 `video_refresh_steps: [0, 1]`；不要套用全分辨率 checkpoint 的配置。

## 运行前准备

每次 Pod 重启后，先生成适合这个无头容器的 NVIDIA Vulkan ICD 文件。它位于 `/tmp`，重启后不会保留：

```bash
python3 - <<'PY'
import json
with open('/etc/vulkan/icd.d/nvidia_icd.json') as source:
    icd = json.load(source)
icd['ICD']['library_path'] = 'libEGL_nvidia.so.0'
with open('/tmp/efficientwam-nvidia-egl-icd.json', 'w') as target:
    json.dump(icd, target)
PY
```

当前 Pod 已有虚拟环境、模型文件、RoboTwin 资产链接和本地配置。评测时使用 `inference/robotwin/EfficientWAM/deploy_policy.local.yml`，它指向 RT stage3 checkpoint；仓库中的 `deploy_policy.yml` 是需要填写路径的模板。
本地配置中的 `python_executable: /root/Efficient-WAM/.venv/bin/python` 明确指定评测解释器。`eval.sh` 会在启动任务前检查它能否导入 `sapien`，因此即使当前终端的 `PATH` 不同，也不会误用系统 Python。

## 跑一个任务、一个回合

```bash
cd /root/Efficient-WAM
PATH="$PWD/.venv/bin:$PATH" \
VK_ICD_FILENAMES=/tmp/efficientwam-nvidia-egl-icd.json \
EFFICIENT_WAM_LOG_ROOT=/workspace/results/efficientwam \
bash inference/robotwin/EfficientWAM/eval.sh \
  --config inference/robotwin/EfficientWAM/deploy_policy.local.yml \
  --task adjust_bottle --episode-num 1
```

`--task` 可以换成 `inference/robotwin/EfficientWAM/tasks_all.txt` 中的其他任务名。`--episode-num 1` 只跑一个回合，适合检查环境是否能用。这个结果不是完整基准成绩。

## 跑完整任务列表

下面的命令会读取 `tasks_all.txt` 中的 **50 个任务**，在 GPU 0 上逐个运行，每个任务 20 回合。运行前按需要修改回合数；这是大批量 GPU 评测，不会因为这里只给出命令而自动启动。

```bash
cd /root/Efficient-WAM
PATH="$PWD/.venv/bin:$PATH" \
VK_ICD_FILENAMES=/tmp/efficientwam-nvidia-egl-icd.json \
EFFICIENT_WAM_LOG_ROOT=/workspace/results/efficientwam \
bash inference/robotwin/EfficientWAM/eval.sh \
  --config inference/robotwin/EfficientWAM/deploy_policy.local.yml \
  --all --tasks tasks_all.txt --gpus 0 --episode-num 20
```

`--episode-num` 在这里是**每个任务**的回合数。想先试较小的任务列表，可以把 `--tasks tasks_all.txt` 改成 `--tasks tasks_30.txt`。当前只有一张 A40，`--gpus 0` 会按顺序调度任务。

## 在 `/workspace` 看结果

每次评测都会新建一个带时间戳的日志目录。以下命令找到最近一次评测并查看汇总、成功率表和某个任务的详细日志：

```bash
RUN_DIR=$(ls -dt /workspace/results/efficientwam/logs_* | head -1)
echo "$RUN_DIR"
cat "$RUN_DIR/evaluation_summary.txt"
cat "$RUN_DIR/task_success_rates.csv"
less "$RUN_DIR/adjust_bottle.log"
```

实际执行视频、模型预测视频和 RoboTwin 的原始结果可这样查找：

```bash
find /workspace/results/robotwin -type f \
  \( -name 'episode*.mp4' -o -name '_result.txt' \) | sort
```

每个任务结果目录中的 `episode0.mp4` 是**实际执行视频**；`efficient_wam_predicted_video/episode0.mp4` 是**模型预测的视频**，两者的帧率设置不同。只看最近一次 `adjust_bottle` 的实际视频路径：

```bash
find /workspace/results/robotwin/adjust_bottle -type f -name 'episode0.mp4' \
  ! -path '*/efficient_wam_predicted_video/*' | sort | tail -1
```

## 切换实际执行视频录制方式

编辑 `inference/robotwin/EfficientWAM/deploy_policy.local.yml` 顶部的这两项：

```yaml
eval_video_mode: smooth  # smooth 或 default
eval_video_fps: 25        # 仅 smooth 模式使用
```

- `smooth`：在一个策略动作的仿真执行期间抓取中间画面，当前设为 25 FPS；画面更连续，但会增加录制开销。
- `default`：恢复 RoboTwin 原始行为，每个策略动作保存一帧，文件固定为 10 FPS；大批评测需要降低录制开销时改成此值。此模式忽略 `eval_video_fps`。

这些设置只影响实际执行视频。模型预测视频由同一配置文件中的 `inference.predicted_video_fps` 单独控制。当前 Pod 的 `.external/RoboTwin` 已应用所需修改；如果重新克隆该目录，按 `AGENTS.md` 中的命令重新应用 `robotwin_eval_video.patch`。
