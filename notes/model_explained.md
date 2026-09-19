你当前运行的 RT stage3 checkpoint，WAM 架构主要对应这几处代码：

   部分                                                    代码
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   整体模型：把视频分支和动作分支按层连接起来              inference/robotwin/EfficientWAM/models/small_wam.py:44 中的 SmallWAMActionModel
  ──────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────
   视频分支：精简版 WAN 的封装                             inference/robotwin/EfficientWAM/models/compact_wan.py:41 中的 CompactWANModel
  ──────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────
   视频 Transformer 本体                                   inference/robotwin/EfficientWAM/models/wan_model.py:368 中的 WanVideoModel
  ──────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────
   动作分支：机器人状态与动作的编码、Transformer 和输出    inference/robotwin/EfficientWAM/models/action_expert.py:310 中的 ActionExpert

  建议先看 SmallWAMActionModel：它最能说明 WAM 怎样把视频和动作两条分支组合起来。你现在的推理流程由 inference/robotwin/EfficientWAM/model_loader.py:504 创建这套模型，再严格加载 RT stage3 checkpoint；具体层数和维度在 inference/robotwin/EfficientWAM/
  deploy_policy.local.yml:57。

  仓库顶层的 models/ 是训练侧的对应实现；你正在跑的 RoboTwin 推理使用上表 inference/robotwin/EfficientWAM/models/ 下的代码。