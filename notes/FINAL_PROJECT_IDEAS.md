# Efficient-WAM Final Project Ideas

## 我的判断

你提出的 **“空间自适应 future token”** 是目前最值得做的方向，而且比“直接 LoRA 跨本体”更像一个完整的 world model 研究问题。

我会把题目收紧成：

> **Where Should a Robot Imagine? Task and Contact Aware Token Allocation for Efficient World Action Models**

核心问题是：

> 在固定 future token 预算下，把更多 token 分配给机械臂、夹爪、目标物体和潜在接触区域，是否比均匀低分辨率预测获得更好的控制性能？

这个问题正好接在 Efficient-WAM 后面。论文目前只比较了均匀的 High、Medium、Low 分辨率，future token 数分别是 240、126、60；低分辨率成功率只小幅下降，但没有研究 **这 60 或 126 个 token 应该放在哪里**。论文还明确承认，低分辨率会影响精细操作，并把动态计算分配列为后续方向。[Efficient-WAM 论文](https://arxiv.org/html/2606.10040)

---

## 我最推荐的项目设计

### 第一部分：先证明哪些 token 真正影响动作

这一步可以几乎不训练，风险很低：

- 遮掉背景 future tokens
- 遮掉机械臂和夹爪区域
- 遮掉目标物体区域
- 遮掉接触附近区域
- 在不同 episode 之间交换背景 tokens
- 在不同 episode 之间交换物体 tokens
- 用随机区域作为 control group

观察：

- RoboTwin 成功率变化
- 输出动作与原动作的距离
- action velocity cosine
- 不同任务阶段的敏感区域
- clean 与 randomized background 下的变化

如果交换背景几乎不影响动作，而交换夹爪或目标物体导致成功率大幅下降，这就是很清楚的因果证据。

这也保证了项目有一个稳固结果：即使后面的自适应模型没有提升成功率，前面的 intervention study 本身仍然是一项完整分析。

### 第二部分：预算相同的 foveated future

不要直接比较“更多 token”和“更少 token”，那样结论不够公平。建议比较：

| 方法 | Token 预算 | 表示方式 |
|---|---:|---|
| Uniform Low | 60 | 全局低分辨率 |
| Uniform Medium | 126 | 全局中分辨率 |
| Random Fovea | 126 | 全局低分辨率 + 随机高清区域 |
| Robot Fovea | 126 | 全局低分辨率 + 夹爪高清区域 |
| Object Fovea | 126 | 全局低分辨率 + 目标物体高清区域 |
| Contact Fovea | 126 | 全局低分辨率 + 潜在接触区域 |
| Oracle Fovea | 126 | 使用仿真真值 mask 的上界 |

模型输入可以表示为：

```text
低分辨率全局 future tokens
            +
高分辨率 ROI future tokens
            ↓
       Action Expert
```

代码上主要会改：

- 多尺度 token 生成：`models/wan_model.py` 中的 `prepare_multiscale_video_tokens`
- 多尺度 RoPE 和联合 attention：`models/small_wam.py` 中的 `_multiscale_joint_attention_fast`
- RoboTwin 三相机预处理：`inference/robotwin/EfficientWAM/preprocess.py`

当前三个相机已经被拼成固定布局，因此最容易的第一个版本其实是：

- early phase：head camera 高分辨率
- contact phase：左右 wrist camera 高分辨率
- 其余视图低分辨率

这样不需要先解决目标检测。

### 第三部分：从 oracle ROI 走到可部署 ROI

可以分成三个难度：

1. **Oracle mask**：直接使用 RoboTwin segmentation 或物体位姿，验证方法上限。
2. **Cheap heuristic**：夹爪位置、光流、帧差、gripper closing signal。
3. **Learned gate**：小网络根据当前视觉、语言和机器人状态选择区域。

Final project 做到前两个已经足够完整；learned gate 可以作为 bonus。

---

## 一个更稳、更容易做完的替代题目

### 动态 future compute allocation

Efficient-WAM-RT 当前固定使用 `[2,10]`，也就是视频更新 2 次、动作更新 10 次。论文自己将“固定 inference schedule”列为限制。[论文限制部分](https://arxiv.org/html/2606.10040)

可以研究：

> 简单阶段少更新 future，接触、遮挡或者模型不确定时增加 future 更新，能否在相同平均延迟下提高成功率？

判断不确定性的信号可以是：

- 连续 denoising step 的 video velocity cosine
- action velocity cosine
- 多次 action sampling 的方差
- 夹爪接近物体或即将闭合
- 世界模型预测与新观测之间的误差

这个仓库已经有相应接口：

- `video_stop_cosine_threshold`
- `action_skip_cosine_threshold`
- `video_refresh_steps`
- TeaCache

核心逻辑就在 `inference/robotwin/EfficientWAM/runner.py`。

这个题目基本可以先做 training free 实验，完成概率最高。缺点是论文已经明确提出过这个未来方向，原创性稍弱于空间 token 分配。

---

## LoRA 和跨本体，我怎么看

### LoRA 可以用，但它应该是训练手段，而不是研究问题

“给 Efficient-WAM 加 LoRA”本身偏工程。需要有更明确的问题，例如：

> 冻结视频世界模型后，只调整动作侧参数，能否用少量新本体 demonstration 完成 embodiment transfer？

当前代码没有现成 LoRA 支持，而且 Action Expert 有自定义的四维 `wan_action_qkv` 参数，标准 PEFT 不会自动覆盖所有关键参数。

在单张 A40 上，我建议按顺序尝试：

1. 冻结 compact WAN，只训练 state encoder、action decoder。
2. 冻结 compact WAN，训练整个 Action Expert。
3. 给 Action Expert 的 Linear 和自定义 QKV 添加 LoRA。
4. 最后才考虑 Stage 3 全参训练。

现有 Stage 2 本来就是冻结视频分支、训练动作分支：`configs/robotwin/stage2_action.yaml`。这通常比先实现 LoRA 更直接。

### 跨机械臂值得做，跨人形全身控制风险很大

RoboTwin 自己支持五种机械臂 embodiment，包括 Aloha、Piper、Franka、ARX-X5 和 UR5，因此可以在同一个任务和仿真体系里研究 embodiment transfer。[RoboTwin 2.0](https://arxiv.org/abs/2506.18088)

但当前 Efficient-WAM checkpoint 固定为：

- `state_dim: 14`
- `action_dim: 14`
- Aloha joint-space qpos action

见 `inference/robotwin/EfficientWAM/deploy_policy.local.yml`。

Piper、UR5 和 ARX 的维度较接近，可以先做。Franka 每臂多一个关节，需要换输入输出 head。更合理的方案是加入：

```text
本体状态 → embodiment adapter → canonical action space
canonical action → embodiment adapter → 机器人动作
```

人形全身控制还需要解决平衡、移动、几十维动作、低层 locomotion policy 和全新训练数据。HumanoidBench 虽然提供 15 个全身操作任务和 12 个 locomotion 任务，但它通常依赖层级控制和可靠的低层技能，已经不是简单修改 action head 或 LoRA 能解决的问题。[HumanoidBench](https://arxiv.org/abs/2403.10506)

因此我的排序是：

| 方向 | 研究价值 | 完成概率 | 建议 |
|---|---:|---:|---|
| Causal analysis + foveated future tokens | 很高 | 中高 | **首选** |
| 动态 future denoising budget | 高 | 很高 | 最稳方案 |
| RoboTwin 跨机械臂 few-shot adaptation | 高 | 中 | 第二项目候选 |
| Contact-aware latent supervision | 高 | 中 | 可作为 foveated 的扩展 |
| 人形全身跨本体 | 很高 | 很低 | 不建议作为课程项目 |

---

## 推荐的最终故事线

我会把 proposal 写成三条研究问题：

1. **Where does imagination matter?**  
   哪些 future visual tokens 对动作生成具有因果影响？

2. **Can imagination be spatially allocated?**  
   在固定 token 预算下，global-low + local-high 是否优于 uniform resolution？

3. **When does fine imagination matter?**  
   高分辨率区域是否主要在接触、插入、精细对齐阶段发挥作用？

评测使用：

- 开发阶段：8–10 个代表性任务
- 最终实验：50 tasks × 20 episodes
- clean 和 randomized 两种环境
- success rate、tokens、延迟、显存
- random ROI 和 budget-matched uniform baseline
- 按 grasp、transfer、contact precision、long horizon 分组分析

Efficient-WAM 原论文的消融本身也是每任务 20 个 rollout，所以这个规模合理。课程要求项目明确 state、transition、action interface 和 evaluation criteria，这个设计也能逐项对应。[CIS 6280 项目说明](https://jiataogu.me/cis6280-world-models/)

如果只选一个方向，我建议做 **“causal token intervention + budget-matched foveated future”**。它比单纯 LoRA 更有明确假设，也比直接跨人形更容易在课程时间内得到可信结论。
