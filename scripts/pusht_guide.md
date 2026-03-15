# PushT 数据集训练与评估指南

## 🎯 核心成果

**ACT 策略在 PushT 环境的成功率从 0.6% 提升至 66%**（110倍提升！）

| 配置               | pc\_success ↑ | 提升幅度     |
| ---------------- | ------------- | -------- |
| ACT 基线 (默认)      | 0.6%          | -        |
| Diffusion Policy | 21.0%         | 35x      |
| **ACT + 最优配置**   | **66.0%**     | **110x** |

> 本实验所有评估均在n\_episodes=500, batch\_size=50下进行

### 📊 评估指标说明

| 指标                   | 说明                                                              |  优化方向 |
| -------------------- | --------------------------------------------------------------- | :---: |
| **pc\_success**      | 成功率（百分比）。当单次 episode 中覆盖率达到 95% 时，该 episode 被判定为成功。**最重要的评估指标** | **↑** |
| **avg\_max\_reward** | 平均最大奖励。每个 episode 的最大奖励取平均，当覆盖率达到 95% 时奖励为 1.0。                 | **↑** |
| **avg\_sum\_reward** | 平均累积奖励。每个 episode 所有时间步的奖励之和取平均。                                | **↑** |
| **eval\_s**          | 评估总耗时（秒）                                                        | **↓** |
| **eval\_ep\_s**      | 平均每个 episode 的评估耗时（秒）                                           | **↓** |

> **↑** 越高越好 | **↓** 越低越好

### 💡 关键发现

1. **Chunk Size 是关键**：默认 `chunk_size=100` 不适合 PushT，`chunk_size=8` 效果最佳（倒U型曲线）
2. **Temporal Ensembling 有害**：TE 会破坏训练-推理一致性，成功率腰斩
3. **数据增强巨大提升**：Affine 变换将成功率从 28% → 42%（+50%）
4. **Decoder 深度重要**：`n_decoder_layers=7` vs `=1`，从 16% → 28%（+75%）
5. **长训练 + Cosine Scheduler**：500k步 + cosine 衰减达到最优 66%

***

## 环境准备

确保已安装 LeRobot 基础环境（miniforge 和 uv 显著加速安装）：

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot

conda install ffmpeg -c conda-forge
uv pip install -e .

# 安装 PushT 仿真环境依赖：
uv pip install -e ".[pusht]"

# 若服务器无桌面环境
uv pip uninstall opencv-python
uv pip install opencv-python-headless
```

# 一、基线训练

1.1 Diffusion Policy 训练

```bash
export HF_USER=lepao
export JOB_NAME=diffusion_pusht_test
lerobot-train \
    --env.type=pusht \
    --policy.type=diffusion \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --batch_size=64 \
    --steps=100000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false
```

```bash
lerobot-eval \
    --env.type=pusht \
    --policy.path=outputs/train/${JOB_NAME}/checkpoints/last/pretrained_model \
    --output_dir=outputs/eval/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --eval.n_episodes=500 \
    --eval.batch_size=50 \
    --policy.device=cuda \
    --policy.use_amp=false
```

1.2 ACT Policy 训练

```bash
export HF_USER=lepao
export JOB_NAME=act_pusht_test
lerobot-train \
    --env.type=pusht \
    --policy.type=act \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --batch_size=64 \
    --steps=100000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false
```

> \[!TIP]
> Episode 终止条件: T 形方块覆盖率达到 95% 或达到最大步数限制（默认 300 步）

### 1.5. 基线评估结果

| 策略        | pc\_success ↑ | avg\_max\_reward ↑ | avg\_sum\_reward ↑ | eval\_s ↓ | eval\_ep\_s ↓ |
| --------- | ------------- | ------------------ | ------------------ | --------- | ------------- |
| Diffusion | 21.0          | 0.574              | 62.51              | 734.11    | 1.47          |
| ACT       | 0.6           | 0.397              | 36.61              | 335.80    | 0.67          |

# 二、 ACT 调优

ACT 基线的默认超参数针对 ALOHA 双臂操作任务设计（chunk\_size=100、n\_action\_steps=100），直接用于 PushT 时效果较差。

### 2.1 Chunk Size 探索

不同 chunk\_size 影响模型的时间预测范围和推理频率。chunk\_size 越小，模型被查询的频率越高，闭环控制越强。

探索的 chunk\_size 值：**4, 8, 16, 32, 64**。

```bash
export HF_USER=lepao
export chunk=64
export JOB_NAME=act_pusht_chunk_${chunk}
lerobot-train \
    --env.type=pusht \
    --policy.type=act \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --batch_size=64 \
    --steps=100000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.chunk_size=${chunk} \
    --policy.n_action_steps=${chunk} \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true
```

| 策略            | pc\_success ↑ | avg\_max\_reward ↑ | avg\_sum\_reward ↑ | eval\_s ↓ | eval\_ep\_s ↓ |
| ------------- | ------------- | ------------------ | ------------------ | --------- | ------------- |
| chunk\_4      | 9.4           | 0.489              | 82.70              | 452.79    | 0.91          |
| chunk\_8      | 22.6          | 0.611              | 77.67              | 810.25    | 1.62          |
| chunk\_16     | 21.0          | 0.650              | 61.90              | 833.18    | 1.67          |
| chunk\_32     | 10.0          | 0.624              | 60.71              | 822.05    | 1.64          |
| chunk\_64     | 2.0           | 0.482              | 42.68              | 821.10    | 1.64          |
| 基线 chunk\_100 | 0.6           | 0.397              | 36.61              | 335.80    | 0.67          |

**分析**：
- **倒U型分布**：chunk\_8 (22.6%) 和 chunk\_16 (21.0%) 位于最优区间，两端急剧下降。chunk\_4 (9.4%) 因推理过于频繁引入累积误差；chunk\_64 (2.0%) 和基线 chunk\_100 (0.6%) 则开环时间过长，无法及时纠偏。
- **最大奖励趋势**：chunk\_16 的 avg\_max\_reward (0.650) 最高，说明更长的动作块偶尔能达到较高覆盖率，但一致性不如 chunk\_8。
- **推理速度权衡**：chunk\_4 最快 (0.91s/ep)，chunk\_8\~64 耗时相近 (~1.6s/ep)，基线 chunk\_100 因开环步数长反而最快 (0.67s/ep)。综合性能与效率，chunk\_8 为最优选择。

### 2.2 Temporal Ensembling (推理侧)

Temporal Ensembling 是 ACT 论文推荐的推理机制：**每步都调用模型预测完整动作块，然后对当前时间步的多次预测做指数加权平均**。动作更平滑，抑制预测抖动，**不需要重新训练**，直接对已训练模型生效

```bash
export chunk=64
lerobot-eval \
    --env.type=pusht \
    --policy.path=outputs/train/act_pusht_chunk_${chunk}/checkpoints/last/pretrained_model \
    --output_dir=outputs/eval/act_pusht_chunk_${chunk} \
    --job_name=act_pusht_chunk_${chunk} \
    --eval.n_episodes=500 \
    --eval.batch_size=50 \
    --policy.device=cuda \
    --policy.use_amp=false \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01
```

> **注意**：启用 temporal ensemble 时必须同时设置 `n_action_steps=1`，否则会报错。

| 策略         | pc\_success ↑ | avg\_max\_reward ↑ | avg\_sum\_reward ↑ | eval\_s ↓ | eval\_ep\_s ↓ |
| ---------- | ------------- | ------------------ | ------------------ | --------- | ------------- |
| chunk\_4   | 5.0           | 0.434              | 79.45              | 776.06    | 1.55          |
| chunk\_8   | 16.2          | 0.567              | 85.53              | 667.02    | 1.35          |
| chunk\_16  | 10.2          | 0.538              | 64.79              | 676.18    | 1.35          |
| chunk\_32  | 2.8           | 0.436              | 39.12              | 675.51    | 1.35          |
| chunk\_64  | 0.2           | 0.331              | 24.42              | 670.07    | 1.34          |
| chunk\_100 | 0.0           | 0.327              | 26.43              | 384.80    | 0.77          |

| chunk\_size | 无 TE (pc\_success) | 有 TE (pc\_success) | 变化     |
| ----------- | ------------------ | ------------------ | ------ |
| 4           | 9.4%               | 5.0%               | ↓ 4.4  |
| 8           | 22.6%              | 16.2%              | ↓ 6.4  |
| 16          | 21.0%              | 10.2%              | ↓ 10.8 |
| 32          | 10.0%              | 2.8%               | ↓ 7.2  |
| 64          | 2.0%               | 0.2%               | ↓ 1.8  |
| 100         | 0.6%               | 0.0%               | ↓ 0.6  |

**分析**：
- **全面下降**：所有 chunk\_size 下 TE 均导致成功率下降，平均降幅约 5.2 个百分点。chunk\_16 降幅最大 (↓10.8)，chunk\_64 降幅最小 (↓1.8，但基数已极低)。
- **根本原因**：模型按"每 N 步推理一次"训练，TE 强制改为"每步都推理+加权平均"，训练与推理节奏不匹配。加权平均还会模糊关键转折点的动作信号。
- **avg\_sum\_reward 的反常上升**：chunk\_8 的 TE 版本 avg\_sum\_reward (85.53) 反而高于无 TE (77.67)，因为 TE 使动作更平滑、每步奖励更稳定，但无法在关键时刻做出精确调整，成功率反降。
- **结论**：TE 不适用于 PushT，后续实验全部弃用。

### 2.3 探索 chunk/action 分离

chunk/action 分离, 模型预测未来 16 步动作，但只执行前 4 步就重新推理，后 12 步丢弃

```bash
export HF_USER=lepao
lerobot-train \
    --env.type=pusht \
    --policy.type=act \
    --policy.repo_id=${HF_USER}/act_pusht_chunk16_act4 \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/act_pusht_chunk16_act4 \
    --job_name=act_pusht_chunk16_act4 \
    --batch_size=64 \
    --steps=100000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.chunk_size=16 \
    --policy.n_action_steps=4 \
    --policy.n_decoder_layers=1 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true
```

| 策略            | pc\_success ↑ | avg\_max\_reward ↑ | avg\_sum\_reward ↑ | eval\_s ↓ | eval\_ep\_s ↓ |
| ------------- | ------------- | ------------------ | ------------------ | --------- | ------------- |
| 基线 chunk\_8   | 22.6          | 0.611              | 77.67              | 810.25    | 1.62          |
| 基线 chunk8\_TE | 16.2          | 0.567              | 85.53              | 667.02    | 1.35          |
| chunk16\_act4 | 20.4          | 0.533              | 74.07              | 666.63    | 1.33          |

**结论**：chunk16_act4 (20.4%) 优于 TE (16.2%) 但不及基线 chunk8 (22.6%)。分离策略通过增加推理频率获得部分增益，但仍存在训练-推理分布偏移问题，不如保持 chunk=n_action 的完全一致性。

### 2.4 调整Decoder层数

不同 decoder 层数影响模型的表示能力和泛化能力。decoder 层数越多，模型越复杂，泛化能力越强。探索的 decoder 层数值：**4, 7**。

```bash
export HF_USER=lepao
export JOB_NAME=act_pusht_dec7
lerobot-train \
    --env.type=pusht \
    --policy.type=act \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --batch_size=64 \
    --steps=100000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.chunk_size=8 \
    --policy.n_action_steps=8 \
    --policy.n_decoder_layers=${JOB_NAME: -1} \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true
```

| 策略                      | pc\_success ↑ | avg\_max\_reward ↑ | avg\_sum\_reward ↑ | eval\_s ↓ | eval\_ep\_s ↓ |
| ----------------------- | ------------- | ------------------ | ------------------ | --------- | ------------- |
| 基线 n\_decoder\_layers=1 | 22.6          | 0.611              | 77.67              | 810.25    | 1.62          |
| n\_decoder\_layers=4    | 20.2          | 0.620              | 82.33              | 504.89    | 1.01          |
| n\_decoder\_layers=7    | 28.4          | 0.649              | 79.53              | 510.79    | 1.02          |

**结论**：更深 decoder (7层) 显著优于浅层 (1层)，从 22.6% → 28.4% (+25%)。4层反而下降 (20.2%)，可能处于欠拟合与过拟合的临界点。7层 decoder 提供了足够的表示能力来建模 PushT 的复杂动作序列，同时 CVAE latent 提供正则化防止过拟合。

### 2.5 学习率与KL调整

lr 调整：1e-5, 2e-5, 5e-5

```bash
export HF_USER=lepao
export JOB_NAME=act_pusht_dec7_lr5e5
lerobot-train \
    --env.type=pusht \
    --policy.type=act \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --batch_size=64 \
    --steps=100000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.chunk_size=8 \
    --policy.n_action_steps=8 \
    --policy.n_decoder_layers=7 \
    --policy.optimizer_lr=2e-5 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true
```

kl 调整：kl10, kl5, kl1

```bash
export HF_USER=lepao
export JOB_NAME=act_pusht_dec7_kl1
lerobot-train \
    --env.type=pusht \
    --policy.type=act \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --batch_size=64 \
    --steps=100000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.chunk_size=8 \
    --policy.n_action_steps=8 \
    --policy.n_decoder_layers=7 \
    --policy.kl_weight=1.0 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true
```

融合已探索的增益训练配置

```bash
export HF_USER=lepao
export JOB_NAME=act_pusht_dec7_lr2e5_chunk16act4
lerobot-train \
    --env.type=pusht \
    --policy.type=act \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --batch_size=64 \
    --steps=100000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.chunk_size=16 \
    --policy.n_action_steps=4 \
    --policy.n_decoder_layers=7 \
    --policy.optimizer_lr=2e-5 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true
```

| 策略                      | pc\_success ↑ | avg\_max\_reward ↑ | avg\_sum\_reward ↑ | eval\_s ↓ | eval\_ep\_s ↓ |
| ----------------------- | ------------- | ------------------ | ------------------ | --------- | ------------- |
| 基线：dec7\_kl10\_lr1e5    | 28.4          | 0.649              | 79.53              | 510.79    | 1.02          |
| act\_pusht\_dec7\_lr2e5 | 29.2          | 0.661              | 83.16              | 880.67    | 1.76          |
| act\_pusht\_dec7\_lr5e5 | 28.5          | 0.693              | 91.96              | 878.14    | 1.76          |
| act\_pusht\_dec7\_kl5   | 22.2          | 0.629              | 85.76              | 880.03    | 1.76          |
| act\_pusht\_dec7\_kl1   | 24.6          | 0.634              | 82.75              | 795.79    | 1.59          |
| 正向策略融合                  | 28.0          | 0.595              | 78.41              | 641.39    | 1.28          |

**结论**：
- **学习率**：`lr=2e-5` 略优于默认 1e-5 (29.2% vs 28.4%)，但 `lr=5e-5` 并无进一步增益 (28.5%)，说明 PushT 任务对学习率不敏感，2e-5 是较稳妥的选择。
- **KL 权重**：降低 KL 权重 (kl5=22.2%, kl1=24.6%) 反而比默认 kl10 (28.4%) 更差。原因：ACT 的 CVAE 结构需要足够的 KL 约束来保持 latent space 的结构，过弱的 KL 导致 latent 分布崩塌，采样时质量下降。

### 2.6 数据集增强

探索不同的数据集增强策略
affine: 10, 15, 20；以及默认的6种策略

```bash
export HF_USER=lepao
export JOB_NAME=act_pusht_dec7_lr2e5_chunk16act4_aug-aff
lerobot-train \
    --env.type=pusht \
    --policy.type=act \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --batch_size=64 \
    --steps=100000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.chunk_size=16 \
    --policy.n_action_steps=4 \
    --policy.n_decoder_layers=7 \
    --policy.optimizer_lr=2e-5 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs='{"affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-10, 10], "translate": [0.1, 0.1]}}}'
```

```bash
export HF_USER=lepao
export JOB_NAME=act_pusht_dec7_lr2e5_chunk16act4_aug
lerobot-train \
    ···
    --dataset.image_transforms.enable=true
```

```bash
export HF_USER=lepao
export JOB_NAME=act_pusht_dec7_lr2e5_chunk16act4_aug-aff15
lerobot-train \
    ···
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs='{"affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-15, 15], "translate": [0.15, 0.15]}}}'
```

```bash
export HF_USER=lepao
export JOB_NAME=act_pusht_dec7_lr2e5_chunk16act4_aug-aff20
lerobot-train \
    ···
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs='{"affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-20, 20], "translate": [0.2, 0.2]}}}'
```

探索10万步下 cosine 优化器设置，对训练的影响

```bash
export HF_USER=lepao
export JOB_NAME=act_pusht_dec7_lr2e5_chunk16act4_aug-aff_cosine
lerobot-train \
    --env.type=pusht \
    --policy.type=act \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --batch_size=64 \
    --steps=100000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.chunk_size=16 \
    --policy.n_action_steps=4 \
    --policy.n_decoder_layers=7 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs='{"affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-10, 10], "translate": [0.1, 0.1]}}}' \
    --use_policy_training_preset=false \
    --optimizer.type=adamw \
    --optimizer.lr=2e-5 \
    --optimizer.weight_decay=1e-4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=2000 \
    --scheduler.num_decay_steps=100000 \
    --scheduler.peak_lr=2e-5 \
    --scheduler.decay_lr=2e-6
```

| 策略                                                    | pc\_success ↑ | avg\_max\_reward ↑ | avg\_sum\_reward ↑ | eval\_s ↓ | eval\_ep\_s ↓ |
| ----------------------------------------------------- | ------------- | ------------------ | ------------------ | --------- | ------------- |
| act\_pusht\_dec7\_lr2e5\_chunk16act4\_aug             | 28.0          | 0.595              | 78.41              | 641.39    | 1.28          |
| act\_pusht\_dec7\_lr2e5\_chunk16act4\_aug-aff         | 41.8          | 0.797              | 104.20             | 435.51    | 0.87          |
| act\_pusht\_dec7\_lr2e5\_chunk16act4\_aug-aff15       | 40.6          | 0.816              | 108.96             | 446.23    | 0.89          |
| act\_pusht\_dec7\_lr2e5\_chunk16act4\_aug-aff20       | 21.6          | 0.774              | 127.84             | 449.50    | 0.90          |
| act\_pusht\_dec7\_lr2e5\_chunk16act4\_aug-aff\_cosine | 36.6          | 0.726              | 96.93              | 444.08    | 0.89          |

**分析**：
- **适度增强最优**：affine ±10°/±10% (41.8%) > ±15°/±15% (40.6%) > ±20°/±20% (21.6%)
- **增强过强有害**：aug-aff20 成功率腰斩至 21.6%，原因是过强的变换扭曲了空间关系，导致模型学到错误的几何先验。PushT 任务高度依赖精确的空间位置，过度旋转/平移会破坏末端执行器与目标的相对位置关系。
- **Cosine 调度未增益**：在 100k 步下 cosine 调度 (36.6%) 不如恒定学习率 (41.8%)，说明短训练周期内学习率衰减过早，模型未充分收敛。

### 2.7、延长训练步数

探索 200000 / 300000 / 400000

```bash
export HF_USER=lepao
export STEPS=400000
export STEPS_K=$((STEPS / 1000))k
lerobot-train \
    --env.type=pusht \
    --policy.type=act \
    --policy.repo_id=${HF_USER}/act_pusht_dec7_lr2e5_chunk16act4_aug-aff_${STEPS_K} \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/act_pusht_dec7_lr2e5_chunk16act4_aug-aff_${STEPS_K} \
    --job_name=act_pusht_dec7_lr2e5_chunk16act4_aug-aff_${STEPS_K} \
    --batch_size=64 \
    --steps=${STEPS} \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.chunk_size=16 \
    --policy.n_action_steps=4 \
    --policy.n_decoder_layers=7 \
    --policy.optimizer_lr=2e-5 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs='{"affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-10, 10], "translate": [0.1, 0.1]}}}'
```

| 策略               | pc\_success ↑ | avg\_max\_reward ↑ | avg\_sum\_reward ↑ | eval\_s ↓  | eval\_ep\_s ↓ |
| ---------------- | ------------- | ------------------ | ------------------ | ---------- | ------------- |
| 100k             | 41.8          | 0.797              | 104.20             | 435.51     | 0.87          |
| 200k             | 51.6          | 0.794              | 90.12              | 600.70     | 1.20          |
| 300k             | 45.8          | 0.821              | 101.27             | 601.55     | 1.20          |
| 400k             | 63.0          | 0.857              | 87.99              | 598.54     | 1.20          |

**结论**：
- **训练收益非单调**：200k (51.6%) → 300k (45.8%) 出现回退，可能因恒定学习率导致后期震荡，模型在局部最优附近波动
- **长训练有效**：400k (63.0%) 显著恢复并超越 200k，说明模型需要足够步数才能收敛到更好的解

# 最佳结果

```bash
export HF_USER=lepao
export JOB_NAME=act_pusht_dec7_lr2e5_chunk8act8_aug-aff_500k_cosine
export STEPS=500000
lerobot-train \
    --env.type=pusht \
    --policy.type=act \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --batch_size=64 \
    --steps=${STEPS} \
    --eval_freq=20000 \
    --save_freq=20000 \
    --num_workers=4 \
    --policy.chunk_size=8 \
    --policy.n_action_steps=8 \
    --policy.n_decoder_layers=7 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs='{"affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-10, 10], "translate": [0.1, 0.1]}}}' \
    --use_policy_training_preset=false \
    --optimizer.type=adamw \
    --optimizer.lr=2e-5 \
    --optimizer.weight_decay=1e-4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=2000 \
    --scheduler.num_decay_steps=${STEPS} \
    --scheduler.peak_lr=2e-5 \
    --scheduler.decay_lr=2e-6
```

评估

```bash
lerobot-eval \
    --env.type=pusht \
    --policy.path=outputs/train/act_pusht_dec7_lr2e5_chunk8act8_aug-aff_500k_cosine/checkpoints/last/pretrained_model \
    --output_dir=outputs/eval/act_pusht_dec7_lr2e5_chunk8act8_aug-aff_500k_cosine \
    --job_name=act_pusht_dec7_lr2e5_chunk8act8_aug-aff_500k_cosine \
    --eval.n_episodes=500 \
    --eval.batch_size=50 \
    --policy.device=cuda \
    --policy.use_amp=false
```

| 策略               | pc\_success ↑ | avg\_max\_reward ↑ | avg\_sum\_reward ↑ | eval\_s ↓  | eval\_ep\_s ↓ |
| ---------------- | ------------- | ------------------ | ------------------ | ---------- | ------------- |
| **500k\_cosine** | **66.0**      | **0.922**          | **101.07**         | **437.58** | **0.88**      |