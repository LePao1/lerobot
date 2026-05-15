# Diffusion Policy 实验指南

## 1、概览

- **任务**: SO101 抓取纸方块 ("Grab the paper cube")
- **数据集**: `lepao/so101_test`（181 ep, 80328 frames, 30fps, 6-DOF）
- **硬件**: SO101 follower+leader，双相机（handeye + fixed, 640x360）
- **对比基线**: ACT v0.1（chunk=100, act=100, resnet18）

### 评价指标

| 指标      | 含义           | 标准                    |
| ------- | ------------ | --------------------- |
| 实机成功率   | 抓取成功 / 总尝试次数 | **最终评判标准**            |
| 离线 loss | 训练/验证 loss   | 越低越好，但不能单看            |
| 推理速度    | 单次推理耗时       | 需满足实时控制（<33ms @30fps） |

> Diffusion Policy 没有 ACT 的 delta ratio / MAE / MSE 离线指标（`eval_act.py` 仅支持 ACT）。SO101 无仿真环境（`lerobot-eval` 只支持 aloha/pusht/libero 等标准 benchmark），**离线只能看 training/validation loss，最终必须实机评测**。

### 离线对比

```bash
# 跨模型对比训练 loss 曲线（wandb）
# 或手动对比 checkpoint 的 val loss
python -c "
from lerobot.policies.diffusion import DiffusionPolicy
policy = DiffusionPolicy.from_pretrained('lepao/diffusion_so101_v01_baseline')
# 加载 dataset 计算 val loss
"
```

## 2、踩坑速查

Diffusion Policy 与 ACT 的关键差异，新实验前必看：

1. **`horizon`** **≠** **`n_action_steps`** — horizon 是预测总长度，n\_action\_steps 是实际执行步数。典型配置 horizon=16, n\_action\_steps=8（预测 16 步，执行前 8 步后重规划）
2. **`n_obs_steps`** **默认 2** — Diffusion 支持多步观测历史，ACT 不支持（硬编码为 1）
3. **推理速度是关键瓶颈** — DDPM 默认 100 步去噪，推理慢；可用 DDIM + `num_inference_steps` 加速
4. **`drop_n_last_frames`** 需与 horizon/n\_action\_steps/n\_obs\_steps 匹配，否则 episode 末尾会过度 padding
5. **无 VAE，无 KL weight** — Diffusion 没有模式坍塌问题，但可能 overshoot 或欠拟合
6. **`use_group_norm`** **与预训练权重互斥** — 用 pretrained backbone 时需关闭 group\_norm

## 3、第一轮实验：Baseline 建立

### 目标

用 Diffusion Policy 默认配置训练 baseline，与 ACT v0.1 对比。

### 实验配置

| 参数                     | 值                 | 说明                         |
| ---------------------- | ----------------- | -------------------------- |
| `horizon`              | 16                | 预测 16 步（\~0.53s @30fps）    |
| `n_action_steps`       | 8                 | 执行 8 步后重规划                 |
| `n_obs_steps`          | 2                 | 2 步观测历史                    |
| `vision_backbone`      | resnet18          | 与 ACT v0.1 对齐              |
| `num_train_timesteps`  | 100               | DDPM 默认                    |
| `num_inference_steps`  | 100               | 推理时去噪步数（先不做加速）             |
| `noise_scheduler_type` | DDPM              | 默认                         |
| `down_dims`            | (512, 1024, 2048) | U-Net 默认架构                 |
| `optimizer_lr`         | 1e-4              | 默认                         |
| `batch_size`           | 64                | Diffusion 默认 batch size 更大 |
| `steps`                | 100000            | 与 ACT 对齐                   |

### 训练命令

```bash
export HF_USER=lepao
export JOB_NAME=diffusion_so101_v01_baseline
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=diffusion \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.n_obs_steps=2 \
    --policy.vision_backbone=resnet18 \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=100 \
    --policy.noise_scheduler_type=DDPM \
    --policy.down_dims=[512,1024,2048] \
    --wandb.enable=true
```

### 预期

- 建立 Diffusion Policy 在 SO101 上的首个 baseline
- 对比 ACT v0.1：Diffusion 理论上动作更平滑、多样性更好
- 关注推理速度是否满足实时控制

***

## 4、第二轮实验：Horizon & Action Steps 扫描

### 核心假设

Diffusion Policy 的 `horizon` 和 `n_action_steps` 决定了预测视野与重规划频率。**更长的 horizon 提供更多上下文但推理更慢，更短的 horizon 响应更快但可能丢失长期规划能力**。

### 实验矩阵

| 实验              | horizon | n\_action\_steps | 预测时长  | 执行时长  | 重规划间隔 |
| --------------- | ------- | ---------------- | ----- | ----- | ----- |
| **v1 baseline** | 16      | 8                | 0.53s | 0.27s | 8 步   |
| **exp2\_h32**   | 32      | 16               | 1.07s | 0.53s | 16 步  |
| **exp3\_h64**   | 64      | 32               | 2.13s | 1.07s | 32 步  |
| **exp4\_h100**  | 100     | 50               | 3.33s | 1.67s | 50 步  |

**设计原则**:

1. `n_action_steps = horizon / 2`：执行一半后重规划，平衡推理频率与规划视野
2. 其他参数固定为 baseline 默认值
3. 统一训练到 100k step

### 训练命令

```bash
# exp2: horizon=32
export HF_USER=lepao
export JOB_NAME=diffusion_so101_v02_exp2_h32
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=diffusion \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.horizon=32 \
    --policy.n_action_steps=16 \
    --policy.n_obs_steps=2 \
    --policy.vision_backbone=resnet18 \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=100 \
    --policy.noise_scheduler_type=DDPM \
    --wandb.enable=true
```

```bash
# exp3: horizon=64
export HF_USER=lepao
export JOB_NAME=diffusion_so101_v02_exp3_h64
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=diffusion \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.horizon=64 \
    --policy.n_action_steps=32 \
    --policy.n_obs_steps=2 \
    --policy.vision_backbone=resnet18 \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=100 \
    --policy.noise_scheduler_type=DDPM \
    --wandb.enable=true
```

```bash
# exp4: horizon=100（与 ACT v0.1 chunk 对齐）
export HF_USER=lepao
export JOB_NAME=diffusion_so101_v02_exp4_h100
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=diffusion \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.horizon=100 \
    --policy.n_action_steps=50 \
    --policy.n_obs_steps=2 \
    --policy.vision_backbone=resnet18 \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=100 \
    --policy.noise_scheduler_type=DDPM \
    --wandb.enable=true
```

***

## 5、第三轮实验：推理加速（DDIM + 减少去噪步数）

### 核心假设

DDPM 默认 100 步去噪推理太慢，无法满足实时控制。**换 DDIM scheduler + 减少** **`num_inference_steps`** **可大幅加速推理，且精度损失可控**。

### 实验矩阵

基于第二轮最优 horizon 配置，扫推理步数：

| 实验                | scheduler | num\_inference\_steps | 预期推理速度    |
| ----------------- | --------- | --------------------- | --------- |
| **exp5\_ddim100** | DDIM      | 100                   | 与 DDPM 持平 |
| **exp6\_ddim50**  | DDIM      | 50                    | \~2x 加速   |
| **exp7\_ddim25**  | DDIM      | 25                    | \~4x 加速   |
| **exp8\_ddim10**  | DDIM      | 10                    | \~10x 加速  |

**设计原则**:

1. 用 DDIM scheduler 训练（`noise_scheduler_type=DDIM`），推理时减少步数
2. DDIM 与 DDPM 共享训练目标，可训练完再调推理步数（无需重新训练）
3. 找到满足实时控制的最少推理步数

### 训练命令

```bash
# exp5-8 共用同一训练（DDIM scheduler），推理时改 num_inference_steps
export HF_USER=lepao
export JOB_NAME=diffusion_so101_v03_ddim
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=diffusion \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.n_obs_steps=2 \
    --policy.vision_backbone=resnet18 \
    --policy.num_train_timesteps=100 \
    --policy.noise_scheduler_type=DDIM \
    --wandb.enable=true
```

### 推理时切换步数

DDIM 与 DDPM 共享训练目标，训练完成后**无需重新训练**，推理时直接改 `num_inference_steps` 即可。实机部署时在 policy config 中设置该参数。

***

## 6、第四轮实验：Visual Backbone 升级

### 核心假设

与 ACT 第四轮同理：resnet18 特征提取能力有限，换更大 backbone 提升空间感知 → 抓取精度提升。

### 实验矩阵

| 实验             | backbone | 参数量 | 特征维度                 |
| -------------- | -------- | --- | -------------------- |
| v1 baseline    | resnet18 | 11M | 64 (spatial softmax) |
| **exp9\_r34**  | resnet34 | 21M | 64                   |
| **exp10\_r50** | resnet50 | 25M | 64                   |

**设计原则**:

1. 固定 baseline 的其他参数，backbone 为唯一变量
2. 使用 ImageNet 预训练权重
3. 关闭 `use_group_norm`（与预训练权重互斥）

### 训练命令

```bash
# exp9: resnet34
export HF_USER=lepao
export JOB_NAME=diffusion_so101_v04_exp9_resnet34
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=diffusion \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=64 \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.n_obs_steps=2 \
    --policy.vision_backbone=resnet34 \
    --policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1 \
    --policy.use_group_norm=false \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=100 \
    --policy.noise_scheduler_type=DDPM \
    --wandb.enable=true
```

```bash
# exp10: resnet50
export HF_USER=lepao
export JOB_NAME=diffusion_so101_v04_exp10_resnet50
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=diffusion \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=64 \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.n_obs_steps=2 \
    --policy.vision_backbone=resnet50 \
    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 \
    --policy.use_group_norm=false \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=100 \
    --policy.noise_scheduler_type=DDPM \
    --wandb.enable=true
```

***

## 7、第五轮实验：U-Net 架构调优

### 核心假设

默认 `down_dims=(512, 1024, 2048)` 是为 PushT 设计的轻量 U-Net。SO101 任务更复杂（双相机 + 6-DOF），可能需要更大的 U-Net。

### 实验矩阵

| 实验          | down\_dims              | 参数量（U-Net） |
| ----------- | ----------------------- | ---------- |
| v1 baseline | (512, 1024, 2048)       | \~30M      |
| **exp11**   | (256, 512, 1024)        | 更轻量        |
| **exp12**   | (512, 1024, 2048, 2048) | 更深（4 层下采样） |

### 训练命令

```bash
# exp11: 更轻量 U-Net
export HF_USER=lepao
export JOB_NAME=diffusion_so101_v05_exp11_unet_small
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=diffusion \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=64 \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.n_obs_steps=2 \
    --policy.vision_backbone=resnet18 \
    --policy.down_dims=[256,512,1024] \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=100 \
    --policy.noise_scheduler_type=DDPM \
    --wandb.enable=true
```

```bash
# exp12: 更深 U-Net
export HF_USER=lepao
export JOB_NAME=diffusion_so101_v05_exp12_unet_deep
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=diffusion \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=64 \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.n_obs_steps=2 \
    --policy.vision_backbone=resnet18 \
    --policy.down_dims=[512,1024,2048,2048] \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=100 \
    --policy.noise_scheduler_type=DDPM \
    --wandb.enable=true
```

***

## 8、第六轮实验：数据增强

### 核心假设

181 ep 数据量较小，Diffusion Policy 参数量大（U-Net + Transformer decoder），数据增强可能缓解过拟合。

### 实验矩阵

| 实验             | 增强方案                       | 说明            |
| -------------- | -------------------------- | ------------- |
| v1 baseline    | 无                          | -             |
| **exp13\_aug** | ColorJitter + RandomAffine | 与 ACT exp4 对齐 |

### 训练命令

```bash
# exp13: 数据增强
export HF_USER=lepao
export JOB_NAME=diffusion_so101_v06_exp13_aug
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=diffusion \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=64 \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.n_obs_steps=2 \
    --policy.vision_backbone=resnet18 \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=100 \
    --policy.noise_scheduler_type=DDPM \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs='{"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-10,10],"translate":[0.1,0.1]}},"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}}}' \
    --wandb.enable=true
```

***

## 9、实机评测协议

固定流程，与 ACT 实验对齐，便于跨模型公平对比：

- **5 个方块位置**: 中央、左前、右前、左后、右后
- 每位置 5 次，共 **25 次/模型**
- 主指标: 成功率；辅助: 平均耗时 + 抓偏距离(cm)

### 实机部署注意事项

1. Diffusion Policy 推理需在 GPU 上运行，确保推理延迟 < 控制周期
2. 如果 DDPM 100 步推理太慢，优先用 DDIM + 减少 `num_inference_steps`
3. 推理时 `n_action_steps` 必须与训练一致

***

## 10、实验路线图

```
第一轮: Baseline (horizon=16, DDPM, resnet18)
  │
  ├─► 第二轮: Horizon 扫描 (16/32/64/100)
  │     │
  │     └─► 选最优 horizon
  │           │
  │           ├─► 第三轮: 推理加速 (DDIM + 减少去噪步数)
  │           │
  │           ├─► 第四轮: Backbone 升级 (resnet34/50)
  │           │
  │           ├─► 第五轮: U-Net 架构调优
  │           │
  │           └─► 第六轮: 数据增强
  │
  └─► 最终: 最优配置 vs ACT v0.1 实机对比
```

### 成功标准

| 结果                         | 下一步                                  |
| -------------------------- | ------------------------------------ |
| Diffusion 实机成功率 > ACT v0.1 | 在最优配置上做消融实验，确认关键因子                   |
| Diffusion ≈ ACT            | 尝试 Diffusion + ACT 的互补策略（如 ensemble） |
| Diffusion < ACT            | 确认 Diffusion 是否适合 SO101 任务，考虑换其他方法   |

