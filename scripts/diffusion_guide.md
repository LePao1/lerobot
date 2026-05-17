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

## 4、下一次实验：长动作窗口 + DDIM 加速

### 结论更新

`diffusion_so101_v01_baseline` 已验证可以远程收到动作，但实机几乎不动。原因是 baseline 只返回 8 步动作：

```text
8 / 30fps = 0.27s 动作缓存
DDPM 100 步推理约 0.45-0.50s
```

推理时间大于动作缓存时长，客户端动作队列持续断粮。因此不再继续训练 `horizon=16, n_action_steps=8` 的 backbone/U-Net/augmentation 变体。

### 新目标

训练一个适合远程实机的 Diffusion：

- 长动作窗口，至少覆盖 1.5s 以上
- 使用 DDIM scheduler，后续推理时可减少去噪步数加速
- 先不扫 backbone/U-Net/augmentation，避免无效实验分叉

### 推荐配置

| 实验 | horizon | n_obs_steps | n_action_steps | 动作缓存(@30fps) | scheduler | 说明 |
| ---- | ------- | ----------- | -------------- | ---------------- | --------- | ---- |
| **exp2_h96_ddim** | 96 | 2 | 48 | 1.60s | DDIM | 优先训练，兼顾长窗口与重规划 |

> Diffusion 约束：`n_action_steps <= horizon - n_obs_steps + 1`。同时 U-Net 要求 `horizon` 能被 `2 ** len(down_dims)` 整除。默认 `down_dims=[512,1024,2048]` 时下采样因子为 8，因此用 `horizon=96` 而不是 100。当前先用 `n_action_steps=48`，避免过长开环导致反应慢。

### 训练命令

```bash
export HF_USER=lepao
export JOB_NAME=diffusion_so101_v02_h96_ddim
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
    --policy.horizon=96 \
    --policy.n_action_steps=48 \
    --policy.n_obs_steps=2 \
    --policy.vision_backbone=resnet18 \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=25 \
    --policy.noise_scheduler_type=DDIM \
    --policy.down_dims=[512,1024,2048] \
    --wandb.enable=true
```

### 推理命令要点

```text
--policy_type=diffusion
--pretrained_name_or_path=lepao/diffusion_so101_v02_h96_ddim
--actions_per_chunk=48
--chunk_size_threshold=0.8
--aggregate_fn_name=latest_only
--observation_image_compression_quality=80
```

### 暂停的实验分支

以下实验先不做，除非 `horizon=100 + DDIM` 实机有效但精度不足：

- horizon=32/64 扫描
- diffusion resnet34/resnet50 backbone 扫描
- U-Net 变大/变小
- 数据增强

***

## 5、实机评测协议

固定流程，与 ACT 实验对齐，便于跨模型公平对比：

- **5 个方块位置**: 中央、左前、右前、左后、右后
- 每位置 5 次，共 **25 次/模型**
- 主指标: 成功率；辅助: 平均耗时 + 抓偏距离(cm)

### 实机部署注意事项

1. Diffusion Policy 推理需在 GPU 上运行，确保推理延迟 < 控制周期
2. 如果 DDPM 100 步推理太慢，优先用 DDIM + 减少 `num_inference_steps`
3. 推理时 `n_action_steps` 必须与训练一致

### Windows 实机推理命令

baseline (`horizon=16`, `n_action_steps=8`)：

```powershell
python -m lerobot.async_inference.robot_client `
    --server_address=127.0.0.1:6006 `
    --robot.type=so101_follower `
    --robot.port=COM3 `
    --robot.id=0 `
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 1, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}}" `
    --task="Grab the paper cube" `
    --policy_type=diffusion `
    --pretrained_name_or_path=lepao/diffusion_so101_v01_baseline `
    --policy_device=cuda `
    --client_device=cpu `
    --actions_per_chunk=8 `
    --chunk_size_threshold=0.8 `
    --aggregate_fn_name=latest_only `
    --debug_visualize_queue_size=true `
    --observation_image_compression_quality=80
```

checkpoint 与 `actions_per_chunk` 对应关系：

| checkpoint | 训练配置 | 推理 `actions_per_chunk` |
| ---------- | -------- | ------------------------ |
| `lepao/diffusion_so101_v01_baseline` | `horizon=16`, `n_action_steps=8` | 8 |
| `lepao/diffusion_so101_v02_exp2_h32` | `horizon=32`, `n_action_steps=16` | 16 |
| `lepao/diffusion_so101_v02_exp3_h64` | `horizon=64`, `n_action_steps=32` | 32 |
| `lepao/diffusion_so101_v02_exp4_h100` | `horizon=100`, `n_action_steps=50` | 50 |
| `lepao/diffusion_so101_v03_ddim` | `horizon=16`, `n_action_steps=8` | 8 |
| `lepao/diffusion_so101_v04_exp9_resnet34` | `horizon=16`, `n_action_steps=8` | 8 |
| `lepao/diffusion_so101_v04_exp10_resnet50` | `horizon=16`, `n_action_steps=8` | 8 |

> 远程 SSH 推理时保留 `observation_image_compression_quality=80`，避免双相机 RGB observation 传输成为瓶颈。

***

## 6、实验路线图

```
第一轮: Baseline (horizon=16, n_action_steps=8, DDPM)
  │
  ├─► 实机结果: 能收到动作，但 8 步缓存太短，远程推理持续断粮
  │
  └─► 第二轮: horizon=96, n_action_steps=48, DDIM
        │
        ├─► 若能连续运动: 对比 ACT exp9/resnet50 实机成功率
        │
        ├─► 若推理仍慢: 降低 num_inference_steps
        │
        └─► 若能动但精度差: 再考虑 backbone / U-Net / 数据增强
```

### 成功标准

| 结果                         | 下一步                                  |
| -------------------------- | ------------------------------------ |
| Diffusion 实机成功率 > ACT v0.1 | 在最优配置上做消融实验，确认关键因子                   |
| Diffusion ≈ ACT            | 尝试 Diffusion + ACT 的互补策略（如 ensemble） |
| Diffusion < ACT            | 确认 Diffusion 是否适合 SO101 任务，考虑换其他方法   |
