# ACT 策略实验指南

## 1、概览

- **任务**: SO101 抓取纸方块 ("Grab the paper cube")
- **数据集**: `lepao/so101_test`（181 ep, 80328 frames, 30fps, 6-DOF）
- **硬件**: SO101 follower+leader，双相机（handeye + fixed, 640x360）

### 评价指标

| 指标          | 含义           | 标准                         |
| ----------- | ------------ | -------------------------- |
| Delta ratio | 预测 Δ / GT Δ  | ≈100% 最佳，<10% 说明不动（VAE 坍塌） |
| MAE / MSE   | 角度误差         | 越低越好，**必须配合 delta ratio**  |
| 实机成功率       | 抓取成功 / 总尝试次数 | **最终评判标准**                 |

> MAE/MSE 低不代表效果好。chunk16 MAE 最低但 delta ratio 仅 1.9%，机械臂不动。

### 离线评估

```bash
python scripts/eval_act.py --model v0.1                     # 单模型
python scripts/eval_act.py --model v0.1 --eval_steps 20     # 模拟前 N 步
python scripts/eval_act.py --model v0.1 exp5 exp6           # 多模型对比
python scripts/eval_act.py --model v0.1 --diagnose          # 单帧逐维诊断
```

模型快捷名: `v0.1` / `chunk16` / `exp2_kl5` / `exp3_dec7` / `exp4_aug` / `exp5` / `exp6` / `exp7` / `exp8` / `exp9` / `exp10` / `exp11` / `exp12` / `exp13`
结果输出: `outputs/eval_results.json`

## 2、踩坑速查 ⚠️

来自第二轮实验 + 实机调试的核心教训，新实验前必看：

1. **`weighted_average`** **在 act\<chunk 时会抵消动作** → 实机不动。统一用 `latest_only`
2. **seetacloud 的 HTTPS URL 不能当** **`--server_address`** → gRPC 需要 HTTP/2 端到端，必须 SSH 端口转发到 `127.0.0.1:6006`
3. **离线 delta ratio 高 ≠ 实机会动** → `chunk=100 + act=20` 离线指标最优但实机不动，必须实机评测
4. **训练-推理 chunk/act 必须一致** → 训 chunk=100 但推理 act=20 是 v0.1 实机精度差的根因
5. **chunk < 16 → VAE 模式坍塌** → 模型输出"安全平均值"，MAE 反而最低但完全不动
6. **ACT 不支持** **`n_obs_steps > 1`**（代码硬编码限制）
7. **数据集默认本地优先** → Hub 更新后训练仍可能读旧缓存；需要加 `--dataset.force_cache_sync=true` 强制同步

### 数据集同步

LeRobot 默认使用 `$HF_LEROBOT_HOME/{repo_id}` 本地缓存；只要本地 `meta/`、`data/` 可读，就不会主动检查 HuggingFace Hub 是否有新提交。

如果 Hub 上的 `lepao/so101_test` 更新了，新训练建议加：

```bash
--dataset.revision=main \
--dataset.force_cache_sync=true
```

其中 `--dataset.revision=main` 指向 Hub 最新 main 分支，`--dataset.force_cache_sync=true` 会跳过本地优先加载并重新调用 `snapshot_download()` 同步文件。

## 3、历史实验

### 第一轮：chunk\_size 探索

**目标**: 找到合适的 chunk\_size，建立 baseline。

| 实验               | chunk | act | kl | dec | aug | MSE(100) | MAE(100) | Δratio     | 实机     |
| ---------------- | ----- | --- | -- | --- | --- | -------- | -------- | ---------- | ------ |
| **v0.1** ⭐       | 100   | 100 | 10 | 1   | ✗   | 33.08    | 2.08     | 106%       | 能动但精度差 |
| chunk16          | 16    | 16  | 10 | 1   | ✗   | **5.71** | **1.24** | **1.9%** ❌ | 不动     |
| chunk16+dec7+aug | 16    | 16  | 10 | 7   | ✓   | 最差       | 最差       | 极低 ❌       | 最差     |

**结论**:

- `chunk=16` → VAE 模式坍塌（MAE 反而最低但完全不动，**MAE/MSE 单看会误导**）
- chunk=100 是当前 baseline
- **关键洞察**: v0.1 前 20 步预测较准（MSE=7.77），后期累积误差大（MSE=33.08）

**训练命令**:

```bash
# v0.1 基线
export HF_USER=lepao
export JOB_NAME=act_so101_v01_chunk100
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --wandb.enable=true
```

```bash
# chunk16
export HF_USER=lepao
export JOB_NAME=act_so101_v01_chunk16
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=16 \
    --policy.n_action_steps=16 \
    --policy.kl_weight=10.0 \
    --wandb.enable=true
```

```bash
# chunk16+dec7+aug
export HF_USER=lepao
export JOB_NAME=act_so101_v01_chunk16_dec7_aug
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=16 \
    --policy.n_action_steps=16 \
    --policy.kl_weight=10.0 \
    --policy.n_decoder_layers=7 \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs='{"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-10,10],"translate":[0.1,0.1]}}}' \
    --wandb.enable=true
```

→ **第二轮方向**: 在 v0.1 + act=20 配置下做单变量超参微调

### 第二轮：v0.1 基础上的超参微调

**目标**: 基于第一轮"前 20 步较准"的洞察，固定 chunk=100 + act=20，单变量试 kl/dec/aug。

| 实验                  | chunk | act | kl    | dec   | aug   | MSE(20)  | MAE(20)  | Δratio | 实机   |
| ------------------- | ----- | --- | ----- | ----- | ----- | -------- | -------- | ------ | ---- |
| **v0.1 (act=20)** ⭐ | 100   | 20  | 10    | 1     | ✗     | **7.77** | **1.48** | 106%   | 精度差  |
| exp2                | 100   | 20  | **5** | 1     | ✗     | 8.11     | 1.56     | 109%   | 精度差  |
| exp3                | 100   | 20  | 10    | **7** | ✗     | 10.02    | 1.56     | 114%   | 几乎不动 |
| exp4                | 100   | 20  | 10    | 1     | **✓** | 15.20    | 1.79     | 105%   | 几乎不动 |

**结论**:

- `kl=5` / `decoder=7` / `data_aug` 均**未超过 v0.1**
- decoder=7 在 teacher-forced val 上略优，但 rollout 视角下不如 v0.1
- 数据增强在 181 ep 上显著退化（MSE 翻倍）
- **不再调这三项超参**

**训练命令**:

```bash
# exp2: kl_weight=5
export HF_USER=lepao
export JOB_NAME=act_so101_v02_exp2_chunk100_act20_kl5
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=20 \
    --policy.kl_weight=5.0 \
    --wandb.enable=true
```

```bash
# exp3: n_decoder_layers=7
export HF_USER=lepao
export JOB_NAME=act_so101_v02_exp3_chunk100_act20_dec7
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=20 \
    --policy.kl_weight=10.0 \
    --policy.n_decoder_layers=7 \
    --wandb.enable=true
```

```bash
# exp4: 数据增强（仅 affine，参考 PushT 最优方案）
export HF_USER=lepao
export JOB_NAME=act_so101_v02_exp4_chunk100_act20_aug
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=20 \
    --policy.kl_weight=10.0 \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs='{"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-10,10],"translate":[0.1,0.1]}}}' \
    --wandb.enable=true
```

→ **实机部署验证**

### 实机部署调试（v0.1）

部署 v0.1 实机，扫描推理参数：

| 配置                                         | 实机表现                |
| ------------------------------------------ | ------------------- |
| act=20 + weighted\_average + threshold=0.1 | 能动，精度差              |
| act=100 + latest\_only + threshold=0.3     | 能动，延迟大              |
| act=100 + latest\_only + threshold=0.6     | 能动，**精度差**          |
| act=100 + latest\_only + threshold=0.7     | 同上，调 threshold 已无收益 |

**失败模式**: B/C 阶段 —— 粗定位方向对，但接近后**抓空 / 抓偏**。

**核心发现**:

- `weighted_average` 在 act\<chunk 时会抵消重叠区动作 → 实机不动
- **训练-推理不一致**是 v0.1 精度差的根因：训 chunk=100 但实机用 act=20，模型容量浪费在永远不会执行的末端 60 步上
- 调推理参数（threshold、aggregate）已无空间，瓶颈在模型本身

→ **第三轮方向**: 训练-推理对齐（chunk == act），扫 chunk 长度找甜区

## 4、第三轮实验：chunk 长度扫描

### 核心假设

v0.1 的 chunk=100 让模型容量浪费在永远不会执行的末端 60 步上，前 30–40 步反而预测不够准。**chunk 缩短到接近实际执行窗口 → 容量集中 → 末端精度提升**。

### 实验矩阵

| 实验        | chunk | act | 时长(@30fps) | threshold | 重规划间隔   |
| --------- | ----- | --- | ---------- | --------- | ------- |
| **exp5**  | 40    | 40  | 1.33s      | 0.25      | 30 步/1s |
| **exp6**  | 60    | 60  | 2.0s       | 0.5       | 30 步/1s |
| **exp7**  | 80    | 80  | 2.67s      | 0.625     | 30 步/1s |
| v0.1 对照   | 100   | 100 | 3.33s      | 0.7       | 30 步/1s |

**设计原则**:

1. `chunk == act`：训推一致，模型容量不浪费
2. 重规划间隔统一 1s：让 chunk 长度成为**唯一变量**
3. 形成 40/60/80/100 四点 scaling 曲线

**预期**: chunk=60 最优（2s 覆盖一次接近+抓取）。

### 训练一致性

**所有模型统一训练到 100k step**，不做中途提前停训，保证跨模型对比的严谨性（避免引入"检查点选择"这个混淆变量）。100k 完成后再统一 eval delta ratio。

### 训练命令

```bash
# exp5: chunk=40
export HF_USER=lepao
export JOB_NAME=act_so101_v03_exp5_chunk40
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=40 \
    --policy.n_action_steps=40 \
    --policy.kl_weight=10.0 \
    --wandb.enable=true
```

```bash
# exp6: chunk=60
export HF_USER=lepao
export JOB_NAME=act_so101_v03_exp6_chunk60
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=60 \
    --policy.n_action_steps=60 \
    --policy.kl_weight=10.0 \
    --wandb.enable=true
```

```bash
# exp7: chunk=80
export HF_USER=lepao
export JOB_NAME=act_so101_v03_exp7_chunk80
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=80 \
    --policy.n_action_steps=80 \
    --policy.kl_weight=10.0 \
    --wandb.enable=true
```

### 离线评估命令

```bash
python scripts/eval_act.py --model exp5 --eval_steps 40 --mode rollout
python scripts/eval_act.py --model exp6 --eval_steps 60 --mode rollout
python scripts/eval_act.py --model exp7 --eval_steps 80 --mode rollout
python scripts/eval_act.py --model v0.1 --eval_steps 100 --mode rollout
```

> 主指标用 **MSE/MAE per step**，不是累积总和（长 chunk 天然吃亏）。

### 离线评估结果

| 实验       | chunk | eval | MSE      | MAE      | Δratio | 趋势                      |
| -------- | ----- | ---- | -------- | -------- | ------ | ----------------------- |
| **exp5** | 40    | 40   | **9.30** | **1.15** | 104.9% | MSE/MAE 最优，但末端 delta 骤降 |
| exp6     | 60    | 60   | 33.91    | 1.79     | 118.9% | 比 v0.1 差，后期 overshoot   |
| exp7     | 80    | 80   | 45.82    | 2.09     | 114.1% | 四项中最差                   |
| v0.1     | 100   | 100  | 29.20    | 1.74     | 112.2% | 最稳定，全程 delta 无骤降        |

**per-step 关键发现**:

- **exp5**: step 0–29 delta ratio 正常（93–135%），step 30 后持续下降，step 38 仅 54.5% → 末端"减速/冻结"，VAE 坍塌风险
- **exp6**: step 20 后 MSE 急剧恶化（12.6→103.6），delta ratio 偏高（118.9%）→ overshoot
- **exp7**: 全程 MSE 最高，step 44 后 delta ratio 波动大（73–131%）→ 不稳定
- **v0.1**: 全程 delta ratio 在 83–160% 之间波动，无末端骤降 → 最稳定

### 结论

1. **chunk 缩短未带来一致的精度提升** — 仅 chunk=40 的 MSE/MAE 优于 v0.1，但末端 delta ratio 骤降表明存在 VAE 坍塌风险
2. **chunk=60/80 均比 v0.1 差**，模型容量不足以在更短窗口内学好完整轨迹
3. **v0.1 (chunk=100) 仍是离线最稳定的模型**
4. **假设被推翻**: 缩短 chunk 并未让容量集中到前段，反而因窗口变短导致模型无法学到完整轨迹结构

### 实机评测协议（标准化）

固定流程，便于跨模型公平对比：

- **5 个方块位置**: 中央、左前、右前、左后、右后
- 每位置 5 次，共 **25 次/模型**
- 主指标: 成功率；辅助: 平均耗时 + 抓偏距离(cm)

### 成功标准 → 第四轮方向

| 结果                | 第四轮                                   |
| ----------------- | ------------------------------------- |
| 某 chunk 实机 > v0.1 | 在该 chunk 上加数据 / 换 visual backbone     |
| 三个 chunk 都不显著好    | ACT 到天花板，换 Diffusion Policy / SmolVLA |
| chunk=40 VAE 坍塌   | 短 chunk 下限确认 ≥60                      |

→ **第三轮结论**: 三个 chunk 离线均不显著好于 v0.1，但离线不等同实机。**第四轮先换 visual backbone（resnet34/50）**，在 v0.1 的 chunk=100 配置上单变量扫。

## 5、第四轮实验：visual backbone 升级

### 核心假设

v0.1 使用 resnet18（11M 参数），视觉特征提取能力有限。**换更大的 backbone → 更强的空间感知 → 抓取精度提升**。

### 实验矩阵

| 实验       | chunk | act | backbone | 参数量 | 特征维度 |
| -------- | ----- | --- | -------- | --- | ---- |
| v0.1 对照  | 100   | 100 | resnet18 | 11M | 512  |
| **exp8** | 100   | 100 | resnet34 | 21M | 512  |
| **exp9** | 100   | 100 | resnet50 | 25M | 2048 |

**设计原则**:

1. 固定 v0.1 的 chunk=100 配置，**backbone 为唯一变量**
2. 仅试 resnet34/50，resnet101/152 在 181ep 上过拟合风险高
3. 统一训练到 100k step

### 训练命令

```bash
# exp8: resnet34
export HF_USER=lepao
export JOB_NAME=act_so101_v04_exp8_resnet34
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --policy.vision_backbone=resnet34 \
    --policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1 \
    --wandb.enable=true
```

```bash
# exp9: resnet50
export HF_USER=lepao
export JOB_NAME=act_so101_v04_exp9_resnet50
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --policy.vision_backbone=resnet50 \
    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 \
    --wandb.enable=true
```

### 离线评估

```bash
python scripts/eval_act.py --model exp8 --eval_steps 100 --mode rollout
python scripts/eval_act.py --model exp9 --eval_steps 100 --mode rollout
```

### 下一次实验：resnet101 与 resnet50 长训

**实机观察**: `act_so101_v04_exp9_resnet50` 比其他 ACT checkpoint 效果更好，说明当前任务可能仍受视觉特征能力限制。虽然 181 ep 数据量下 resnet101/152 有过拟合风险，但值得按从小到大的顺序做一次受控实验。

**建议顺序**:

1. **exp10: resnet101** — 优先实验，容量大于 resnet50，但训练/推理成本仍可控
2. **exp11: resnet50 长训 + cosine 学习率调度** — 不做 resnet152，先验证 exp9 是否只是训练步数不足

**保持不变**:

- `chunk_size=100`
- `n_action_steps=100`
- `kl_weight=10.0`
- `batch_size=8`（若爆显存，降到 4，并在结果表中标注）

```bash
# exp10: resnet101
export HF_USER=lepao
export JOB_NAME=act_so101_v04_exp10_resnet101
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=20000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --policy.vision_backbone=resnet101 \
    --policy.pretrained_backbone_weights=ResNet101_Weights.IMAGENET1K_V1 \
    --wandb.enable=true
```

```bash
# exp11: resnet50 长训 + cosine 学习率调度
export HF_USER=lepao
export JOB_NAME=act_so101_v04_exp11_resnet50_300k_cosine
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=50000 \
    --steps=300000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --policy.vision_backbone=resnet50 \
    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 \
    --wandb.enable=true \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=5000 \
    --scheduler.num_decay_steps=300000 \
    --scheduler.peak_lr=1e-5 \
    --scheduler.decay_lr=1e-6
```

**exp11 设计意图**:

- 与 exp9 只改变训练策略：`100k constant lr` → `300k cosine warmup/decay`
- 保持 backbone 仍为 resnet50，避免同时改变模型容量和训练长度
- 如果 exp11 优于 exp9，说明 ACT 仍受训练充分性影响；如果无提升，再停止长训方向
- 暂不做 resnet152，避免在小数据集上过拟合并增加远程推理延迟

**推理建议**:

```bash
--actions_per_chunk=100 \
--chunk_size_threshold=0.8 \
--aggregate_fn_name=latest_only \
--observation_image_compression_quality=80
```

> 远程 SSH 推理时不要降低 `actions_per_chunk`，否则动作缓存变短，网络抖动更容易导致机械臂停顿。

### exp10 / exp11 离线评估结果

评估命令：

```bash
python scripts/eval_act.py --model exp10 exp11 --eval_steps 100 --mode rollout --output outputs/eval_exp10_exp11_rollout100.json
```

| 实验        | backbone | 训练策略           | eval | MSE      | MAE      | Δratio | 趋势                         |
| --------- | -------- | -------------- | ---- | -------- | -------- | ------ | -------------------------- |
| exp9 对照   | resnet50 | 100k constant  | 100  | 36.12    | 1.96     | 116.4% | 实机观察优于其他 ACT checkpoint |
| **exp10** | resnet101 | 100k constant | 100  | 85.64    | 2.97     | 131.8% | 明显退化，后半段 overshoot 严重     |
| **exp11** | resnet50 | 300k cosine    | 100  | **6.10** | **0.90** | 105.4% | 显著最优，delta 接近 GT，长训有效     |

**per-step 关键发现**:

- **exp10**: step 20 后误差快速累积，step 44 MSE 已到 128.8，step 99 达 388.4；整体 delta ratio 131.8%，属于大容量 backbone 在小数据集上不稳定 / overshoot。
- **exp11**: step 0–31 MSE 基本维持在 1.6–4.1，step 44–89 主要在 10–14 区间，末端 step 99 为 13.35；delta ratio 105.4%，没有明显 VAE 坍塌或动作放大。

**结论**:

1. **resnet101 不值得继续**：离线误差和 delta ratio 均显著差于 resnet50，符合 181 ep 小数据下大 backbone 过拟合/不稳定风险。
2. **resnet50 长训方向成立**：exp11 相比 exp9，MSE 从 36.12 降到 6.10，MAE 从 1.96 降到 0.90，delta ratio 从 116.4% 收敛到 105.4%。
3. **下一步优先实机评测 exp11**：使用同一套 25 次协议；若实机也提升，后续只在 resnet50 + 长训策略上微调，不再扩大 backbone。

### 下一轮实验：长训变量拆解

**目标**: 验证 exp11 的收益来自长训/学习率调度本身，还是与 backbone/数据增强存在交互。

| 实验        | backbone | steps | scheduler | aug | 对照对象 | 目的                         |
| --------- | -------- | ----- | --------- | --- | ---- | -------------------------- |
| **exp12** | resnet101 | 300k | cosine    | ✗   | exp10 | 判断 resnet101 是否只是 100k 未训够 |
| **exp13** | resnet50  | 300k | cosine    | 默认全部 | exp11 | 判断长训后默认图像增强是否从退化变为正收益   |

**设计原则**:

1. exp12 只在 exp10 基础上改变训练步数和 scheduler，不同时加 aug。
2. exp13 只在 exp11 基础上加图像增强，不改变 backbone / steps / scheduler。
3. 评估仍使用 rollout 100-step，并优先和 exp11 比较；实机只测离线不退化的模型。

```bash
# exp12: resnet101 长训 + cosine 学习率调度
export HF_USER=lepao
export JOB_NAME=act_so101_v04_exp12_resnet101_300k_cosine
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --dataset.revision=main \
    --dataset.force_cache_sync=true \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=50000 \
    --steps=300000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --policy.vision_backbone=resnet101 \
    --policy.pretrained_backbone_weights=ResNet101_Weights.IMAGENET1K_V1 \
    --wandb.enable=true \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=5000 \
    --scheduler.num_decay_steps=300000 \
    --scheduler.peak_lr=1e-5 \
    --scheduler.decay_lr=1e-6
```

```bash
# exp13: resnet50 长训 + cosine + 默认全部图像增强
export HF_USER=lepao
export JOB_NAME=act_so101_v04_exp13_resnet50_300k_cosine_aug
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=50000 \
    --steps=300000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --policy.vision_backbone=resnet50 \
    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 \
    --dataset.image_transforms.enable=true \
    --wandb.enable=true \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=5000 \
    --scheduler.num_decay_steps=300000 \
    --scheduler.peak_lr=1e-5 \
    --scheduler.decay_lr=1e-6
```

评估命令：

```bash
python scripts/eval_act.py --model exp12 exp13 --eval_steps 100 --mode rollout --output outputs/eval_exp12_exp13_rollout100.json
```

**预期判读**:

| 结果 | 结论 | 下一步 |
| ---- | ---- | ---- |
| exp12 仍明显差于 exp11 | resnet101 容量过大，不再扩大 backbone | 停止 resnet101/152 |
| exp12 接近或超过 exp11 | resnet101 需要长训才稳定 | 实机测 exp12 |
| exp13 优于 exp11 | 长训后 aug 有正收益 | 实机测 exp13，并考虑更温和 aug sweep |
| exp13 差于 exp11 | 数据增强仍破坏小数据分布 | 保持 exp11 作为 ACT 最强候选 |

### exp13 离线评估结果

评估命令：

```bash
python scripts/eval_act.py --model exp13 --eval_steps 100 --mode rollout --output outputs/eval_exp13_rollout100.json
```

| 实验 | backbone | checkpoint | 训练策略 | aug | eval | MSE | MAE | Δratio | 结论 |
| ---- | -------- | ---------- | -------- | --- | ---- | ---: | ---: | -----: | ---- |
| exp11 | resnet50 | 300k | 300k cosine | 否 | 100 | **6.10** | **0.90** | 105.4% | 当前最优 ACT 候选 |
| exp12 | resnet101 | 200k | 300k cosine | 否 | 100 | 42.26 | 1.99 | **104.8%** | 动作幅度接近 GT，但方向/时序误差大 |
| exp13 | resnet50 | 300k | 300k cosine | 是 | 100 | 34.66 | 1.78 | 108.1% | 明显差于 exp11 |

**结果判读**:

1. exp13 相比 exp11，MSE 从 6.10 升至 34.66，约退化 5.7x；MAE 从 0.90 升至 1.78，约退化 2.0x。
2. exp12 在 200k checkpoint 时 MSE 为 42.26、MAE 为 1.99，仍明显差于 exp11，也略差于 exp13。
3. exp12 的 Δratio 为 104.8%，动作幅度最接近 GT，但误差仍高，说明问题主要不是动作放大，而是动作方向或时序匹配变差。
4. 默认图像增强在当前 SO101 小数据集上仍然破坏视觉分布，不建议作为实机优先模型。

**结论**: 当前实验显示 resnet50 是 SO101 ACT 的性能甜点。保持 `act_so101_v04_exp11_resnet50_300k_cosine` 作为 ACT 最强候选，优先进行实机评测；`exp12` 200k 和 `exp13` 不进入优先实机队列。

exp12 200k 评估命令：

```bash
python scripts/eval_act.py --model outputs/train/act_so101_v04_exp12_resnet101_300k_cosine/checkpoints/200000/pretrained_model --eval_steps 100 --mode rollout --output outputs/eval_exp12_200k_rollout100.json
```

### v05 实验设计：600 ep + resnet50 + 500k

**目标**: 数据集扩充到 600 episodes 后，基于当前最优的 `exp11` 结论继续验证 resnet50 长训路线。不再继续 resnet101/resnet152，也不做从旧 checkpoint 微调。

共同固定配置：

```text
policy: ACT
backbone: resnet50
chunk_size: 100
n_action_steps: 100
kl_weight: 10.0
steps: 500000
batch_size: 8
save_freq: 50000
scheduler: cosine_decay_with_warmup
```

| 实验 | 目的 | 初始化 | aug | peak_lr | decay_lr | 结论判读 |
| ---- | ---- | ------ | --- | ------: | -------: | -------- |
| exp14 | 600 ep 新主基线 | 从头训练 | 否 | 1e-5 | 1e-6 | 若显著优于 exp11，说明扩数据 + 长训有效 |
| exp15 | 低学习率长训对照 | 从头训练 | 否 | 5e-6 | 5e-7 | 若优于 exp14，说明 600 ep + 500k 需要更保守 LR |
| exp16 | 温和图像增强对照 | 从头训练 | 温和 color jitter | 1e-5 | 1e-6 | **未训练，编号转入 v06 fixed 数据集实验** |

训练顺序建议：`exp14 -> exp15`。原计划 `exp16` 尚未训练，编号转入 `lepao/so101_test_fixed` 上重新定义。

#### exp14: resnet50 500k cosine 主基线

```bash
export HF_USER=lepao
export JOB_NAME=act_so101_v05_exp14_resnet50_500k_cosine_600ep

lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --dataset.revision=main \
    --dataset.force_cache_sync=true \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=50000 \
    --steps=500000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --policy.vision_backbone=resnet50 \
    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 \
    --wandb.enable=true \
    --use_policy_training_preset=false \
    --optimizer.type=adamw \
    --optimizer.lr=1e-5 \
    --optimizer.weight_decay=0.0001 \
    --optimizer.grad_clip_norm=10 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=10000 \
    --scheduler.num_decay_steps=500000 \
    --scheduler.peak_lr=1e-5 \
    --scheduler.decay_lr=1e-6
```

#### exp15: resnet50 500k cosine 低学习率对照

```bash
export HF_USER=lepao
export JOB_NAME=act_so101_v05_exp15_resnet50_500k_cosine_lr5e6_600ep

lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --dataset.revision=main \
    --dataset.force_cache_sync=true \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=50000 \
    --steps=500000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --policy.vision_backbone=resnet50 \
    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 \
    --use_policy_training_preset=false \
    --optimizer.type=adamw \
    --optimizer.lr=5e-6 \
    --optimizer.weight_decay=0.0001 \
    --optimizer.grad_clip_norm=10 \
    --policy.optimizer_lr_backbone=5e-6 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=10000 \
    --scheduler.num_decay_steps=500000 \
    --scheduler.peak_lr=5e-6 \
    --scheduler.decay_lr=5e-7 \
    --wandb.enable=true
```

#### exp16: resnet50 500k cosine 温和增强对照（已迁移）

原计划在 `lepao/so101_test` 上对比 `exp14` 增加温和 color jitter；该实验尚未训练，编号保留并转入 v06 `lepao/so101_test_fixed` 实验中，不再使用下面旧数据集命令。

```bash
export HF_USER=lepao
export JOB_NAME=act_so101_v05_exp16_resnet50_500k_cosine_mildaug_600ep

lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --dataset.revision=main \
    --dataset.force_cache_sync=true \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=50000 \
    --steps=500000 \
    --batch_size=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --policy.vision_backbone=resnet50 \
    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=2 \
    --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"saturation":{"weight":0.5,"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}}}' \
    --use_policy_training_preset=false \
    --optimizer.type=adamw \
    --optimizer.lr=1e-5 \
    --optimizer.weight_decay=0.0001 \
    --optimizer.grad_clip_norm=10 \
    --policy.optimizer_lr_backbone=1e-5 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=10000 \
    --scheduler.num_decay_steps=500000 \
    --scheduler.peak_lr=1e-5 \
    --scheduler.decay_lr=1e-6 \
    --wandb.enable=true \
    --wandb.disable_artifact=true
```

评估建议：每个实验在 `100k/200k/300k/400k/500k` checkpoint 上统一跑 rollout 100-step 离线评估，先看 val/test split，再决定是否进入实机评测。

### exp14 / exp15 离线评估结果

评估命令：

```bash
# 全量 600 ep，按 checkpoint 单独保存，避免同名模型结果覆盖
python scripts/eval_act.py \
    --model outputs/train/act_so101_v05_exp14_resnet50_500k_cosine_600ep/checkpoints/500000/pretrained_model \
    --eval_steps 100 \
    --mode rollout \
    --output outputs/eval_exp14_500000_rollout100.json

python scripts/eval_act.py \
    --model outputs/train/act_so101_v05_exp15_resnet50_500k_cosine_lr5e6_600ep/checkpoints/500000/pretrained_model \
    --eval_steps 100 \
    --mode rollout \
    --output outputs/eval_exp15_500000_rollout100.json

# test split 复核
python scripts/eval_act.py \
    --model outputs/train/act_so101_v05_exp14_resnet50_500k_cosine_600ep/checkpoints/500000/pretrained_model \
            outputs/train/act_so101_v05_exp15_resnet50_500k_cosine_lr5e6_600ep/checkpoints/500000/pretrained_model \
    --eval_steps 100 \
    --mode rollout \
    --split test \
    --output outputs/eval_exp14_exp15_500k_test_rollout100.json
```

> 注意：`eval_act.py` 当前用模型目录名作为 JSON key；同一实验的多个 checkpoint 一次性传入时会互相覆盖。比较多 checkpoint 时应按 checkpoint 单独输出，或先修复 key 命名。

全量 600 ep rollout 100-step：

| 实验 | checkpoint | MSE | MAE | Δratio | MSE p50 | MSE p90 | 结论 |
| ---- | ----------: | --: | --: | -----: | ------: | ------: | ---- |
| exp14 | 100k | 172.34 | 6.02 | 139.4% | 77.18 | 455.77 | 早期未收敛 |
| exp14 | 200k | 174.05 | 5.69 | 129.2% | 65.96 | 469.45 | 仍未收敛 |
| exp14 | 300k | 151.56 | 4.79 | 125.3% | 30.62 | 410.96 | 开始改善但误差仍高 |
| exp14 | 400k | 22.28 | 1.98 | 111.8% | 6.50 | 43.73 | 明显收敛 |
| **exp14** | **500k** | **10.90** | **1.46** | **109.4%** | **4.03** | **17.50** | **全量最优** |
| exp15 | 100k | 159.64 | 6.09 | 153.6% | 76.71 | 385.42 | 早期未收敛 |
| exp15 | 200k | 87.65 | 4.40 | 128.7% | 34.80 | 243.60 | 比 exp14 同步数好 |
| exp15 | 300k | 34.99 | 2.70 | 121.7% | 13.44 | 73.95 | 稳定改善 |
| exp15 | 400k | 15.47 | 2.03 | 118.0% | 8.71 | 29.63 | 接近 exp14 |
| exp15 | 500k | 12.15 | 1.77 | 117.5% | 6.73 | 19.08 | 次优，动作幅度偏大 |

500k test split 复核：

| 实验 | checkpoint | MSE | MAE | Δratio | MSE p50 | MSE p90 | 结论 |
| ---- | ----------: | --: | --: | -----: | ------: | ------: | ---- |
| **exp14** | **500k** | **6.29** | **1.31** | **108.8%** | **3.33** | **9.94** | **test split 仍最优** |
| exp15 | 500k | 10.34 | 1.83 | 116.3% | 6.89 | 19.22 | 明显差于 exp14 |

**结果判读**：

1. **exp14@500k 是当前 ACT 最强离线候选**：全量 MSE=10.90、MAE=1.46、Δratio=109.4%；test split MSE=6.29、MAE=1.31、Δratio=108.8%。
2. **exp15 低学习率未超过 exp14**：虽然 200k/300k 阶段比 exp14 收敛更平滑，但 500k 终点在 MSE、MAE、Δratio 上均更差，表现为偏保守/欠拟合，同时动作幅度仍偏大。
3. **600 ep + resnet50 + 500k 路线成立**：同当前 600 ep 数据口径重评 exp11，MSE=477.77、MAE=12.58、Δratio=127.0%，说明旧 181 ep 模型已经明显不适配新数据分布。
4. **优先实机评测 exp14@500k**：路径为 `outputs/train/act_so101_v05_exp14_resnet50_500k_cosine_600ep/checkpoints/500000/pretrained_model`。后续 fixed 数据集实验以 v05 配置为基础，编号从 v06 exp16 继续。

### v06 实验设计：fixed 数据集主基线

**目标**: 数据集切换到 `lepao/so101_test_fixed` 后，使用与 `exp15` 一致的低学习率，并将训练延长到 700k step，建立 fixed 数据集新 baseline。原 v05 的 `exp16` 尚未训练，编号转入本轮作为 fixed 数据集主基线。

**关键变化**:

1. 数据集从 `lepao/so101_test` 切换为 `lepao/so101_test_fixed`。
2. 训练配置沿用 ACT 长训路线：`resnet50 + chunk100 + 700k + cosine`，学习率与 `exp15` 保持一致。
3. 所有新实验启用 W&B 日志但禁用 artifact：`--wandb.disable_artifact=true`，避免模型权重上传到 W&B。
4. 仍保留 `--policy.push_to_hub=true`，模型权重上传到 HuggingFace Hub，而不是 W&B artifact。

共同固定配置：

```text
dataset: lepao/so101_test_fixed
policy: ACT
backbone: resnet50
chunk_size: 100
n_action_steps: 100
kl_weight: 10.0
steps: 700000
batch_size: 8
save_freq: 50000
scheduler: cosine_decay_with_warmup
wandb.disable_artifact: true
```

| 实验 | 目的 | 初始化 | aug | peak_lr | decay_lr | 结论判读 |
| ---- | ---- | ------ | --- | ------: | -------: | -------- |
| **exp16** | fixed 数据集主基线 | 从头训练 | 否 | 5e-6 | 5e-7 | 和 exp15 对比，判断 fixed 数据集 + 700k 是否提升 ACT 上限 |
| exp17 | fixed 数据集温和增强对照 | 从头训练 | 温和 color jitter | 5e-6 | 5e-7 | 若优于 exp16，说明 fixed 数据集下轻量增强有正收益 |

训练顺序建议：先跑 `exp16`。只有当 `exp16` 离线和/或实机表现稳定后，再决定是否跑 `exp17`。

评估建议：每个实验在 `100k/200k/300k/400k/500k/600k/700k` checkpoint 上统一跑 rollout 100-step，重点比较 500k 后是否继续下降，避免 700k 只是过拟合或收益很小。

#### exp16: fixed 数据集 resnet50 700k cosine 主基线

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_USER=lepao
export JOB_NAME=act_so101_v06_exp16_resnet50_700k_cosine_fixed
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test_fixed \
    --dataset.revision=main \
    --dataset.force_cache_sync=true \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=50000 \
    --steps=700000 \
    --batch_size=8 \
    --num_workers=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --policy.vision_backbone=resnet50 \
    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --use_policy_training_preset=false \
    --optimizer.type=adamw \
    --optimizer.lr=5e-6 \
    --optimizer.weight_decay=0.0001 \
    --optimizer.grad_clip_norm=10 \
    --policy.optimizer_lr_backbone=5e-6 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=10000 \
    --scheduler.num_decay_steps=700000 \
    --scheduler.peak_lr=5e-6 \
    --scheduler.decay_lr=5e-7
```

#### exp17: fixed 数据集 resnet50 700k cosine 温和增强对照

仅对比 `exp16` 增加温和 color jitter，不使用 affine/crop/rotation/perspective，避免破坏空间几何关系。

```bash
export CUDA_VISIBLE_DEVICES=1
export HF_USER=lepao
export JOB_NAME=act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test_fixed \
    --dataset.revision=main \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=50000 \
    --steps=700000 \
    --batch_size=8 \
    --num_workers=8 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.kl_weight=10.0 \
    --policy.vision_backbone=resnet50 \
    --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=2 \
    --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"saturation":{"weight":0.5,"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}}}' \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --use_policy_training_preset=false \
    --optimizer.type=adamw \
    --optimizer.lr=5e-6 \
    --optimizer.weight_decay=0.0001 \
    --optimizer.grad_clip_norm=10 \
    --policy.optimizer_lr_backbone=5e-6 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=10000 \
    --scheduler.num_decay_steps=700000 \
    --scheduler.peak_lr=5e-6 \
    --scheduler.decay_lr=5e-7
```

评估命令：

```bash
python scripts/eval_act.py \
    --dataset_repo lepao/so101_test_fixed \
    --model outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/700000/pretrained_model \
    --eval_steps 100 \
    --mode rollout \
    --output outputs/eval_exp16_fixed_700000_rollout100.json

python scripts/eval_act.py \
    --dataset_repo lepao/so101_test_fixed \
    --model outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/700000/pretrained_model \
    --eval_steps 100 \
    --mode rollout \
    --split test \
    --output outputs/eval_exp16_fixed_700000_test_rollout100.json
```

**预期判读**：

| 结果 | 结论 | 下一步 |
| ---- | ---- | ------ |
| exp16 明显优于 exp15 | fixed 数据集修复有效 | 优先实机评测 exp16 |
| exp16 接近 exp15 | fixed 数据集没有显著改变离线指标 | 仍实机测 exp16，确认数据修复是否改善真实成功率 |
| exp16 差于 exp15 | fixed 数据集分布或标注变化影响训练 | 检查数据集差异、episode 分布和动作尺度 |
| exp16 稳定后仍需提升泛化 | 进入 exp17 温和增强对照 | 只加 color jitter，不加几何增强 |

### exp16 / exp17 离线评估结果

评估命令：

```bash
python scripts/eval_act.py \
    --dataset_repo lepao/so101_test_fixed \
    --model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/050000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/100000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/150000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/200000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/250000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/300000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/350000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/400000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/450000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/500000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/550000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/600000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/650000/pretrained_model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/700000/pretrained_model \
    --eval_steps 100 \
    --mode rollout \
    --output outputs/eval_exp16_fixed_checkpoints_rollout100.json

python scripts/eval_act.py \
    --dataset_repo lepao/so101_test_fixed \
    --model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/050000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/100000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/150000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/200000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/250000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/300000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/350000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/400000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/450000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/500000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/550000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/600000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/650000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/700000/pretrained_model \
    --eval_steps 100 \
    --mode rollout \
    --output outputs/eval_exp17_fixed_checkpoints_rollout100.json
```

全量 fixed 数据集 rollout 100-step：

| checkpoint | exp16 MSE | exp16 MAE | exp16 Δratio | exp17 MSE | exp17 MAE | exp17 Δratio |
| ---------: | --------: | --------: | ------------: | --------: | --------: | ------------: |
| 050000 | 247.44 | 8.22 | 169.6% | 270.11 | 8.57 | 173.2% |
| 100000 | 145.07 | 5.88 | 151.7% | 166.35 | 6.30 | 153.6% |
| 150000 | 70.62 | 3.98 | 130.6% | 95.03 | 4.42 | 135.9% |
| 200000 | 97.63 | 4.50 | 127.3% | 78.64 | 4.08 | 128.7% |
| 250000 | 42.29 | 3.05 | 118.7% | 50.00 | 3.27 | 122.3% |
| 300000 | 60.86 | 3.40 | 121.0% | 77.48 | 3.75 | 124.2% |
| 350000 | 22.63 | 2.29 | 115.1% | 28.39 | 2.47 | 115.6% |
| 400000 | 20.30 | 2.21 | 115.3% | 21.62 | 2.19 | 114.3% |
| 450000 | 14.20 | 1.88 | 114.7% | 15.93 | 1.98 | 115.1% |
| 500000 | 10.83 | 1.66 | 113.6% | 15.50 | 1.90 | 114.7% |
| 550000 | 8.98 | 1.57 | 113.7% | 11.22 | 1.69 | 113.4% |
| 600000 | 8.28 | 1.49 | 113.4% | 9.90 | 1.58 | 113.6% |
| 650000 | 7.80 | 1.48 | 113.3% | 8.68 | 1.51 | 112.9% |
| **700000** | **7.01** | **1.41** | **113.4%** | **8.43** | **1.50** | **113.1%** |

700k test split 复核：

评估命令：

```bash
python scripts/eval_act.py \
    --dataset_repo lepao/so101_test_fixed \
    --model \
        outputs/train/act_so101_v06_exp16_resnet50_700k_cosine_fixed/checkpoints/700000/pretrained_model \
        outputs/train/act_so101_v06_exp17_resnet50_700k_cosine_mildaug_fixed/checkpoints/700000/pretrained_model \
    --eval_steps 100 \
    --mode rollout \
    --split test \
    --output outputs/eval_exp16_exp17_fixed_700000_test_rollout100.json
```

| 实验 | checkpoint | MSE | MAE | Δratio | MSE p50 | MSE p90 | 结论 |
| ---- | ----------: | --: | --: | -----: | ------: | ------: | ---- |
| **exp16** | **700k** | **6.41** | **1.52** | 114.4% | **4.50** | **11.81** | test split 最优 |
| exp17 | 700k | 7.43 | 1.55 | **113.9%** | 4.69 | 12.63 | 温和增强未提升 |

**结果判读**：

1. 两个模型都在 700k 达到各自最优，500k 后仍有持续收益，700k 不是明显过拟合。
2. **exp16@700k 是 fixed 数据集当前最优 ACT 候选**：全量 MSE=7.01、MAE=1.41、Δratio=113.4%；test split MSE=6.41、MAE=1.52、Δratio=114.4%。
3. exp17 的温和 color jitter 未超过 exp16：700k 全量 MSE 从 7.01 升至 8.43，test MSE 从 6.41 升至 7.43。
4. 继续保持不加图像增强的 `exp16@700k` 作为实机优先模型；`exp17@700k` 暂不进入优先实机队列。
