# ACT 策略实验指南

## 1、项目背景

- **任务**: SO101 机械臂抓取纸方块 ("Grab the paper cube")
- **数据集**: `lepao/so101_test`（181 episodes, 80328 frames, 30fps, 6-DOF joint positions）
- **硬件**: SO101 follower + leader 双臂，双相机（handeye + fixed, 640x360）

## 2、评价指标

| 指标              | 含义                 | 判断标准                     |
| --------------- | ------------------ | ------------------------ |
| **Delta ratio** | 预测动作变化幅度 / GT 变化幅度 | 接近 100% 最佳，< 10% 说明机械臂不动 |
| **MAE**         | 平均绝对误差（度）          | 越低越好，但需配合 delta ratio 看  |
| **MSE**         | 均方误差               | 越低越好，对异常值敏感              |
| **实机效果**        | 机械臂实际抓取表现          | 最终评判标准                   |

> **注意**: MAE/MSE 低不代表效果好。chunk16 模型 MAE 最低但 delta ratio 仅 1.9%，机械臂不动。

## 3、离线评估工具

统一评估脚本 `scripts/eval_act.py`，支持三种模式：

```bash
# 单模型评估（含 MSE/MAE/Delta ratio/Per-step 详细指标）
python scripts/eval_act.py --model v0.1

# 模拟 n_action_steps=20（只评估前 20 步）
python scripts/eval_act.py --model v0.1 --eval_steps 20

# 多模型对比
python scripts/eval_act.py --model v0.1 chunk16 chunk16_dec7_aug

# 单帧详细诊断（值域、delta、逐维对比）
python scripts/eval_act.py --model v0.1 --diagnose
```

模型快捷名: `v0.1` / `chunk16` / `chunk16_dec7_aug` / `exp2_kl5` / `exp3_dec7` / `exp4_aug`

结果输出: `outputs/eval_results.json`

## 4、第一轮实验：三模型对比

### 配置对比

| <br />             | v0.1 (基线) | chunk16 | chunk16+dec7+aug |
| ------------------ | --------- | ------- | ---------------- |
| chunk\_size        | 100       | 16      | 16               |
| n\_action\_steps   | 100       | 16      | 16               |
| kl\_weight         | 10        | 10      | 10               |
| n\_obs\_steps      | 1         | 1       | 1                |
| n\_decoder\_layers | 1         | 1       | **7**            |
| 参数量                | 52M       | 52M     | 84M              |
| 数据增强               | ✗         | ✗       | **✓**            |

### 离线评估结果

| <br />      | v0.1         | chunk16    | chunk16+dec7+aug |
| ----------- | ------------ | ---------- | ---------------- |
| MSE         | 33.08        | **5.71**   | 最差               |
| MAE         | 2.08°        | **1.24°**  | 最差               |
| Delta ratio | **106.4%** ✅ | **1.9%** ❌ | 极低 ❌             |
| 实机效果        | **最好**（仍不理想） | 不动         | 最差               |

### 关键结论

1. **chunk\_size=16 导致 VAE 模式坍塌**: 预测 delta 仅为 GT 的 2%，机械臂几乎不动。MSE/MAE 反而最低（输出"安全平均值"），但不能用
2. **v0.1 前 20 步预测较准，后期误差累积**: MSE(20步)=7.77 vs MSE(100步)=33.08
3. **chunk16+dec7+aug 三个变量混合，无法单独归因**

## 5、实验1: v0.1 + n\_action\_steps=20（离线验证已完成）

不需要重新训练，仅推理时执行前 20 步后重规划。

| 指标          | v0.1 (act=100) | 实验1 (act=20) | 改善         |
| ----------- | -------------- | ------------ | ---------- |
| MSE         | 33.08          | **7.77**     | **-76.5%** |
| MAE         | 2.08°          | **1.48°**    | **-29.1%** |
| Delta ratio | 106.4%         | **106.4%**   | 不变         |

**结论**: 执行前 20 步可大幅降低误差，动作幅度不受影响。等待实机验证。

## 6、第二轮实验：已完成

### 实验矩阵

| 实验      | chunk | act    | kl    | decoder | aug   | 状态       | 验证目标            |
| ------- | ----- | ------ | ----- | ------- | ----- | -------- | --------------- |
| v0.1 基线 | 100   | 100→20 | 10    | 1       | ✗     | 已完成      | -               |
| 实验1     | 100   | 20     | 10    | 1       | ✗     | 离线验证完成   | act=20 效果       |
| 实验2     | 100   | 20     | **5** | 1       | ✗     | 训练+验证完成 | kl\_weight 降到 5 |
| 实验3     | 100   | 20     | 10    | **7**   | ✗     | 训练+验证完成 | decoder 深度      |
| 实验4     | 100   | 20     | 10    | 1       | **✓** | 训练+验证完成 | 数据增强            |

> **注意**: ACT 策略不支持 `n_obs_steps > 1`（代码硬编码限制），历史观测帧实验已移除。

### 第二轮总结

1. `kl_weight=5` 对当前数据集没有显著收益
2. `decoder=7` 在 standard val 上最好，但 rollout/部署视角不如 `v0.1`
3. 数据增强在当前 181 episodes 实机数据上显著退化
4. 下一步不应继续深挖这三项超参，优先做实机部署参数和数据规模优化

### 训练命令

```bash
# 实验2: kl_weight=5
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

# 实验3(对照): n_decoder_layers=7
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

# 实验4(对照): 数据增强（仅 affine，参考 PushT 最优方案）
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
    --dataset.image_transforms.tfs='{"affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-10, 10], "translate": [0.1, 0.1]}}}' \
    --wandb.enable=true
```

### 推理命令（远程异步）

```powershell
# 1. 远程 GPU 启动 PolicyServer
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=6006

# 2. 本地 SSH 端口转发
ssh -CNg -L 6006:127.0.0.1:6006 root@<REMOTE_GPU_IP> -p <SSH_PORT>

# 3. 本地启动 RobotClient（替换模型路径）
python -m lerobot.async_inference.robot_client `
    --server_address=127.0.0.1:6006 `
    --robot.type=so101_follower `
    --robot.port=COM3 `
    --robot.id=0 `
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 1, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}}" `
    --task="Grab the paper cube" `
    --policy_type=act `
    --pretrained_name_or_path=lepao/<MODEL_NAME> `
    --policy_device=cuda `
    --client_device=cpu `
    --actions_per_chunk=20 `
    --chunk_size_threshold=0.1 `
    --aggregate_fn_name=weighted_average `
    --debug_visualize_queue_size=true
```

## 7、实验结果记录

### 离线对比总表（eval_steps=20）

| 模型 | MSE(20步)↓ | MAE(20步)↓ | Delta ratio | 排名 |
|---|---|---|---|---|
| **v0.1 基线** | **7.77** | **1.479** | 106.4% | **1** |
| 实验2 (kl=5) | 8.11 | 1.562 | **108.9%** | 2 |
| 实验3 (dec=7) | 10.02 | 1.556 | 113.8% | 3 |
| 实验4 (aug) | 15.20 | 1.788 | 105.2% | 4 |

> 数据增强效果最差（MSE 翻倍），kl=5 和 dec=7 均未超过基线。**v0.1 + act=20（实验1）仍是最佳方案**。

### Val 集对比（episode split = 80/10/10）

#### Standard（teacher-forced）

| 模型 | MSE(100步)↓ | MAE(100步)↓ | Delta ratio |
|---|---|---|---|
| v0.1 | 20.63 | 1.827 | 113.9% |
| 实验2 (kl=5) | 21.35 | 1.885 | 111.2% |
| 实验3 (dec=7) | **19.42** | **1.786** | 116.6% |
| 实验4 (aug) | 48.53 | 2.444 | 112.5% |

#### Rollout-style（更接近重规划）

| 模型 | MSE(20步)↓ | MAE(20步)↓ | MSE(100步)↓ | 结论 |
|---|---|---|---|---|
| v0.1 | **9.22** | **1.580** | **21.82** | 更稳，最贴近实机 |
| 实验2 (kl=5) | 10.06 | 1.690 | 24.28 | 中性偏差 |
| 实验3 (dec=7) | 10.96 | 1.581 | 21.40 | standard 最好，但 rollout 不如 v0.1 |
| 实验4 (aug) | 12.86 | 1.805 | 51.70 | 最差 |

> 结论分裂：**teacher-forced 离线最优是实验3(dec=7)，但 rollout/部署视角下 v0.1 更优**。这与实机上 `act=100` 比 `act=20` 更稳的现象一致。

### 当前判断

1. **离线最优**: 实验3(dec=7)
2. **部署/实机优先**: v0.1
3. **数据增强不推荐**: 在 181 episodes 上明显退化
4. **kl=5 收益不明显**: 不值得优先继续追


## 9、相关文件

| 文件 | 说明 |
|---|---|
| `scripts/eval_act.py` | 统一离线评估脚本（评估+诊断+对比） |
| `scripts/quickstart_win.md` | Windows 本地操作指南 |
| `scripts/quickstart.md` | Linux 操作指南 |
| `outputs/eval_results.json` | 评估结果输出 |
