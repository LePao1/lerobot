# 0、环境安装

```powershell
conda create -y -n lerobot python=3.12
conda activate lerobot

conda install ffmpeg -c conda-forge
conda install evdev -c conda-forge

pip install -e .
pip install -e ".[feetech]"
```


# 1、连接硬件

win直接连接即可，可查看具体映射设备
```powershell
lerobot-find-port
```


# 2、双臂校准

进行中位校准以及关节运动最大角度

follower（青色）
``` powershell
lerobot-calibrate `
    --robot.type=so101_follower `
    --robot.port=COM4 `
    --robot.id=0
```
leader（橙色）
```powershell
lerobot-calibrate `
    --teleop.type=so101_leader `
    --teleop.port=COM5 `
    --teleop.id=1
```
标定后参数文件保存在
```powershell
# 在 PowerShell 中查看目录结构
tree $HOME\.cache\huggingface\lerobot
```
```powershell
C:\USERS\13461\.CACHE\HUGGINGFACE\LEROBOT
└── calibration
    ├── robots
    │   └── so101_follower
    │       └── 0.json
    └── teleoperators
        └── so101_leader
            └── 1.json
```


# 3、双臂遥操作

## 3.1 遥操作

```powershell
lerobot-teleoperate `
    --robot.type=so101_follower `
    --robot.port=COM4 `
    --robot.id=0 `
    --teleop.type=so101_leader `
    --teleop.port=COM5 `
    --teleop.id=1 
```
## 3.2 遥操作+双相机

遥操作+双相机

查找相机设备，会在`outputs/captured_images`目录下捕获相机图像
```powershell
lerobot-find-cameras
```

查看相机支持格式
```powershell
# 1. 先列出所有摄像头设备名称
ffmpeg -list_devices true -f dshow -i dummy

# 2. 根据设备名称查看支持的格式、分辨率和 FPS
# 注意：请将 "USB 2.0 Camera" 替换为您在第1步中看到的实际名称
ffmpeg -f dshow -list_options true -i video="USB 2.0 Camera"
```

遥操作+相机启动命令
```powershell
lerobot-teleoperate `
    --robot.type=so101_follower `
    --robot.port=COM4 `
    --robot.id=0 `
    --teleop.type=so101_leader `
    --teleop.port=COM5 `
    --teleop.id=1 `
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 1, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}}" `
    --display_data=true
```


# 4、录制数据集

开始录制，可选择--display_data=false 关闭画面数据实时显示
```powershell
# 删除旧的数据集缓存（可选）
rm -r $HOME\.cache\huggingface\lerobot\lepao\so101_test
```

```powershell
$env:HF_USER="lepao"
lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM4 `
    --robot.id=0 `
    --teleop.type=so101_leader `
    --teleop.port=COM5 `
    --teleop.id=1 `
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 1, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}}" `
    --dataset.repo_id=$env:HF_USER/so101_test `
    --dataset.num_episodes=50 `
    --dataset.episode_time_s=15 `
    --dataset.single_task="Grab the paper cube" `
    --dataset.push_to_hub=true `
    --dataset.streaming_encoding=true `
    --dataset.encoder_threads=2 `
    --robot.disable_torque_on_disconnect=true `
    --resume=true 
```

查看数据集
```powershell
lerobot-dataset-viz --repo-id lepao/so101_test --episode-index 0
```


# 5、训练

训练 act 模型
```powershell
export HF_USER=lepao
lerobot-train `
    --dataset.repo_id=${HF_USER}/so101_test `
    --policy.type=act \
    --output_dir=outputs/train/act_so101_test `
    --job_name=act_so101_test `
    --policy.device=cuda `
    --policy.push_to_hub=true `
    --policy.repo_id=${HF_USER}/act_so101_test `
    --save_freq=5000 `
    --steps=20000 `
    --batch_size=128 `
    --wandb.enable=false
```

训练smolvla
```powershell
pip install -e ".[smolvla]"
```
```powershell
export HF_USER=lepao
lerobot-train `
    --dataset.repo_id=${HF_USER}/so101_test `
    --policy.type=smolvla `
    --output_dir=outputs/train/smolvla_so101_test `
    --job_name=smolvla_so101_test `
    --policy.device=cuda `
    --policy.push_to_hub=true `
    --policy.repo_id=${HF_USER}/smolvla_so101_test `
    --save_freq=1000 `
    --batch_size=128 `
    --steps=20000 `
    --wandb.enable=false
```
继续训练模型
```bash
lerobot-train `
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model/train_config.json `
  --resume=true

lerobot-train `
  --config_path=outputs/train/smolvla_so101_test/checkpoints/last/pretrained_model/train_config.json `
  --resume=true
```

上传模型
```bash
hf upload lepao/act_so101_test `
  outputs/train/act_so101_test/checkpoints/last/pretrained_model

hf upload lepao/smolvla_so101_test `
  outputs/train/smolvla_so101_test/checkpoints/last/pretrained_model
```


# 6、推理

## 6.1、本地推理

smolvla_so101_test 模型

```powershell
lerobot-record  `
  --robot.type=so101_follower `
  --robot.port=COM4 `
  --robot.id=0 `
  --teleop.type=so101_leader `
  --teleop.port=COM5 `
  --teleop.id=1 `
  --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 1, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}}" `
  --policy.path=lepao/smolvla_so101_test `
  --dataset.single_task="Grab the paper cube" `
  --policy.device=cpu `
  --dataset.repo_id=lepao/eval_so101 `
  --dataset.push_to_hub=false
```
act_so101_test 模型

```powershell
lerobot-record  `
  --robot.type=so101_follower `
  --robot.port=COM4 `
  --robot.id=0 `
  --teleop.type=so101_leader `
  --teleop.port=COM5 `
  --teleop.id=1 `
  --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 1, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}}" `
  --policy.path=lepao/act_so101_test `
  --dataset.single_task="Grab the paper cube" `
  --policy.device=cpu `
  --dataset.repo_id=lepao/eval_so101 `
  --dataset.push_to_hub=false
```

上传数据集
```
hf upload lepao/so101_test \
  outputs/dataset/so101_test
```

## 6.2、远程推理（本地算力不足时使用）

当本地电脑算力不足时，可以将模型放在远程 GPU 服务器上进行推理，本地电脑只负责连接 SO-101 执行动作。

## 架构说明

```
┌─────────────────────┐          gRPC          ┌─────────────────────┐
│  远程算力服务器      │ ◄──────────────────────► │  本地电脑            │
│  (GPU 服务器)        │                          │  (连接 SO-101)        │
│                     │                          │                      │
│  PolicyServer       │   observations ──────►  │  RobotClient         │
│  - 加载模型           │                          │  - 采集观测           │
│  - 运行推理          │ ◄────── actions ───────  │  - 发送给机器人        │
│  - 返回动作          │                          │  - 执行动作           │
└─────────────────────┘                          └─────────────────────┘
```

### 6.2.1、安装依赖（两边都要）

```powershell
pip install -e ".[async]"
```

### 6.2.2、在远程服务器启动 PolicyServer

```powershell
# 在远程 GPU 服务器上运行
python -m lerobot.async_inference.policy_server `
     --host=0.0.0.0 `
     --port=8080
```

### 6.2.3、通过ssh将远程服务器端口映射到本地
```powershell
 ssh -CNg -L 8080:127.0.0.1:8080 root@123.456.789.123 -p 30499
```

### 6.2.4、在本地电脑启动 RobotClient（连接 SO-101）

act 模型
```powershell
python -m lerobot.async_inference.robot_client `
    --server_address=127.0.0.1:8080 `
    --robot.type=so101_follower `
    --robot.port=COM4 `
    --robot.id=0 `
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 1, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}}" `
    --task="Grab the paper cube" `
    --policy_type=act `
    --pretrained_name_or_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model `
    --policy_device=cuda `
    --client_device=cpu `
    --actions_per_chunk=100 `
    --chunk_size_threshold=0.1 `
    --aggregate_fn_name=weighted_average `
    --debug_visualize_queue_size=true
```
smolvla 模型
```powershell
python -m lerobot.async_inference.robot_client `
    --server_address=127.0.0.1:8080 `
    --robot.type=so101_follower `
    --robot.port=COM4 `
    --robot.id=0 `
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 1, 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG'}}" `
    --task="Grab the paper cube" `
    --policy_type=smolvla `
    --pretrained_name_or_path=outputs/train/smolvla_so101_test/checkpoints/last/pretrained_model `
    --policy_device=cuda `
    --client_device=cpu `
    --actions_per_chunk=60 `
    --chunk_size_threshold=0.3 `
    --aggregate_fn_name=weighted_average `
    --debug_visualize_queue_size=true
```