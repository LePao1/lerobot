# 0、环境安装

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot

conda install -y ffmpeg -c conda-forge
conda install -y evdev -c conda-forge

pip install -e .
pip install -e ".[feetech]"
```


# 1、硬件映射

连接好硬件后，在powershell内执行，完成 windows 硬件映射 wsl
```powershell
usbipd list
usbipd bind --busid 6-1
usbipd bind --busid 6-2
usbipd bind --busid 6-3
usbipd bind --busid 6-4
usbipd attach --wsl --busid 6-1
usbipd attach --wsl --busid 6-2
usbipd attach --wsl --busid 6-3
usbipd attach --wsl --busid 6-4
```
wsl 内可查看具体映射设备
```bash
lerobot-find-port
```


# 2、双臂校准

进行中位校准以及关节运动最大角度

follower（青色）
``` bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=0
```
leader（橙色）
```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=1
```
标定后参数文件保存在
```bash
tree ~/.cache/huggingface/lerobot                    
```
```bash
/home/lepao/.cache/huggingface/lerobot
└── calibration
    ├── robots
    │   └── so101_follower
    │       └── 0.json
    └── teleoperators
        └── so101_leader
            └── 0.json
```


# 3、双臂遥操作

## 3.1 遥操作

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=1 
```

## 3.2 遥操作+双相机

查找相机设备，会在`outputs/captured_images`目录下捕获相机图像
```bash
sudo chmod 666 /dev/video*
lerobot-find-cameras
```

查看相机支持格式
```bash
sudo apt update && sudo apt install v4l-utils
```
```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

遥操作+双相机 启动命令
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=1 \
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': '/dev/video0', 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG', 'backend': 'V4L2'}, 'fixed': {'type': 'opencv', 'index_or_path': '/dev/video2', 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG', 'backend': 'V4L2'}}" \
    --display_data=true
```


# 4、录制数据集

增加语音提示
```bash
sudo apt install speech-dispatcher
```
开始录制，可选择--display_data=false 关闭画面数据实时显示
```bash
rm -rf ~/.cache/huggingface/lerobot/lepao/so101_test
```

```bash
export HF_USER=lepao
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=1 \
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': '/dev/video0', 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG', 'backend': 'V4L2'}, 'fixed': {'type': 'opencv', 'index_or_path': '/dev/video2', 'width': 640, 'height': 360, 'fps': 30, 'fourcc': 'MJPG', 'backend': 'V4L2'}}" \
    --dataset.repo_id=${HF_USER}/so101_test \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=20 \
    --dataset.single_task="Grab the paper cube" \
    --dataset.push_to_hub=false \
    --robot.disable_torque_on_disconnect=true
```

查看数据集
```bash
lerobot-dataset-viz \
    --repo-id lepao/so101_test \
    --episode-index 0
```

查看所有相机的数据增强效果
```bash
lerobot-imgtransform-viz \
    --repo_id lepao/so101_test \
    --episodes "[0]" \
    --image_transforms.enable true \
    --all_cameras true
```

指定第 100 帧，并指定输出目录
```bash
lerobot-imgtransform-viz \
    --repo_id lepao/so101_test \
    --episodes "[0]" \
    --image_transforms.enable true \
    --all_cameras true \
    --frame_index 100 \
    --output_dir outputs/my_image_transforms
```

输出目录示例
```bash
outputs/image_transforms/so101_test/
├── observation_images_fixed/
└── observation_images_handeye/
```


# 5、训练

训练 act 模型（基线）
```bash
export HF_USER=lepao
export JOB_NAME=act_so101_test_v0.1
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=10000 \
    --steps=100000 \
    --batch_size=8 \
    --wandb.enable=true
```

基于 PushT 调优经验，抓方块任务建议按下面顺序做 ACT 小规模对比实验


ACT 调参实验模板（优先做 chunk sweep）
```bash
export HF_USER=lepao
export CHUNK=16
export ACTION=16
export JOB_NAME=act_so101_chunk${CHUNK}_act${ACTION}
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/${JOB_NAME} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/${JOB_NAME} \
    --save_freq=10000 \
    --steps=100000 \
    --batch_size=8 \
    --policy.chunk_size=${CHUNK} \
    --policy.n_action_steps=${ACTION} \
    --wandb.enable=true
```

ACT 调参实验模板（更深 decoder + 轻量增强）
```bash
export HF_USER=lepao
export JOB_NAME=act_so101_chunk16_dec7_aug
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
    --policy.n_decoder_layers=7 \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs='{"affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-10, 10], "translate": [0.1, 0.1]}}}' \
    --wandb.enable=true
```

ACT 长训练模板（在上面实验中选出最优配置后再继续拉长）
```bash
export HF_USER=lepao
export JOB_NAME=act_so101_chunk16_dec7_aug_100k
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
    --policy.n_decoder_layers=7 \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.tfs='{"affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-10, 10], "translate": [0.1, 0.1]}}}' \
    --wandb.enable=true
```

继续训练模型
```bash
lerobot-train \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

上传模型
```bash
hf upload lepao/act_so101_test \
  outputs/train/act_so101_test/checkpoints/last/pretrained_model
```


# 6、推理

## 6.1、远程推理（本地算力不足时使用）

当本地电脑算力不足时，可以将模型放在远程 GPU 服务器上进行推理，本地电脑只负责连接 SO-101 执行动作。


```
┌─────────────────────┐          gRPC            ┌─────────────────────┐
│  远程算力服务器      │ ◄──────────────────────► │  本地电脑            │
│  (GPU 服务器)        │                          │  (连接 SO-101)       │
│                     │                          │                      │
│  PolicyServer       │   observations ──────►   │  RobotClient         │
│  - 加载模型          │                          │  - 采集观测           │
│  - 运行推理          │ ◄────── actions ───────  │  - 发送给机器人        │
│  - 返回动作          │                          │  - 执行动作           │
└─────────────────────┘                          └─────────────────────┘
```

### 6.2.1、安装依赖（两边都要）

```bash
pip install -e ".[async]"
```

### 6.2.2、在远程服务器启动 PolicyServer

```bash
# 在远程 GPU 服务器上运行
python -m lerobot.async_inference.policy_server \
     --host=0.0.0.0 \
     --port=6006
```
