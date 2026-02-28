# 硬件映射
连接好硬件后，在powershell内执行，完成 windows 硬件映射 wsl，
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
可查看具体映射设备
```bash
lerobot-find-port
```
# 手动标注
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

# 双臂遥操作启动命令
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=1 
```

# 遥操作+双相机
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

# 遥操作+相机启动命令
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

# 录制数据集
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

# 训练
```bash
export HF_USER=lepao
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_test \
    --job_name=act_so101_test \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/act_so101_test \
    --save_freq=5000 \
    --batch_size=128 \
    --wandb.enable=false
```

# 训练smolvla
```bash
export HF_USER=lepao
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=smolvla \
    --output_dir=outputs/train/smolvla_so101_test \
    --job_name=smolvla_so101_test \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/smolvla_so101_test \
    --save_freq=1000 \
    --batch_size=128 \
    --steps=20000 \
    --wandb.enable=false
```
# 继续训练模型
```bash
lerobot-train \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true

lerobot-train \
  --config_path=outputs/train/smolvla_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

# 上传模型
```bash
hf upload lepao/act_so101_test \
  outputs/train/act_so101_test/checkpoints/last/pretrained_model

hf upload lepao/smolvla_so101_test \
  outputs/train/smolvla_so101_test/checkpoints/last/pretrained_model
```