# 连接硬件
可查看具体映射设备
```powershell
lerobot-find-port
```
# 手动标注
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

# 双臂遥操作启动命令
```powershell
lerobot-teleoperate `
    --robot.type=so101_follower `
    --robot.port=COM4 `
    --robot.id=0 `
    --teleop.type=so101_leader `
    --teleop.port=COM5 `
    --teleop.id=1 
```

# 遥操作+双相机
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

# 遥操作+相机启动命令
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

# 录制数据集
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

# 查看数据集
```powershell
lerobot-dataset-viz --repo-id lepao/so101_test --episode-index 0
```

# 训练
```powershell
$env:HF_USER="lepao"
lerobot-train `
    --dataset.repo_id=$env:HF_USER/so101_test `
    --policy.type=act `
    --output_dir=outputs/train/act_so101_test `
    --job_name=act_so101_test `
    --policy.device=cuda `
    --policy.push_to_hub=true `
    --wandb.enable=true
```

  # 推理并录制
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