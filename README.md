# GraspLab
- 这个文件当中是自定义的一个mujoco环境，其中包含了GraspNet实现以及标准的PPO实现

## Run

- 在无头模式下运行：
```python
python train_ppo.py --use_image_obs --robot_type panda --total_timesteps 500000
# 或者 
python train_ppo.py --use_image_obs --robot_type panda --total_timesteps 500000 --headless
```
- 在有头模式下运行
```python
python train_ppo.py --use_image_obs --robot_type panda --total_timesteps 500000 --with_display
```

## New
  1. 运行训练（自动录制）

  ./run_improved_training.sh

  2. 测试视频功能

  python3 test_video_recording.py

  3. 自定义配置训练

  python3 train_improved_ppo.py \
      --video_interval 3 \        # 每3次evaluation录制
      --max_videos_per_eval 5 \   # 每次录制5个episodes
      --video_fps 60              # 60fps高帧率

