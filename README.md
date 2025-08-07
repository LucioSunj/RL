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