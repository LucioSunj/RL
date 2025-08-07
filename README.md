# GraspLab
- 这个文件当中是自定义的一个mujoco环境，其中包含了GraspNet实现以及标准的PPO实现

## Run

PYTHONPATH=./grasp_lab/manipulator_grasp python train_ppo.py --use_image_obs --robot_type panda --total_timesteps 500000