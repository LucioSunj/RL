#!/bin/bash

# 改进的PPO训练脚本 - 解决收敛问题
# 使用优化的超参数和简化的环境
# 包含evaluation视频录制功能

echo "Starting Improved PPO Training for Grasp Task"
echo "=============================================="

# 检查Python环境
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# 训练配置
ROBOT_TYPE="panda"
DIFFICULTY="easy"  # 从简单开始
MAX_EPISODE_STEPS=800
TOTAL_TIMESTEPS=500000

# 优化的PPO超参数
LR_ACTOR=3e-4
LR_CRITIC=3e-4
ENTROPY_COEF=0.05  # 增加探索
TARGET_KL=0.02     # 放宽KL限制
N_EPOCHS=8         # 减少更新轮数
BATCH_SIZE=64
BUFFER_SIZE=2048

# 视频录制配置
VIDEO_INTERVAL=5        # 每5次evaluation录制一次视频
MAX_VIDEOS_PER_EVAL=3   # 每次录制最多3个episodes
VIDEO_FPS=30           # 视频帧率

# 创建实验目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="improved_grasp_${ROBOT_TYPE}_${DIFFICULTY}_${TIMESTAMP}"
CHECKPOINT_DIR="./checkpoints_improved/${EXPERIMENT_NAME}"

echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Video recording: Every $VIDEO_INTERVAL evaluations"

# 开始训练
python3 train_improved_ppo.py \
    --robot_type $ROBOT_TYPE \
    --difficulty $DIFFICULTY \
    --max_episode_steps $MAX_EPISODE_STEPS \
    --total_timesteps $TOTAL_TIMESTEPS \
    --lr_actor $LR_ACTOR \
    --lr_critic $LR_CRITIC \
    --entropy_coef $ENTROPY_COEF \
    --target_kl $TARGET_KL \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --buffer_size $BUFFER_SIZE \
    --checkpoint_dir ./checkpoints_improved \
    --experiment_name $EXPERIMENT_NAME \
    --use_wandb \
    --wandb_offline \
    --wandb_project "improved_grasp_ppo" \
    --log_interval 10 \
    --save_interval 100 \
    --eval_interval 50 \
    --video_interval $VIDEO_INTERVAL \
    --max_videos_per_eval $MAX_VIDEOS_PER_EVAL \
    --video_fps $VIDEO_FPS \
    --headless

echo "Training completed!"
echo "Checkpoints saved in: $CHECKPOINT_DIR"
echo "Evaluation videos saved in: $CHECKPOINT_DIR/evaluation_videos"