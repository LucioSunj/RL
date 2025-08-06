# PPO强化学习抓取系统

这是一个完整的PPO强化学习系统，用于替换原有的GraspNet抓取检测和轨迹规划。系统包含训练、测试、集成等完整功能。

## 📁 文件结构

```
grasp_lab/
├── rl_grasp_env.py          # RL环境包装类
├── network.py               # 神经网络架构
├── ppo_agent.py            # PPO算法实现
├── train_ppo.py            # 训练脚本
├── test_ppo.py             # 测试脚本
├── ppo_integration.py      # 集成脚本（替换GraspNet）
├── PPO_README.md           # 本文档
└── checkpoints/            # 模型检查点目录
```

## 🚀 快速开始

### 1. 环境依赖

除了原有依赖外，还需要安装：

```bash
pip install torch torchvision
pip install gymnasium
pip install wandb
pip install opencv-python
pip install matplotlib
```

### 2. 快速训练

**最简单的训练命令：**
```bash
# 训练Panda机械臂，使用状态观察（不含图像）
python train_ppo.py --use_simple_env --robot_type panda --total_timesteps 500000

# 训练UR5e机械臂
python train_ppo.py --use_simple_env --robot_type ur5e --total_timesteps 500000
```

**包含图像观察的训练：**
```bash
python train_ppo.py --use_image_obs --robot_type panda --total_timesteps 1000000
```

### 3. 快速测试

```bash
# 测试训练好的模型
python test_ppo.py ./checkpoints/best_model.pt --n_episodes 10 --render

# 性能基准测试
python test_ppo.py ./checkpoints/best_model.pt --benchmark --n_episodes 100
```

### 4. 集成到现有环境

```bash
# 使用PPO替换GraspNet执行抓取
python ppo_integration.py --model_path ./checkpoints/best_model.pt --target 1.4 0.2 0.9

# 运行完整演示
python ppo_integration.py --demo
```

## 🎯 系统特性

### 环境特性
- **多机械臂支持**: Panda (7DOF) 和 UR5e (6DOF)
- **多模态观察**: RGB图像、深度图像、关节状态、末端位置
- **灵活奖励设计**: 密集奖励 vs 稀疏奖励
- **目标随机化**: 支持多目标训练
- **状态归一化**: 自动归一化观察空间

### PPO算法特性
- **标准PPO实现**: Clip objective + GAE
- **多输入网络**: 支持图像+状态的混合输入
- **自适应学习率**: StepLR调度器
- **梯度裁剪**: 防止梯度爆炸
- **早停机制**: KL散度监控

### 训练监控
- **Wandb集成**: 实时监控训练过程
- **详细统计**: 奖励、成功率、策略损失等
- **自动保存**: 最佳模型和定期检查点
- **评估循环**: 定期评估策略性能

## 📊 训练配置说明

### 环境配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--robot_type` | panda | 机械臂类型 (panda/ur5e) |
| `--use_simple_env` | False | 使用简化环境（仅状态） |
| `--use_image_obs` | False | 使用图像观察 |
| `--max_episode_steps` | 500 | 每episode最大步数 |
| `--success_threshold` | 0.05 | 成功距离阈值 |
| `--randomize_target` | True | 随机化目标位置 |
| `--sparse_reward` | False | 使用稀疏奖励 |

### PPO超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr_actor` | 3e-4 | Actor学习率 |
| `--lr_critic` | 3e-4 | Critic学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--gae_lambda` | 0.95 | GAE lambda |
| `--clip_epsilon` | 0.2 | PPO clip范围 |
| `--entropy_coef` | 0.01 | 熵系数 |
| `--n_epochs` | 10 | 每次更新的epoch数 |
| `--batch_size` | 64 | 批次大小 |
| `--buffer_size` | 2048 | 经验缓冲区大小 |

## 🎛️ 详细使用指南

### 训练脚本 (train_ppo.py)

**基础训练：**
```bash
python train_ppo.py \
    --robot_type panda \
    --use_simple_env \
    --total_timesteps 500000 \
    --use_wandb \
    --experiment_name "ppo_panda_basic"
```

**高级训练（包含图像）：**
```bash
python train_ppo.py \
    --robot_type panda \
    --use_image_obs \
    --image_size 84 84 \
    --total_timesteps 1000000 \
    --batch_size 32 \
    --buffer_size 4096 \
    --use_wandb \
    --experiment_name "ppo_panda_vision"
```

**多目标训练：**
```bash
python train_ppo.py \
    --robot_type panda \
    --randomize_target \
    --success_threshold 0.03 \
    --max_episode_steps 300 \
    --total_timesteps 1000000
```

### 测试脚本 (test_ppo.py)

**标准测试：**
```bash
python test_ppo.py ./checkpoints/best_model.pt \
    --n_episodes 20 \
    --deterministic \
    --render
```

**保存测试视频：**
```bash
python test_ppo.py ./checkpoints/best_model.pt \
    --n_episodes 5 \
    --save_video \
    --render
```

**交互式测试：**
```bash
python test_ppo.py ./checkpoints/best_model.pt --interactive
```

**性能基准：**
```bash
python test_ppo.py ./checkpoints/best_model.pt \
    --benchmark \
    --n_episodes 100
```

### 集成脚本 (ppo_integration.py)

**单目标执行：**
```bash
python ppo_integration.py \
    --model_path ./checkpoints/best_model.pt \
    --robot_type panda \
    --target 1.4 0.2 0.9 \
    --render
```

**完整演示：**
```bash
python ppo_integration.py --demo
```

## 📈 监控和可视化

### Wandb监控

训练时会自动记录以下指标：
- **Episode统计**: 奖励、长度、成功率
- **训练损失**: 策略损失、价值损失、熵
- **性能指标**: KL散度、解释方差、裁剪比例
- **评估结果**: 定期评估的奖励和成功率

### 本地统计

每次保存检查点时会生成 `stats_*.json` 文件，包含：
- 训练历史数据
- 超参数配置
- 最佳模型信息

## 🎯 训练建议

### 新手建议

1. **从简单开始**: 使用 `--use_simple_env` 先训练状态版本
2. **小批量测试**: 用较小的 `--total_timesteps` 验证设置
3. **监控收敛**: 关注成功率和平均奖励曲线
4. **调整奖励**: 根据任务需求调整奖励函数权重

### 进阶建议

1. **图像训练**: 成功训练状态版本后再尝试图像版本
2. **超参数调优**: 调整学习率、批次大小等
3. **课程学习**: 从简单目标逐渐增加到复杂目标
4. **多环境并行**: 使用多个环境并行训练提高效率

### 典型训练时间

| 配置 | 预计时间 | 成功率目标 |
|------|----------|-----------|
| 简单环境(CPU) | 2-4小时 | >80% |
| 简单环境(GPU) | 30-60分钟 | >80% |
| 图像环境(GPU) | 2-6小时 | >70% |

## 🔧 故障排除

### 常见问题

**1. 训练不收敛**
- 检查奖励函数设计
- 降低学习率
- 增加训练步数
- 检查动作空间范围

**2. 成功率低**
- 调整成功阈值
- 检查目标位置是否合理
- 增加训练时间
- 调整奖励权重

**3. 内存不足**
- 减小 `--batch_size`
- 减小 `--buffer_size`
- 不使用图像观察

**4. 训练速度慢**
- 使用GPU训练
- 减小图像尺寸
- 使用简化环境

### 调试技巧

**查看训练进度：**
```bash
# 检查检查点文件
ls -la checkpoints/

# 查看wandb日志
wandb sync  # 同步本地日志到云端
```

**测试环境设置：**
```python
# 快速测试环境
from rl_grasp_env import RLGraspEnv
env = RLGraspEnv(robot_type='panda', use_image_obs=False)
obs, info = env.reset()
print("Environment test OK")
```

## 🔄 与原系统对比

| 特性 | GraspNet方案 | PPO方案 |
|------|-------------|---------|
| 抓取检测 | 深度学习预测 | 端到端学习 |
| 轨迹规划 | 传统规划算法 | 强化学习策略 |
| 适应性 | 固定规则 | 自适应学习 |
| 实时性 | 较快 | 快速 |
| 鲁棒性 | 依赖感知质量 | 端到端鲁棒 |
| 可扩展性 | 需重新训练检测器 | 可持续学习 |

## 📚 API参考

### RLGraspEnv类

```python
env = RLGraspEnv(
    robot_type='panda',           # 机械臂类型
    image_size=(84, 84),          # 图像尺寸
    max_episode_steps=500,        # 最大步数
    success_distance_threshold=0.05, # 成功阈值
    use_image_obs=True,           # 使用图像观察
    randomize_target=True         # 随机化目标
)
```

### PPOAgent类

```python
agent = PPOAgent(
    observation_space,            # 观察空间
    action_space,                # 动作空间
    lr_actor=3e-4,               # Actor学习率
    lr_critic=3e-4,              # Critic学习率
    device='auto'                # 计算设备
)
```

### PPOGraspController类

```python
controller = PPOGraspController(
    model_path='./best_model.pt', # 模型路径
    robot_type='panda',           # 机械臂类型
    success_threshold=0.05        # 成功阈值
)
```

## 📝 更新日志

- **v1.0**: 初始版本，包含完整PPO训练和测试系统
- 支持Panda和UR5e机械臂
- 支持图像和状态观察
- 集成wandb监控
- 完整的集成示例

## 🤝 贡献指南

如需扩展或改进系统：

1. **添加新机械臂**: 在 `rl_grasp_env.py` 中添加新的环境类
2. **改进奖励函数**: 修改 `_compute_reward` 方法
3. **新的网络架构**: 修改 `network.py` 中的网络定义
4. **算法改进**: 修改 `ppo_agent.py` 中的PPO实现

欢迎提交Issue和Pull Request！