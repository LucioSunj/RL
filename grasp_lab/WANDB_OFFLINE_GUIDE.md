# Wandb离线模式使用指南

## 📖 概述

Wandb的离线模式允许你在没有网络连接或不想上传数据到云端的情况下进行实验记录。所有的日志和模型都会保存在本地，之后可以选择性地同步到云端。

## 🚀 使用方法

### 方法1：默认离线模式（推荐）

现在代码默认使用离线模式，直接运行即可：

```bash
# 默认使用离线模式
python train_ppo.py --use_simple_env --robot_type panda --total_timesteps 200000

# 或者明确指定离线模式
python train_ppo.py --use_simple_env --robot_type panda --wandb_offline
```

### 方法2：切换到在线模式

如果需要在线同步，使用 `--wandb_online` 参数：

```bash
python train_ppo.py --use_simple_env --robot_type panda --wandb_online
```

### 方法3：完全禁用wandb

如果不想使用任何日志记录：

```bash
python train_ppo.py --use_simple_env --robot_type panda --no-use_wandb
```

### 方法4：环境变量方式

也可以通过环境变量控制：

```bash
# 离线模式
export WANDB_MODE=offline
python train_ppo.py --use_simple_env --robot_type panda

# 禁用wandb
export WANDB_MODE=disabled
python train_ppo.py --use_simple_env --robot_type panda
```

## 📁 离线数据位置

离线模式下，数据保存在以下位置：

```
./wandb/
├── offline-run-{timestamp}/
│   ├── run-{id}.wandb          # 运行数据
│   ├── config.yaml             # 配置文件
│   ├── requirements.txt        # 依赖列表
│   └── files/                  # 代码快照
│       ├── train_ppo.py
│       ├── network.py
│       └── ...
└── settings                    # wandb设置
```

## 🔄 数据同步

### 查看离线runs

```bash
# 查看所有离线runs
wandb offline

# 查看特定目录的runs
wandb offline --directory ./wandb
```

### 同步到云端

```bash
# 同步所有离线runs
wandb sync ./wandb/offline-run-*

# 同步特定run
wandb sync ./wandb/offline-run-20241201_123456

# 批量同步
find ./wandb -name "offline-run-*" -exec wandb sync {} \;
```

### 选择性同步

```bash
# 只同步成功的实验
wandb sync ./wandb/offline-run-* --include-synced

# 排除某些文件
wandb sync ./wandb/offline-run-* --exclude="*.pt,*.pth"
```

## 📊 本地可视化

### 使用wandb本地服务器

```bash
# 启动本地wandb服务器
wandb server

# 指定端口
wandb server --port 8080
```

然后在浏览器中访问 `http://localhost:8080` 查看实验结果。

### 导出数据

```bash
# 导出为CSV
wandb export <project_name> --format csv

# 导出为JSON
wandb export <project_name> --format json
```

## ⚙️ 配置选项

### 在代码中配置

```python
import wandb

# 离线模式
wandb.init(mode="offline")

# 禁用模式
wandb.init(mode="disabled")

# 在线模式（默认）
wandb.init(mode="online")
```

### 全局配置

```bash
# 设置默认为离线模式
wandb config --set mode offline

# 查看当前配置
wandb config --show
```

## 🎯 使用场景

### 适合离线模式的场景

1. **服务器训练**: 无网络或网络受限的服务器环境
2. **大量实验**: 避免频繁上传大量数据
3. **敏感数据**: 不想将实验数据上传到云端
4. **调试阶段**: 频繁测试时避免污染云端项目

### 离线模式的优势

- ✅ 无需网络连接
- ✅ 数据完全本地化
- ✅ 上传速度不影响训练
- ✅ 可选择性同步重要实验
- ✅ 支持所有wandb功能

### 离线模式的限制

- ❌ 无法实时查看云端dashboard  
- ❌ 无法与团队实时共享
- ❌ 需要手动管理本地存储空间
- ❌ 断电可能丢失未同步数据

## 🔧 故障排除

### 常见问题

**1. 离线run文件太大**
```bash
# 清理大文件
find ./wandb -name "*.wandb" -size +100M -delete

# 压缩旧的runs
tar -czf wandb_backup.tar.gz ./wandb/offline-run-*
```

**2. 同步失败**
```bash
# 检查网络连接
wandb login

# 重新登录
wandb logout
wandb login

# 验证同步
wandb sync --test
```

**3. 磁盘空间不足**
```bash
# 查看wandb占用空间
du -sh ./wandb

# 清理老旧的runs
wandb sweep --cleanup ./wandb
```

## 📈 性能对比

| 模式 | 启动速度 | 日志速度 | 网络依赖 | 存储位置 |
|------|----------|----------|----------|----------|
| 离线 | 快 | 快 | 无 | 本地 |
| 在线 | 中等 | 中等 | 是 | 云端+本地 |
| 禁用 | 最快 | 最快 | 无 | 仅本地文件 |

## 💡 最佳实践

1. **开发阶段**: 使用离线模式进行快速迭代
2. **重要实验**: 完成后选择性同步到云端
3. **长期存储**: 定期备份和清理本地wandb数据
4. **团队协作**: 建立离线数据的共享和同步流程

## 🎮 快速测试

创建一个简单的测试脚本：

```python
import wandb
import time
import random

# 离线模式测试
wandb.init(
    project="test_offline",
    mode="offline",
    config={"learning_rate": 0.01}
)

for i in range(100):
    wandb.log({
        "loss": random.random(),
        "accuracy": random.random(),
        "step": i
    })
    time.sleep(0.1)

wandb.finish()
print("离线测试完成！数据保存在 ./wandb/ 目录")
```

现在你可以安全地在离线环境中进行PPO训练，所有的实验数据都会本地保存，需要时再选择性地同步到云端！