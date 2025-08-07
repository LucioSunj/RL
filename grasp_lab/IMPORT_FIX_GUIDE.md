# 导入问题修复指南

## 问题概述

项目中存在大量使用绝对导入 `from arm.xxx` 的地方，需要改为相对导入。

## 已修复的文件

✅ `manipulator_grasp/arm/robot/robot.py`
✅ `manipulator_grasp/arm/robot/panda.py`  
✅ `manipulator_grasp/arm/robot/ur5e.py`
✅ `manipulator_grasp/arm/geometry/simplex/geometry.py`
✅ `manipulator_grasp/arm/motion_planning/__init__.py`
✅ `manipulator_grasp/arm/utils/math_utils.py`
✅ `manipulator_grasp/arm/interface/strategy.py`

## 修复规则

| 原导入 | 修改为 |
|--------|---------|
| `from arm.geometry import xxx` | `from ..geometry import xxx` 或 `from ..geometry.xxx import yyy` |
| `from arm.constanst import xxx` | `from ..constanst import xxx` 或更多层级的 `...constanst` |
| `from arm.interface import xxx` | `from ..interface import xxx` |
| `from arm.utils import xxx` | `from ..utils import xxx` |
| `from arm.robot import xxx` | `from ..robot import xxx` |
| `Geometry3D` | `Geometry` (类名替换) |

## 手动修复脚本

如果你可以在服务器上运行Python脚本，可以使用以下命令：

```bash
# 运行修复脚本
python fix_imports.py

# 测试导入
python test_imports.py

# 如果测试通过，运行训练
python train_ppo.py --use_simple_env --robot_type panda --total_timesteps 200000
```

## 替代解决方案

如果导入问题太多，你也可以：

1. **临时环境变量方法**：
   ```python
   import sys
   import os
   sys.path.append(os.path.join(os.path.dirname(__file__), 'manipulator_grasp'))
   ```

2. **使用__init__.py导入**：
   在 `manipulator_grasp/__init__.py` 中添加：
   ```python
   import sys
   import os
   
   # 添加arm模块到路径
   arm_path = os.path.join(os.path.dirname(__file__), 'arm')
   if arm_path not in sys.path:
       sys.path.insert(0, arm_path)
   ```

## 快速测试

运行以下代码测试是否修复成功：

```python
try:
    from manipulator_grasp.env.panda_grasp_env import PandaGraspEnv
    from rl_grasp_env import RLGraspEnv
    print("✅ 导入修复成功！")
except ImportError as e:
    print(f"❌ 还有导入问题: {e}")
```

## 如果仍有问题

如果修复后仍有问题，可能需要修复更多文件。最常见的剩余问题：

1. **cartesian_planner.py**:
   ```python
   # 修改前
   from arm.interface import ModeEnum
   from arm.geometry import SE3Impl
   
   # 修改后  
   from ...interface import ModeEnum
   from ...geometry import SE3Impl
   ```

2. **time_optimal_planner.py**:
   ```python
   # 修改前
   from arm.robot import Robot
   
   # 修改后
   from ...robot import Robot
   ```

## 最后的解决方案

如果以上都不工作，作为最后的解决方案，你可以在每个有问题的文件开头添加：

```python
import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加arm目录到路径  
arm_dir = os.path.join(project_root, 'grasp_lab', 'manipulator_grasp', 'arm')
if arm_dir not in sys.path:
    sys.path.insert(0, arm_dir)
```

然后保持原来的 `from arm.xxx` 导入不变。

## 训练命令

修复完成后，使用以下命令开始训练：

```bash
# 简单环境（推荐先试这个）
python train_ppo.py --use_simple_env --robot_type panda --total_timesteps 200000

# 完整环境
python train_ppo.py --use_image_obs --robot_type panda --total_timesteps 500000
```