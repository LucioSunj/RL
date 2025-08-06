# 机械臂抓取仿真框架文档

## 项目概述

这是一个基于MuJoCo的机械臂抓取仿真框架，集成了GraspNet深度学习抓取检测和完整的运动规划系统。框架支持Panda和UR5e两种机械臂，为强化学习算法（如PPO）提供了完整的仿真环境。

## 1. 项目架构

```
grasp_lab/
├── main.py                    # Panda机械臂主程序入口
├── main_ur5e.py              # UR5e机械臂主程序入口  
├── graspnet-baseline/         # GraspNet抓取检测模型
├── logs/                      # 训练日志和检查点
├── manipulator_grasp/         # 核心框架代码
│   ├── arm/                   # 机械臂相关模块
│   │   ├── controller/        # 控制器实现
│   │   ├── geometry/          # 几何计算模块
│   │   ├── motion_planning/   # 运动规划系统
│   │   ├── robot/            # 机械臂定义
│   │   └── utils/            # 工具函数
│   ├── assets/               # MuJoCo资产文件
│   │   ├── franka_emika_panda/ # Panda机械臂模型
│   │   ├── universal_robots_ur5e/ # UR5e机械臂模型
│   │   ├── robotiq_2f85/     # Robotiq夹爪模型
│   │   └── scenes/           # 仿真场景配置
│   ├── env/                  # 仿真环境
│   └── utils/                # 环境工具函数
└── README.md
```

## 2. MuJoCo仿真入口和配置

### 2.1 仿真入口

**Panda机械臂入口:**
- 文件: `main.py`
- 环境类: `PandaGraspEnv` (位于 `manipulator_grasp/env/panda_grasp_env.py`)

**UR5e机械臂入口:**
- 文件: `main_ur5e.py`  
- 环境类: `UR5GraspEnv` (位于 `manipulator_grasp/env/ur5_grasp_env.py`)

### 2.2 仿真环境核心配置

```python
# 仿真频率配置
self.sim_hz = 500  # 500Hz仿真频率

# 仿真时间步长
timestep = 0.002   # 2ms时间步长 (在XML文件中配置)

# 相机配置
self.height = 256      # 图像高度
self.width = 256       # 图像宽度
self.fovy = np.pi / 4  # 视野角度
```

### 2.3 MuJoCo场景XML文件配置

**Panda场景配置:**
- 文件: `manipulator_grasp/assets/scenes/scene.xml`
- 包含: Panda机械臂、桌子、目标物体、相机

**UR5e场景配置:**
- 文件: `manipulator_grasp/assets/scenes/scene_ur5.xml`  
- 包含: UR5e机械臂、Robotiq夹爪、桌子、目标物体、多个相机

## 3. 机械臂配置详解

### 3.1 机械臂类层次结构

```python
Robot (基类)              # manipulator_grasp/arm/robot/robot.py
├── Panda                # manipulator_grasp/arm/robot/panda.py
└── UR5e                 # manipulator_grasp/arm/robot/ur5e.py
```

### 3.2 Panda机械臂配置

```python
# 关节数量: 7个关节
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

# 初始关节位置
robot_q = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

# 末端执行器工具坐标系
robot_tool = sm.SE3.Trans(0.0, 0.0, 0.1) * sm.SE3.RPY(-np.pi/2, -np.pi/2, 0.0)

# 动作空间: 8维 (7个关节 + 1个夹爪)
action = np.zeros(8)
```

### 3.3 UR5e机械臂配置

```python
# 关节数量: 6个关节
joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# 初始关节位置
robot_q = np.array([0.0, 0.0, np.pi/2*0, 0.0, -np.pi/2*0, 0.0])

# 末端执行器工具坐标系
robot_tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi/2, -np.pi/2, 0.0)

# 动作空间: 7维 (6个关节 + 1个夹爪)
action = np.zeros(7)
```

## 4. 运动规划系统架构

### 4.1 运动规划层次结构

```
TrajectoryPlanner                    # 轨迹规划器
├── PathPlanner                      # 路径规划器
│   ├── CartesianPlanner            # 笛卡尔空间规划
│   │   ├── PositionPlanner         # 位置规划
│   │   └── AttitudePlanner         # 姿态规划
│   ├── JointPlanner               # 关节空间规划
│   ├── RRTPlanner                 # RRT路径规划
│   └── BlendPlanner               # 混合路径规划
└── VelocityPlanner                 # 速度规划器
    ├── QuinticVelocityPlanner     # 五次多项式速度规划
    └── CubicVelocityPlanner       # 三次多项式速度规划
```

### 4.2 运动规划使用示例

```python
# 1. 关节空间规划
parameter0 = JointParameter(q0, q1)
velocity_parameter0 = QuinticVelocityParameter(time0)
trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
planner0 = TrajectoryPlanner(trajectory_parameter0)

# 2. 笛卡尔空间规划
position_parameter1 = LinePositionParameter(T1.t, T2.t)
attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R))
cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
velocity_parameter1 = QuinticVelocityParameter(time1)
trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
planner1 = TrajectoryPlanner(trajectory_parameter1)

# 3. 轨迹插值执行
planner_interpolate = planner.interpolate(time)
```

## 5. 环境接口规范

### 5.1 环境基本接口

```python
class PandaGraspEnv / UR5GraspEnv:
    def reset(self) -> None:
        """重置环境到初始状态"""
        
    def step(self, action=None) -> None:
        """执行一步仿真
        Args:
            action: 动作数组 [joint1, joint2, ..., gripper]
        """
        
    def render(self) -> dict:
        """渲染并获取观察
        Returns:
            dict: {'img': rgb_image, 'depth': depth_image, 'cam_2': camera2_image}
        """
        
    def close(self) -> None:
        """关闭环境和释放资源"""
```

### 5.2 观察空间

```python
# 图像观察
observation = env.render()
rgb_image = observation['img']        # (256, 256, 3) RGB图像
depth_image = observation['depth']    # (256, 256) 深度图像
camera2_image = observation['cam_2']  # (640, 480, 3) 第二个相机图像 (仅UR5e)

# 状态观察
robot_joint_pos = robot.get_joint()         # 当前关节位置
robot_cartesian_pose = robot.get_cartesian() # 当前末端位置姿态
```

## 6. GraspNet抓取检测集成

### 6.1 抓取检测流程

```python
# 1. 获取神经网络模型
net = get_net()

# 2. 预处理图像数据
end_points, cloud = get_and_process_data(imgs)

# 3. 生成抓取候选
gg = get_grasps(net, end_points)

# 4. 碰撞检测过滤
gg = collision_detection(gg, np.array(cloud.points))

# 5. 非极大值抑制和排序
gg.nms()
gg.sort_by_score()
gg = gg[:1]  # 选择最佳抓取
```

### 6.2 抓取到机械臂坐标转换

```python
# 相机坐标系到世界坐标系的转换
T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))

# 抓取位姿变换
T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(
    sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))

# 目标抓取位姿
T_wo = T_wc * T_co
```

## 7. RL算法集成指南

### 7.1 环境包装建议

为了集成PPO等RL算法，建议创建标准的Gym环境包装:

```python
import gym
from gym import spaces
import numpy as np

class RLGraspEnv(gym.Env):
    def __init__(self, robot_type='panda'):
        super().__init__()
        
        # 选择机械臂类型
        if robot_type == 'panda':
            self.env = PandaGraspEnv()
            self.action_dim = 8  # 7关节 + 1夹爪
        else:
            self.env = UR5GraspEnv() 
            self.action_dim = 7  # 6关节 + 1夹爪
            
        # 定义动作空间和观察空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
            'depth': spaces.Box(low=0, high=10, shape=(256, 256), dtype=np.float32),
            'joint_pos': spaces.Box(low=-np.pi, high=np.pi, shape=(self.action_dim-1,), dtype=np.float32),
            'target_pos': spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32)
        })
        
    def reset(self):
        self.env.reset()
        return self._get_observation()
        
    def step(self, action):
        # 将动作从[-1,1]映射到实际关节空间
        scaled_action = self._scale_action(action)
        self.env.step(scaled_action)
        
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}
        
        return obs, reward, done, info
        
    def _get_observation(self):
        imgs = self.env.render()
        joint_pos = self.env.robot.get_joint()
        
        return {
            'image': imgs['img'],
            'depth': imgs['depth'], 
            'joint_pos': joint_pos,
            'target_pos': self._get_target_position()
        }
        
    def _compute_reward(self):
        # 实现奖励函数
        # 可以基于：
        # 1. 距离目标物体的距离
        # 2. 抓取成功检测
        # 3. 碰撞惩罚
        # 4. 动作平滑性
        pass
        
    def _scale_action(self, action):
        # 将[-1,1]的动作映射到实际的关节空间
        pass
```

### 7.2 状态表示建议

1. **视觉观察**: RGB-D图像 (256×256×4)
2. **本体感知**: 关节位置、关节速度、末端位姿
3. **任务相关**: 目标物体位置、当前抓取状态

### 7.3 动作空间设计

1. **连续动作**: 关节目标位置 + 夹爪开合度
2. **离散动作**: 预定义的运动基元
3. **混合动作**: 高级策略 + 底层控制

### 7.4 奖励函数设计要点

```python
def compute_reward(self):
    reward = 0.0
    
    # 1. 距离奖励 - 鼓励接近目标
    target_distance = np.linalg.norm(end_effector_pos - target_pos)
    distance_reward = -target_distance
    
    # 2. 抓取成功奖励
    if grasp_success:
        grasp_reward = 100.0
    
    # 3. 碰撞惩罚
    if collision_detected:
        collision_penalty = -50.0
    
    # 4. 动作平滑性
    action_smoothness = -np.linalg.norm(action - prev_action)
    
    return distance_reward + grasp_reward + collision_penalty + action_smoothness
```

## 8. 使用说明

### 8.1 环境配置

按照`README.md`中的安装说明配置环境:

```bash
conda create -n graspnet python=3.9
conda activate graspnet
pip install graspnetAPI
pip install trimesh==3.9.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install spatialmath-python mujoco modern-robotics roboticstoolbox-python
pip install numpy==1.23.0
```

### 8.2 运行基本示例

```bash
# 运行Panda机械臂抓取示例
python main.py

# 运行UR5e机械臂抓取示例  
python main_ur5e.py
```

### 8.3 修改仿真配置

**修改机械臂类型:**
- 编辑对应的`main.py`或`main_ur5e.py`
- 更换env实例化: `env = PandaGraspEnv()` 或 `env = UR5GraspEnv()`

**修改场景配置:**
- 编辑XML文件: `manipulator_grasp/assets/scenes/scene.xml` 或 `scene_ur5.xml`
- 可以修改:物体位置、相机位置、光照、材质等

**修改机械臂参数:**
- 编辑对应机械臂类: `manipulator_grasp/arm/robot/panda.py` 或 `ur5e.py`
- 可以修改:DH参数、关节限制、工具坐标系等

## 9. 扩展开发建议

### 9.1 添加新机械臂

1. 在`manipulator_grasp/arm/robot/`中创建新机械臂类
2. 继承`Robot`基类，实现必要的接口方法
3. 在`manipulator_grasp/assets/`中添加对应的URDF/XML模型
4. 创建对应的环境类和场景配置

### 9.2 集成新的抓取算法

1. 在对应位置实现新的抓取检测网络
2. 修改`generate_grasps()`函数以支持新算法
3. 可以创建算法选择机制来动态切换

### 9.3 添加更多传感器

1. 在XML场景文件中定义新的传感器
2. 在环境类的`render()`方法中添加传感器数据获取
3. 更新观察空间定义

这个框架为机械臂抓取任务提供了完整的仿真基础，可以方便地集成各种RL算法进行policy训练。关键是要合理设计状态表示、动作空间和奖励函数。