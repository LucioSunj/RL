import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import cv2
from typing import Dict, Any, Tuple, Optional
import time

from manipulator_grasp.env.panda_grasp_env import PandaGraspEnv
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv


class RLGraspEnv(gym.Env):
    """
    RL训练用的抓取环境
    将原有的GraspNet+轨迹规划替换为端到端的RL policy
    """
    
    def __init__(self, 
                 robot_type='panda',
                 image_size=(84, 84),
                 max_episode_steps=500,
                 success_distance_threshold=0.05,
                 collision_penalty_threshold=0.02,
                 use_image_obs=True,
                 normalize_obs=True,
                 randomize_target=True,
                 sparse_reward=False,
                 headless=True):
        
        super().__init__()
        
        self.robot_type = robot_type
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.success_distance_threshold = success_distance_threshold
        self.collision_penalty_threshold = collision_penalty_threshold
        self.use_image_obs = use_image_obs
        self.normalize_obs = normalize_obs
        self.randomize_target = randomize_target
        self.sparse_reward = sparse_reward
        self.headless = headless
        
        # 初始化对应的环境
        if robot_type == 'panda':
            self.env = PandaGraspEnv(headless=headless)
            self.joint_dim = 7
            self.joint_limits = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
                                        [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525],
                                        [-2.8973, 2.8973]])
        elif robot_type == 'ur5e':
            self.env = UR5GraspEnv()  # TODO: 添加headless支持
            self.joint_dim = 6
            self.joint_limits = np.array([[-2*np.pi, 2*np.pi]] * 6)
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")
            
        # 动作空间: 关节增量控制 + 夹爪控制
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, 
            shape=(self.joint_dim + 1,), 
            dtype=np.float32
        )
        
        # 观察空间
        obs_spaces = {}
        
        if self.use_image_obs:
            obs_spaces['image'] = spaces.Box(
                low=0, high=255, 
                shape=(*self.image_size, 3), 
                dtype=np.uint8
            )
            obs_spaces['depth'] = spaces.Box(
                low=0, high=2.0, 
                shape=self.image_size, 
                dtype=np.float32
            )
        
        obs_spaces.update({
            'joint_pos': spaces.Box(
                low=-1.0, high=1.0, 
                shape=(self.joint_dim,), 
                dtype=np.float32
            ),
            'joint_vel': spaces.Box(
                low=-1.0, high=1.0, 
                shape=(self.joint_dim,), 
                dtype=np.float32
            ),
            'ee_pos': spaces.Box(
                low=-1.0, high=1.0, 
                shape=(3,), 
                dtype=np.float32
            ),
            'target_relative': spaces.Box(
                low=-1.0, high=1.0, 
                shape=(3,), 
                dtype=np.float32
            ),
            'gripper_state': spaces.Box(
                low=0.0, high=1.0, 
                shape=(1,), 
                dtype=np.float32
            )
        })
        
        self.observation_space = spaces.Dict(obs_spaces)
        
        # 状态变量
        self.step_count = 0
        self.prev_joint_pos = None
        self.target_position = None
        self.initial_ee_pos = None
        self.best_distance = float('inf')
        self.gripper_state = 0.0
        
        # 统计信息
        self.episode_reward = 0.0
        self.success_count = 0
        self.total_episodes = 0
        
        # 工作空间限制
        self.workspace_bounds = {
            'x': [0.3, 1.5],
            'y': [-0.6, 0.6], 
            'z': [0.0, 1.5]
        }
        
        # 目标位置候选
        self.target_candidates = [
            np.array([1.4, 0.2, 0.9]),
            np.array([1.0, 0.3, 0.85]),
            np.array([1.2, -0.2, 0.95]),
            np.array([0.8, 0.4, 0.8]),
            np.array([1.3, -0.3, 0.9])
        ]
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置基础环境
        self.env.reset()
        
        # 让环境稳定
        for _ in range(100):
            self.env.step()
            
        # 重置状态
        self.step_count = 0
        self.prev_joint_pos = self.env.robot.get_joint()
        self.initial_ee_pos = self.env.robot.get_cartesian().t
        self.best_distance = float('inf')
        self.episode_reward = 0.0
        self.gripper_state = 0.0
        
        # 设置目标位置
        if self.randomize_target:
            self.target_position = self._get_random_target()
        else:
            self.target_position = self.target_candidates[0]
            
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action: np.ndarray):
        """执行一步动作"""
        self.step_count += 1
        
        # 动作处理和执行
        scaled_action = self._scale_action(action)
        self.env.step(scaled_action)
        
        # 更新夹爪状态
        self.gripper_state = np.clip(self.gripper_state + action[-1] * 0.1, 0.0, 1.0)
        
        # 获取新状态
        observation = self._get_observation()
        reward = self._compute_reward(action)
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_episode_steps
        info = self._get_info()
        
        self.episode_reward += reward
        
        if terminated or truncated:
            if terminated and info['success']:
                self.success_count += 1
            self.total_episodes += 1
            info['episode_reward'] = self.episode_reward
            info['success_rate'] = self.success_count / max(self.total_episodes, 1)
            
        return observation, reward, terminated, truncated, info
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取观察"""
        obs = {}
        
        # 图像观察
        if self.use_image_obs:
            imgs = self.env.render()
            
            # RGB图像
            rgb_img = imgs['img']
            rgb_resized = cv2.resize(rgb_img, self.image_size)
            obs['image'] = rgb_resized.astype(np.uint8)
            
            # 深度图像
            depth_img = imgs['depth']
            depth_resized = cv2.resize(depth_img, self.image_size)
            depth_normalized = np.clip(depth_resized, 0, 2.0)
            obs['depth'] = depth_normalized.astype(np.float32)
        
        # 机械臂状态
        joint_pos = self.env.robot.get_joint()
        ee_transform = self.env.robot.get_cartesian()
        ee_pos = ee_transform.t
        
        # 计算关节速度
        if self.prev_joint_pos is not None:
            joint_vel = (joint_pos - self.prev_joint_pos) / 0.002
        else:
            joint_vel = np.zeros_like(joint_pos)
        self.prev_joint_pos = joint_pos.copy()
        
        # 相对目标位置
        target_relative = self.target_position - ee_pos
        
        # 状态归一化
        if self.normalize_obs:
            joint_pos_norm = self._normalize_joint_pos(joint_pos)
            joint_vel_norm = np.clip(joint_vel / 10.0, -1.0, 1.0)
            ee_pos_norm = self._normalize_position(ee_pos)
            target_relative_norm = self._normalize_position(target_relative, center_zero=True)
        else:
            joint_pos_norm = joint_pos
            joint_vel_norm = joint_vel
            ee_pos_norm = ee_pos
            target_relative_norm = target_relative
        
        obs.update({
            'joint_pos': joint_pos_norm.astype(np.float32),
            'joint_vel': joint_vel_norm.astype(np.float32),
            'ee_pos': ee_pos_norm.astype(np.float32),
            'target_relative': target_relative_norm.astype(np.float32),
            'gripper_state': np.array([self.gripper_state], dtype=np.float32)
        })
        
        return obs
        
    def _compute_reward(self, action: np.ndarray) -> float:
        """计算奖励函数"""
        reward = 0.0
        
        # 获取当前末端位置
        ee_pos = self.env.robot.get_cartesian().t
        target_distance = np.linalg.norm(ee_pos - self.target_position)
        
        if self.sparse_reward:
            # 稀疏奖励：只在成功时给奖励
            if target_distance < self.success_distance_threshold:
                reward = 100.0
            else:
                reward = -0.1  # 时间惩罚
        else:
            # 密集奖励
            # 1. 距离奖励 (主要驱动力)
            max_distance = 2.0
            distance_reward = (max_distance - target_distance) / max_distance * 10.0
            reward += distance_reward
            
            # 2. 改进奖励 (鼓励持续改进)
            if target_distance < self.best_distance:
                improvement_reward = (self.best_distance - target_distance) * 50.0
                reward += improvement_reward
                self.best_distance = target_distance
                
            # 3. 成功奖励
            if target_distance < self.success_distance_threshold:
                success_reward = 100.0
                reward += success_reward
                
            # 4. 动作平滑性惩罚
            action_penalty = np.sum(np.square(action)) * 0.1
            reward -= action_penalty
            
            # 5. 关节限制惩罚
            joint_pos = self.env.robot.get_joint()
            for i, (pos, (low, high)) in enumerate(zip(joint_pos, self.joint_limits)):
                if pos < low + 0.1 or pos > high - 0.1:
                    reward -= 5.0
                    
            # 6. 工作空间限制惩罚
            if not self._is_in_workspace(ee_pos):
                reward -= 10.0
                
            # 7. 碰撞惩罚 (简化版本)
            if ee_pos[2] < 0.05:  # 太接近地面
                reward -= 20.0
                
            # 8. 时间惩罚 (鼓励效率)
            reward -= 0.1
        
        return reward
        
    def _check_terminated(self) -> bool:
        """检查是否终止"""
        ee_pos = self.env.robot.get_cartesian().t
        target_distance = np.linalg.norm(ee_pos - self.target_position)
        
        # 成功条件
        if target_distance < self.success_distance_threshold:
            return True
            
        # 失败条件：超出工作空间
        if not self._is_in_workspace(ee_pos):
            return True
            
        # 失败条件：碰撞
        if ee_pos[2] < 0.0:  # 撞到地面
            return True
            
        return False
        
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """将RL动作转换为环境动作"""
        joint_increments = action[:-1]
        gripper_increment = action[-1]
        
        # 获取当前关节位置
        current_joints = self.env.robot.get_joint()
        
        # 计算目标关节位置
        target_joints = current_joints + joint_increments
        
        # 关节限制
        for i, (low, high) in enumerate(self.joint_limits):
            target_joints[i] = np.clip(target_joints[i], low, high)
        
        # 构造完整动作
        if self.robot_type == 'panda':
            full_action = np.zeros(8)
            full_action[:7] = target_joints
            full_action[7] = self.gripper_state * 255  # 当前夹爪状态
        else:  # ur5e
            full_action = np.zeros(7)
            full_action[:6] = target_joints
            full_action[6] = self.gripper_state * 255
            
        return full_action
        
    def _get_random_target(self) -> np.ndarray:
        """生成随机目标位置"""
        if np.random.random() < 0.7:  # 70%概率选择预设目标
            return np.random.choice(self.target_candidates).copy()
        else:  # 30%概率完全随机
            x = np.random.uniform(*self.workspace_bounds['x'])
            y = np.random.uniform(*self.workspace_bounds['y'])
            z = np.random.uniform(0.5, 1.2)
            return np.array([x, y, z])
        
    def _normalize_joint_pos(self, joint_pos: np.ndarray) -> np.ndarray:
        """归一化关节位置到[-1, 1]"""
        normalized = np.zeros_like(joint_pos)
        for i, (pos, (low, high)) in enumerate(zip(joint_pos, self.joint_limits)):
            normalized[i] = 2 * (pos - low) / (high - low) - 1
        return normalized
        
    def _normalize_position(self, pos: np.ndarray, center_zero=False) -> np.ndarray:
        """归一化位置"""
        if center_zero:
            # 对于相对位置，直接除以最大可能距离
            return np.clip(pos / 2.0, -1.0, 1.0)
        else:
            # 对于绝对位置，映射到工作空间
            x_norm = 2 * (pos[0] - self.workspace_bounds['x'][0]) / (self.workspace_bounds['x'][1] - self.workspace_bounds['x'][0]) - 1
            y_norm = 2 * (pos[1] - self.workspace_bounds['y'][0]) / (self.workspace_bounds['y'][1] - self.workspace_bounds['y'][0]) - 1
            z_norm = 2 * (pos[2] - self.workspace_bounds['z'][0]) / (self.workspace_bounds['z'][1] - self.workspace_bounds['z'][0]) - 1
            return np.array([x_norm, y_norm, z_norm])
        
    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        """检查位置是否在工作空间内"""
        return (self.workspace_bounds['x'][0] <= pos[0] <= self.workspace_bounds['x'][1] and
                self.workspace_bounds['y'][0] <= pos[1] <= self.workspace_bounds['y'][1] and
                self.workspace_bounds['z'][0] <= pos[2] <= self.workspace_bounds['z'][1])
        
    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        ee_pos = self.env.robot.get_cartesian().t
        target_distance = np.linalg.norm(ee_pos - self.target_position)
        
        return {
            'target_distance': target_distance,
            'success': target_distance < self.success_distance_threshold,
            'step_count': self.step_count,
            'ee_position': ee_pos,
            'target_position': self.target_position,
            'best_distance': self.best_distance,
            'gripper_state': self.gripper_state,
            'in_workspace': self._is_in_workspace(ee_pos)
        }
        
    def close(self):
        """关闭环境"""
        self.env.close()
        
    def render(self):
        """渲染环境"""
        return self.env.render()


class SimpleRLGraspEnv(RLGraspEnv):
    """
    简化版RL环境，只使用状态信息，不使用图像
    适合快速原型开发和调试
    """
    
    def __init__(self, **kwargs):
        kwargs['use_image_obs'] = False
        super().__init__(**kwargs)


if __name__ == '__main__':
    # 测试环境
    print("Testing RLGraspEnv...")
    env = RLGraspEnv(robot_type='panda', use_image_obs=True)
    
    obs, info = env.reset()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation keys:", obs.keys())
    if 'image' in obs:
        print("Image shape:", obs['image'].shape)
    print("Initial info:", info)
    
    # 随机动作测试
    total_reward = 0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 10 == 0:
            print(f"Step {i}: reward={reward:.3f}, distance={info['target_distance']:.3f}, "
                  f"success={info['success']}")
            
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            print(f"Total reward: {total_reward:.3f}")
            print(f"Final info: {info}")
            break
    
    env.close()
    print("Environment test completed!")