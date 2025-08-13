"""
改进的抓取环境 - 解决PPO训练收敛问题
主要改进：
1. 简化奖励函数，更稳定的塑形
2. 减少观察空间复杂度
3. 更合理的阶段设计
4. 增加训练稳定性
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import cv2
from typing import Dict, Any, Tuple, Optional
import time
import mujoco

from manipulator_grasp.env.panda_grasp_env import PandaGraspEnv
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv


class ImprovedGraspEnv(gym.Env):
    """
    改进的抓取环境 - 专注于收敛性
    """
    
    def __init__(self, 
                 robot_type='panda',
                 max_episode_steps=1000,  # 减少episode长度
                 use_image_obs=False,  # 默认关闭图像，降低复杂度
                 normalize_obs=True,
                 difficulty='easy',  # 'easy', 'medium', 'hard'
                 headless=True):
        
        super().__init__()
        
        self.robot_type = robot_type
        self.max_episode_steps = max_episode_steps
        self.use_image_obs = use_image_obs
        self.normalize_obs = normalize_obs
        self.difficulty = difficulty
        self.headless = headless
        
        # 初始化环境
        if robot_type == 'panda':
            self.env = PandaGraspEnv()
            self.joint_dim = 7
            self.joint_limits = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
                                        [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525],
                                        [-2.8973, 2.8973]])
        elif robot_type == 'ur5e':
            self.env = UR5GraspEnv()
            self.joint_dim = 6
            self.joint_limits = np.array([[-2*np.pi, 2*np.pi]] * 6)
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")
            
        # 动作空间: 关节增量 + 夹爪控制
        self.action_space = spaces.Box(
            low=-0.03, high=0.03,  # 更小的动作幅度，提高稳定性
            shape=(self.joint_dim + 1,), 
            dtype=np.float32
        )
        
        # 简化的观察空间
        obs_spaces = {}
        
        if self.use_image_obs:
            obs_spaces['image'] = spaces.Box(
                low=0, high=255, 
                shape=(64, 64, 3),  # 更小的图像尺寸
                dtype=np.uint8
            )
        
        # 核心状态观察
        obs_spaces.update({
            'ee_pos': spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
            'ee_to_object': spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),  # 相对向量
            'object_pos': spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
            'object_to_target': spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
            'gripper_state': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'phase_info': spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),  # [approach, grasp, place]
        })
        
        self.observation_space = spaces.Dict(obs_spaces)
        
        # 任务配置 - 根据难度调整
        self._setup_task_config()
        
        # 状态变量
        self.reset_state()
        
    def _setup_task_config(self):
        """根据难度设置任务配置"""
        if self.difficulty == 'easy':
            self.object_initial_pos = np.array([1.3, 0.0, 0.9])
            self.target_pos = np.array([0.7, 0.0, 0.73])
            self.success_threshold = 0.08
            self.grasp_threshold = 0.04
        elif self.difficulty == 'medium':
            self.object_initial_pos = np.array([1.4, 0.2, 0.9])
            self.target_pos = np.array([0.6, -0.2, 0.73])
            self.success_threshold = 0.06
            self.grasp_threshold = 0.03
        else:  # hard
            self.object_initial_pos = np.array([1.4, 0.3, 0.9])
            self.target_pos = np.array([0.5, -0.3, 0.73])
            self.success_threshold = 0.05
            self.grasp_threshold = 0.025
        
        self.object_size = 0.025
        
    def reset_state(self):
        """重置状态变量"""
        self.step_count = 0
        self.prev_joint_pos = None
        self.gripper_state = 0.0
        
        # 简化的任务阶段
        self.phase = 0  # 0: approach, 1: grasp, 2: place
        self.object_grasped = False
        self.task_completed = False
        
        # 连续的奖励塑形变量
        self.last_ee_to_object_dist = float('inf')
        self.last_object_to_target_dist = float('inf')
        self.last_ee_height = 0.0
        
        # 成功标志
        self.grasp_success = False
        self.place_success = False
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置基础环境
        self.env.reset()
        
        # 环境稳定
        for _ in range(100):
            self.env.step()
            
        # 重置状态
        self.reset_state()
        self.prev_joint_pos = self.env.robot.get_joint()
        
        # 初始化距离记录
        ee_pos = self.env.robot.get_cartesian().t
        object_pos = self._get_object_position()
        self.last_ee_to_object_dist = np.linalg.norm(ee_pos - object_pos)
        self.last_object_to_target_dist = np.linalg.norm(object_pos - self.target_pos)
        self.last_ee_height = ee_pos[2]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action: np.ndarray):
        """执行一步动作"""
        self.step_count += 1
        
        # 处理动作
        scaled_action = self._scale_action(action)
        self.env.step(scaled_action)
        
        # 更新夹爪状态（更平滑的控制）
        gripper_delta = action[-1] * 0.05
        self.gripper_state = np.clip(self.gripper_state + gripper_delta, 0.0, 1.0)
        
        # 获取状态
        ee_pos = self.env.robot.get_cartesian().t
        object_pos = self._get_object_position()
        
        # 更新任务状态
        self._update_task_state(ee_pos, object_pos)
        
        # 计算奖励
        reward = self._compute_improved_reward(action, ee_pos, object_pos)
        
        # 检查终止条件
        terminated = self._check_terminated(ee_pos, object_pos)
        truncated = self.step_count >= self.max_episode_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
        
    def _compute_improved_reward(self, action: np.ndarray, ee_pos: np.ndarray, object_pos: np.ndarray) -> float:
        """改进的奖励函数 - 更稳定和连续"""
        reward = 0.0
        
        # 计算关键距离
        ee_to_object_dist = np.linalg.norm(ee_pos - object_pos)
        object_to_target_dist = np.linalg.norm(object_pos[:2] - self.target_pos[:2])
        
        # 1. 阶段性奖励 - 稳定的距离塑形
        if self.phase == 0:  # 接近阶段
            # 鼓励接近物体
            approach_reward = max(0, 1.0 - ee_to_object_dist / 0.5) * 10.0
            reward += approach_reward
            
            # 进度奖励
            if ee_to_object_dist < self.last_ee_to_object_dist:
                reward += (self.last_ee_to_object_dist - ee_to_object_dist) * 50.0
            
        elif self.phase == 1:  # 抓取阶段
            # 接近 + 夹爪控制奖励
            grasp_reward = max(0, 1.0 - ee_to_object_dist / 0.1) * 20.0
            reward += grasp_reward
            
            # 夹爪状态奖励
            if ee_to_object_dist < self.grasp_threshold:
                reward += self.gripper_state * 15.0
                
            # 抓取成功奖励
            if self.object_grasped and not self.grasp_success:
                reward += 100.0
                self.grasp_success = True
                
        elif self.phase == 2:  # 放置阶段
            # 鼓励保持抓取
            if self.object_grasped:
                reward += 5.0
                
                # 搬运到目标奖励
                transport_reward = max(0, 1.0 - object_to_target_dist / 1.0) * 15.0
                reward += transport_reward
                
                # 进度奖励
                if object_to_target_dist < self.last_object_to_target_dist:
                    reward += (self.last_object_to_target_dist - object_to_target_dist) * 30.0
                
                # 到达目标区域后鼓励松开夹爪
                if object_to_target_dist < self.success_threshold:
                    reward += (1.0 - self.gripper_state) * 10.0
            else:
                # 失去抓取的惩罚
                reward -= 20.0
        
        # 2. 通用奖励
        # 高度保持奖励（防止撞击地面）
        if ee_pos[2] > 0.5:
            reward += 1.0
        else:
            reward -= 10.0
            
        # 动作平滑性（更温和的惩罚）
        action_penalty = np.sum(np.square(action)) * 0.5
        reward -= action_penalty
        
        # 时间效率奖励
        reward -= 0.1
        
        # 3. 任务完成奖励
        if self.task_completed and not self.place_success:
            reward += 200.0
            self.place_success = True
        
        # 更新历史记录
        self.last_ee_to_object_dist = ee_to_object_dist
        self.last_object_to_target_dist = object_to_target_dist
        self.last_ee_height = ee_pos[2]
        
        # 奖励范围限制
        reward = np.clip(reward, -50.0, 250.0)
        
        return reward
        
    def _update_task_state(self, ee_pos: np.ndarray, object_pos: np.ndarray):
        """更新任务状态 - 简化的阶段转换"""
        ee_to_object_dist = np.linalg.norm(ee_pos - object_pos)
        object_to_target_dist = np.linalg.norm(object_pos[:2] - self.target_pos[:2])
        
        # 检查是否抓取到物体
        if (ee_to_object_dist < self.grasp_threshold and 
            self.gripper_state > 0.6 and 
            object_pos[2] > self.object_initial_pos[2] + 0.02):
            self.object_grasped = True
        
        # 检查是否丢失物体
        if self.object_grasped and object_pos[2] < self.object_initial_pos[2] - 0.05:
            self.object_grasped = False
        
        # 阶段转换
        if self.phase == 0 and ee_to_object_dist < self.grasp_threshold * 2:
            self.phase = 1
        elif self.phase == 1 and self.object_grasped:
            self.phase = 2
        
        # 检查任务完成
        if (self.phase == 2 and 
            object_to_target_dist < self.success_threshold and 
            self.gripper_state < 0.4 and 
            object_pos[2] > self.target_pos[2] - 0.1):
            self.task_completed = True
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取观察"""
        obs = {}
        
        # 图像观察（如果需要）
        if self.use_image_obs:
            imgs = self.env.render()
            rgb_img = imgs['img']
            rgb_resized = cv2.resize(rgb_img, (64, 64))
            obs['image'] = rgb_resized.astype(np.uint8)
        
        # 状态观察
        ee_pos = self.env.robot.get_cartesian().t
        object_pos = self._get_object_position()
        
        # 相对向量（关键特征）
        ee_to_object = object_pos - ee_pos
        object_to_target = self.target_pos - object_pos
        
        # 阶段信息（one-hot编码）
        phase_info = np.zeros(3)
        phase_info[self.phase] = 1.0
        
        obs.update({
            'ee_pos': ee_pos.astype(np.float32),
            'ee_to_object': ee_to_object.astype(np.float32),
            'object_pos': object_pos.astype(np.float32),
            'object_to_target': object_to_target.astype(np.float32),
            'gripper_state': np.array([self.gripper_state], dtype=np.float32),
            'phase_info': phase_info.astype(np.float32)
        })
        
        return obs
        
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """动作缩放"""
        joint_increments = action[:-1]
        
        # 获取当前关节位置
        current_joints = self.env.robot.get_joint()
        target_joints = current_joints + joint_increments
        
        # 关节限制
        for i, (low, high) in enumerate(self.joint_limits):
            target_joints[i] = np.clip(target_joints[i], low, high)
        
        # 构造环境动作
        if self.robot_type == 'panda':
            full_action = np.zeros(8)
            full_action[:7] = target_joints
            full_action[7] = self.gripper_state * 255
        else:  # ur5e
            full_action = np.zeros(7)
            full_action[:6] = target_joints
            full_action[6] = self.gripper_state * 255
            
        return full_action
        
    def _get_object_position(self) -> np.ndarray:
        """获取物体位置"""
        try:
            body_id = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "Box")
            object_pos = self.env.mj_data.xpos[body_id].copy()
            return object_pos
        except:
            return self.object_initial_pos.copy()
    
    def _check_terminated(self, ee_pos: np.ndarray, object_pos: np.ndarray) -> bool:
        """检查终止条件"""
        # 成功完成
        if self.task_completed:
            return True
            
        # 失败条件
        if (ee_pos[2] < 0.2 or  # 末端执行器太低
            object_pos[2] < 0.4 or  # 物体掉落
            not self._is_in_workspace(ee_pos)):  # 超出工作空间
            return True
            
        return False
        
    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        """检查是否在工作空间内"""
        return (0.3 <= pos[0] <= 1.6 and
                -0.6 <= pos[1] <= 0.6 and
                0.2 <= pos[2] <= 1.5)
        
    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        ee_pos = self.env.robot.get_cartesian().t
        object_pos = self._get_object_position()
        
        return {
            'step_count': self.step_count,
            'phase': self.phase,
            'object_grasped': self.object_grasped,
            'task_completed': self.task_completed,
            'ee_to_object_dist': np.linalg.norm(ee_pos - object_pos),
            'object_to_target_dist': np.linalg.norm(object_pos - self.target_pos),
            'success': self.task_completed,
            'ee_position': ee_pos,
            'object_position': object_pos,
            'gripper_state': self.gripper_state
        }
        
    def close(self):
        """关闭环境"""
        self.env.close()
        
    def render(self):
        """渲染环境"""
        return self.env.render()


if __name__ == '__main__':
    # 测试改进的环境
    print("Testing ImprovedGraspEnv...")
    
    env = ImprovedGraspEnv(robot_type='panda', difficulty='easy', use_image_obs=False)
    
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    obs, info = env.reset()
    print("Initial info:", info)
    
    # 随机动作测试
    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 50 == 0:
            print(f"Step {step}: Phase={info['phase']}, Reward={reward:.2f}, "
                  f"EE-Obj={info['ee_to_object_dist']:.3f}, "
                  f"Obj-Target={info['object_to_target_dist']:.3f}")
            
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            print(f"Final info: {info}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.close()
    print("Environment test completed!")