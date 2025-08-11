"""
完整抓取任务环境
实现与原始GraspNet+规划系统相同的抓取-搬运-放置任务
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


class GraspTaskEnv(gym.Env):
    """
    完整抓取任务环境
    任务：抓取立方体并放置到指定位置
    """
    
    def __init__(self, 
                 robot_type='panda',
                 image_size=(84, 84),
                 max_episode_steps=2000,  # 增加步数以完成完整任务
                 use_image_obs=True,
                 normalize_obs=True,
                 randomize_object=False,  # 是否随机化物体位置
                 sparse_reward=False,
                 headless=True):
        
        super().__init__()
        
        self.robot_type = robot_type
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.use_image_obs = use_image_obs
        self.normalize_obs = normalize_obs
        self.randomize_object = randomize_object
        self.sparse_reward = sparse_reward
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
            low=-0.05, high=0.05,  # 减小动作幅度以提高精度
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
        
        # 状态观察
        obs_spaces.update({
            'joint_pos': spaces.Box(low=-1.0, high=1.0, shape=(self.joint_dim,), dtype=np.float32),
            'joint_vel': spaces.Box(low=-1.0, high=1.0, shape=(self.joint_dim,), dtype=np.float32),
            'ee_pos': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'ee_quat': spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            'gripper_state': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'object_pos': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'object_in_gripper': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'task_phase': spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),  # one-hot编码当前阶段
            'target_pos': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'relative_pos': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        })
        
        self.observation_space = spaces.Dict(obs_spaces)
        
        # 任务设定
        self.object_initial_pos = np.array([1.4, 0.2, 0.9])  # 立方体初始位置
        self.pickup_zone = np.array([1.4, 0.6, 0.73])        # 抓取区域中心
        self.drop_zone = np.array([0.2, 0.2, 0.73])          # 放置区域中心
        self.object_size = 0.025  # 立方体边长
        
        # 任务阶段定义
        self.PHASE_APPROACH = 0    # 接近物体
        self.PHASE_GRASP = 1       # 抓取物体
        self.PHASE_LIFT = 2        # 抬起物体
        self.PHASE_TRANSPORT = 3   # 搬运物体
        self.PHASE_PLACE = 4       # 放置物体
        
        # 状态变量
        self.reset_state()
        
    def reset_state(self):
        """重置所有状态变量"""
        self.step_count = 0
        self.current_phase = self.PHASE_APPROACH
        self.prev_joint_pos = None
        self.gripper_state = 0.0  # 0=张开, 1=闭合
        self.object_grasped = False
        self.object_lifted = False
        self.task_completed = False
        
        # 阶段特定目标
        self.approach_target = self.object_initial_pos + np.array([0, 0, 0.1])  # 物体上方10cm
        self.grasp_target = self.object_initial_pos.copy()                       # 抓取位置
        self.lift_target = self.object_initial_pos + np.array([0, 0, 0.15])     # 抬起位置
        self.transport_target = self.drop_zone + np.array([0, 0, 0.15])         # 搬运目标上方
        self.place_target = self.drop_zone + np.array([0, 0, 0.05])             # 放置位置
        
        # 当前目标
        self.current_target = self.approach_target.copy()
        
        # 距离记录 - 使用大但有限的值，避免inf
        self.best_distance_to_target = 1000.0  # 大但有限的初始值
        self.best_distance_to_object = 1000.0

        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置基础环境
        self.env.reset()
        
        # 环境稳定
        for _ in range(200):
            self.env.step()
            
        # 重置状态
        self.reset_state()
        self.prev_joint_pos = self.env.robot.get_joint()
        
        # 随机化物体位置（可选）
        if self.randomize_object:
            self.object_initial_pos = self._randomize_object_position()
            self._update_phase_targets()
            
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action: np.ndarray):
        """执行一步动作"""
        self.step_count += 1
        
        # 处理动作
        scaled_action = self._scale_action(action)
        self.env.step(scaled_action)
        
        # 更新夹爪状态
        gripper_action = action[-1]
        self.gripper_state = np.clip(self.gripper_state + gripper_action * 0.1, 0.0, 1.0)
        
        # 获取物体和机械臂状态
        ee_pos = self.env.robot.get_cartesian().t
        object_pos = self._get_object_position()
        
        # 更新任务阶段
        self._update_task_phase(ee_pos, object_pos)
        
        # 计算奖励
        reward = self._compute_reward(action, ee_pos, object_pos)
        
        # 检查终止条件
        terminated = self._check_terminated(ee_pos, object_pos)
        truncated = self.step_count >= self.max_episode_steps
        
        # 获取观察和信息
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取观察"""
        obs = {}
        
        # 图像观察
        if self.use_image_obs:
            imgs = self.env.render()
            
            rgb_img = imgs['img']
            rgb_resized = cv2.resize(rgb_img, self.image_size)
            obs['image'] = rgb_resized.astype(np.uint8)
            
            depth_img = imgs['depth']
            depth_resized = cv2.resize(depth_img, self.image_size)
            depth_normalized = np.clip(depth_resized, 0, 2.0)
            obs['depth'] = depth_normalized.astype(np.float32)
        
        # 机械臂状态
        joint_pos = self.env.robot.get_joint()
        ee_transform = self.env.robot.get_cartesian()
        ee_pos = ee_transform.t
        ee_quat = ee_transform.UnitQuaternion().vec  # [w, x, y, z]
        
        # 计算关节速度
        if self.prev_joint_pos is not None:
            joint_vel = (joint_pos - self.prev_joint_pos) / 0.002
        else:
            joint_vel = np.zeros_like(joint_pos)
        self.prev_joint_pos = joint_pos.copy()
        
        # 物体状态
        object_pos = self._get_object_position()
        
        # 任务阶段编码
        phase_encoding = np.zeros(5)
        phase_encoding[self.current_phase] = 1.0
        
        # 相对位置
        relative_pos = self.current_target - ee_pos
        
        # 状态归一化
        if self.normalize_obs:
            joint_pos_norm = self._normalize_joint_pos(joint_pos)
            joint_vel_norm = np.clip(joint_vel / 10.0, -1.0, 1.0)
            ee_pos_norm = self._normalize_position(ee_pos)
            ee_quat_norm = ee_quat  # 四元数已经是归一化的
            object_pos_norm = self._normalize_position(object_pos)
            target_pos_norm = self._normalize_position(self.current_target)
            relative_pos_norm = self._normalize_position(relative_pos, center_zero=True)
        else:
            joint_pos_norm = joint_pos
            joint_vel_norm = joint_vel
            ee_pos_norm = ee_pos
            ee_quat_norm = ee_quat
            object_pos_norm = object_pos
            target_pos_norm = self.current_target
            relative_pos_norm = relative_pos
        
        obs.update({
            'joint_pos': joint_pos_norm.astype(np.float32),
            'joint_vel': joint_vel_norm.astype(np.float32),
            'ee_pos': ee_pos_norm.astype(np.float32),
            'ee_quat': ee_quat_norm.astype(np.float32),
            'gripper_state': np.array([self.gripper_state], dtype=np.float32),
            'object_pos': object_pos_norm.astype(np.float32),
            'object_in_gripper': np.array([float(self.object_grasped)], dtype=np.float32),
            'task_phase': phase_encoding.astype(np.float32),
            'target_pos': target_pos_norm.astype(np.float32),
            'relative_pos': relative_pos_norm.astype(np.float32)
        })
        
        return obs
        
    def _compute_reward(self, action: np.ndarray, ee_pos: np.ndarray, object_pos: np.ndarray) -> float:
        """计算奖励函数"""
        reward = 0.0
        
        if self.sparse_reward:
            # 稀疏奖励：只在任务完成时给奖励
            if self.task_completed:
                reward = 1000.0
            else:
                reward = -1.0  # 时间惩罚
        else:
            # 密集奖励：根据任务阶段给予不同奖励
            
            # 1. 基础距离奖励
            distance_to_target = np.linalg.norm(ee_pos - self.current_target)
            distance_reward = -distance_to_target * 10.0
            reward += distance_rewardimage.png
            
            # 2. 改进奖励
            if distance_to_target < self.best_distance_to_target:
                improvement_reward = (self.best_distance_to_target - distance_to_target) * 50.0
                reward += improvement_reward
                self.best_distance_to_target = distance_to_target
            
            # 3. 阶段特定奖励
            if self.current_phase == self.PHASE_APPROACH:
                # 接近阶段：鼓励接近物体上方
                if distance_to_target < 0.05:
                    reward += 50.0
                    
            elif self.current_phase == self.PHASE_GRASP:
                # 抓取阶段：鼓励接近物体并闭合夹爪
                distance_to_object = np.linalg.norm(ee_pos - object_pos)
                reward += -distance_to_object * 20.0
                
                if distance_to_object < 0.03 and self.gripper_state > 0.5:
                    reward += 100.0  # 抓取成功奖励
                    
            elif self.current_phase == self.PHASE_LIFT:
                # 抬起阶段：鼓励向上移动
                if object_pos[2] > self.object_initial_pos[2] + 0.08:
                    reward += 100.0
                    
            elif self.current_phase == self.PHASE_TRANSPORT:
                # 搬运阶段：鼓励移动到放置区域上方
                if distance_to_target < 0.05:
                    reward += 100.0
                    
            elif self.current_phase == self.PHASE_PLACE:
                # 放置阶段：鼓励放置物体
                drop_distance = np.linalg.norm(object_pos[:2] - self.drop_zone[:2])
                reward += -drop_distance * 50.0
                
                if drop_distance < 0.1 and self.gripper_state < 0.5:
                    reward += 200.0  # 放置成功奖励
            
            # 4. 任务完成巨额奖励
            if self.task_completed:
                reward += 1000.0
            
            # 5. 动作平滑性惩罚
            action_penalty = np.sum(np.square(action)) * 0.1
            reward -= action_penalty
            
            # 6. 夹爪状态奖励
            if self.current_phase in [self.PHASE_GRASP, self.PHASE_LIFT, self.PHASE_TRANSPORT]:
                if self.gripper_state > 0.7:  # 鼓励保持抓取
                    reward += 5.0
            elif self.current_phase == self.PHASE_PLACE:
                if self.gripper_state < 0.3:  # 鼓励释放物体
                    reward += 5.0
            
            # 7. 时间惩罚
            reward -= 0.1
        
        # 8. 奖励安全检查 - 防止NaN和inf
        if np.isnan(reward) or np.isinf(reward):
            print(f"Warning: Invalid reward detected: {reward}")
            print(f"  Distance to target: {distance_to_target}")
            print(f"  Best distance: {self.best_distance_to_target}")
            print(f"  Current phase: {self.current_phase}")
            print(f"  EE position: {ee_pos}")
            print(f"  Object position: {object_pos}")
            reward = -1.0  # 安全的默认奖励

        # 限制奖励范围，防止过大的值
        reward = np.clip(reward, -1000.0, 1000.0)

        return reward
        
    def _update_task_phase(self, ee_pos: np.ndarray, object_pos: np.ndarray):
        """更新任务阶段"""
        distance_to_target = np.linalg.norm(ee_pos - self.current_target)
        distance_to_object = np.linalg.norm(ee_pos - object_pos)
        
        if self.current_phase == self.PHASE_APPROACH:
            # 接近阶段 → 抓取阶段
            if distance_to_target < 0.05:
                self.current_phase = self.PHASE_GRASP
                self.current_target = self.grasp_target.copy()
                
        elif self.current_phase == self.PHASE_GRASP:
            # 抓取阶段 → 抬起阶段
            if distance_to_object < 0.03 and self.gripper_state > 0.5:
                self.object_grasped = True
                self.current_phase = self.PHASE_LIFT
                self.current_target = self.lift_target.copy()
                
        elif self.current_phase == self.PHASE_LIFT:
            # 抬起阶段 → 搬运阶段
            if object_pos[2] > self.object_initial_pos[2] + 0.08:
                self.object_lifted = True
                self.current_phase = self.PHASE_TRANSPORT
                self.current_target = self.transport_target.copy()
                
        elif self.current_phase == self.PHASE_TRANSPORT:
            # 搬运阶段 → 放置阶段
            if distance_to_target < 0.05:
                self.current_phase = self.PHASE_PLACE
                self.current_target = self.place_target.copy()
                
        elif self.current_phase == self.PHASE_PLACE:
            # 放置阶段 → 任务完成
            drop_distance = np.linalg.norm(object_pos[:2] - self.drop_zone[:2])
            if drop_distance < 0.1 and self.gripper_state < 0.3 and object_pos[2] < 0.8:
                self.task_completed = True
        
    def _check_terminated(self, ee_pos: np.ndarray, object_pos: np.ndarray) -> bool:
        """检查终止条件"""
        # 成功完成任务
        if self.task_completed:
            return True
            
        # 失败条件
        if (ee_pos[2] < 0.0 or  # 撞到地面
            not self._is_in_workspace(ee_pos) or  # 超出工作空间
            object_pos[2] < 0.5):  # 物体掉落
            return True
            
        return False
        
    def _get_object_position(self) -> np.ndarray:
        """获取物体位置"""
        try:
            # 通过mujoco获取物体位置
            body_id = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "Box")
            object_pos = self.env.mj_data.xpos[body_id].copy()
            return object_pos
        except:
            # 如果获取失败，返回初始位置
            return self.object_initial_pos.copy()
        
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """将RL动作转换为环境动作"""
        joint_increments = action[:-1]
        gripper_increment = action[-1]
        
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
            full_action[7] = self.gripper_state * 255  # 夹爪状态
        else:  # ur5e
            full_action = np.zeros(7)
            full_action[:6] = target_joints
            full_action[6] = self.gripper_state * 255
            
        return full_action
        
    def _randomize_object_position(self) -> np.ndarray:
        """随机化物体位置"""
        x = np.random.uniform(1.2, 1.6)
        y = np.random.uniform(0.0, 0.4)
        z = 0.9  # 固定高度
        return np.array([x, y, z])
        
    def _update_phase_targets(self):
        """更新阶段目标位置"""
        self.approach_target = self.object_initial_pos + np.array([0, 0, 0.1])
        self.grasp_target = self.object_initial_pos.copy()
        self.lift_target = self.object_initial_pos + np.array([0, 0, 0.15])
        self.current_target = self.approach_target.copy()
        
    def _normalize_joint_pos(self, joint_pos: np.ndarray) -> np.ndarray:
        """归一化关节位置"""
        normalized = np.zeros_like(joint_pos)
        for i, (pos, (low, high)) in enumerate(zip(joint_pos, self.joint_limits)):
            # 防止除零错误
            range_val = high - low
            if abs(range_val) < 1e-8:  # 避免除零
                normalized[i] = 0.0
            else:
                normalized[i] = 2 * (pos - low) / range_val - 1
                # 确保结果在合理范围内
                normalized[i] = np.clip(normalized[i], -1.0, 1.0)

        # 检查是否有NaN或inf值
        if np.any(np.isnan(normalized)) or np.any(np.isinf(normalized)):
            print(f"Warning: NaN/inf in normalized joint positions: {normalized}")
            print(f"Original joint positions: {joint_pos}")
            print(f"Joint limits: {self.joint_limits}")
            normalized = np.where(np.isnan(normalized) | np.isinf(normalized),
                                 0.0, normalized)

        return normalized
        
    def _normalize_position(self, pos: np.ndarray, center_zero=False) -> np.ndarray:
        """归一化位置"""
        if center_zero:
            return np.clip(pos / 2.0, -1.0, 1.0)
        else:
            # 映射到工作空间 [-1, 1]
            x_norm = (pos[0] - 0.5) / 1.0  # 工作空间 x: [0.5, 1.5]
            y_norm = pos[1] / 0.8          # 工作空间 y: [-0.8, 0.8]
            z_norm = (pos[2] - 0.5) / 1.0  # 工作空间 z: [0.5, 1.5]
            return np.array([x_norm, y_norm, z_norm])
        
    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        """检查是否在工作空间内"""
        return (0.2 <= pos[0] <= 1.8 and
                -0.8 <= pos[1] <= 0.8 and
                0.0 <= pos[2] <= 1.8)
        
    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        ee_pos = self.env.robot.get_cartesian().t
        object_pos = self._get_object_position()
        distance_to_target = np.linalg.norm(ee_pos - self.current_target)
        distance_to_object = np.linalg.norm(ee_pos - object_pos)
        drop_distance = np.linalg.norm(object_pos[:2] - self.drop_zone[:2])
        
        return {
            'step_count': self.step_count,
            'task_phase': self.current_phase,
            'task_completed': self.task_completed,
            'object_grasped': self.object_grasped,
            'object_lifted': self.object_lifted,
            'distance_to_target': distance_to_target,
            'distance_to_object': distance_to_object,
            'drop_distance': drop_distance,
            'ee_position': ee_pos,
            'object_position': object_pos,
            'current_target': self.current_target,
            'gripper_state': self.gripper_state,
            'success': self.task_completed
        }
        
    def close(self):
        """关闭环境"""
        self.env.close()
        
    def render(self):
        """渲染环境"""
        return self.env.render()


if __name__ == '__main__':
    # 测试环境
    print("Testing GraspTaskEnv...")
    
    env = GraspTaskEnv(robot_type='panda', use_image_obs=False)
    
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
            print(f"Step {step}: Phase={info['task_phase']}, Reward={reward:.2f}, "
                  f"Distance={info['distance_to_target']:.3f}")
            
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            print(f"Final info: {info}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.close()
    print("Environment test completed!")