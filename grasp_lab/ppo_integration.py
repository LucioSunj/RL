"""
PPO集成脚本：展示如何将训练好的PPO模型集成到现有仿真环境中
替换原有的GraspNet抓取检测和轨迹规划系统
"""

import os
import sys
import numpy as np
import torch
import time
import argparse
from typing import Dict, Optional

# 导入现有仿真环境
from manipulator_grasp.env.panda_grasp_env import PandaGraspEnv
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

# 导入PPO相关模块
from ppo_agent import PPOAgent
from rl_grasp_env import RLGraspEnv


class PPOGraspController:
    """PPO抓取控制器 - 替换GraspNet的智能控制器"""
    
    def __init__(self, 
                 model_path: str,
                 robot_type: str = 'panda',
                 success_threshold: float = 0.05,
                 max_steps: int = 500,
                 device: str = 'auto'):
        
        self.robot_type = robot_type
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 创建临时环境来获取空间定义
        temp_env = RLGraspEnv(robot_type=robot_type, use_image_obs=False)
        
        # 创建PPO agent
        self.agent = PPOAgent(
            observation_space=temp_env.observation_space,
            action_space=temp_env.action_space,
            device=self.device
        )
        
        # 加载训练好的模型
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"PPO model loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        temp_env.close()
        
        # 控制器状态
        self.step_count = 0
        self.target_position = None
        self.prev_joint_pos = None
        self.best_distance = float('inf')
        
        # 工作空间限制
        self.workspace_bounds = {
            'x': [0.3, 1.5],
            'y': [-0.6, 0.6], 
            'z': [0.0, 1.5]
        }
        
        # 关节限制
        if robot_type == 'panda':
            self.joint_limits = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
                                        [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525],
                                        [-2.8973, 2.8973]])
        else:  # ur5e
            self.joint_limits = np.array([[-2*np.pi, 2*np.pi]] * 6)
    
    def set_target(self, target_position: np.ndarray):
        """设置目标抓取位置"""
        self.target_position = target_position.copy()
        self.step_count = 0
        self.best_distance = float('inf')
        print(f"Target set to: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
    
    def get_action(self, env, deterministic: bool = True) -> np.ndarray:
        """
        获取下一步动作
        
        Args:
            env: 仿真环境 (PandaGraspEnv 或 UR5GraspEnv)
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 环境动作 [joint1, joint2, ..., gripper]
        """
        if self.target_position is None:
            raise ValueError("Target position not set. Call set_target() first.")
        
        # 获取观察
        observation = self._get_observation(env)
        
        # PPO预测动作
        action, _, _ = self.agent.get_action(observation, deterministic=deterministic)
        
        # 转换为环境动作格式
        env_action = self._convert_to_env_action(action[0], env)
        
        self.step_count += 1
        
        return env_action
    
    def is_task_complete(self, env) -> Dict[str, bool]:
        """
        检查任务是否完成
        
        Returns:
            dict: {'success': bool, 'failed': bool, 'timeout': bool}
        """
        if self.target_position is None:
            return {'success': False, 'failed': True, 'timeout': False}
        
        # 获取当前末端位置
        ee_pos = env.robot.get_cartesian().t
        distance = np.linalg.norm(ee_pos - self.target_position)
        
        # 成功条件
        success = distance < self.success_threshold
        
        # 失败条件
        failed = (not self._is_in_workspace(ee_pos) or 
                 ee_pos[2] < 0.0 or  # 撞到地面
                 self._is_joint_limit_violated(env.robot.get_joint()))
        
        # 超时条件
        timeout = self.step_count >= self.max_steps
        
        return {
            'success': success,
            'failed': failed,
            'timeout': timeout,
            'distance': distance,
            'steps': self.step_count
        }
    
    def execute_grasp_sequence(self, env, target_position: np.ndarray, 
                              render: bool = True, verbose: bool = True) -> Dict:
        """
        执行完整的抓取序列
        
        Args:
            env: 仿真环境
            target_position: 目标位置
            render: 是否渲染
            verbose: 是否输出详细信息
            
        Returns:
            dict: 执行结果
        """
        print(f"\n{'='*60}")
        print("PPO GRASP EXECUTION")
        print(f"{'='*60}")
        
        # 设置目标
        self.set_target(target_position)
        
        # 获取初始状态
        initial_ee_pos = env.robot.get_cartesian().t
        initial_distance = np.linalg.norm(initial_ee_pos - target_position)
        
        print(f"Initial EE Position: [{initial_ee_pos[0]:.3f}, {initial_ee_pos[1]:.3f}, {initial_ee_pos[2]:.3f}]")
        print(f"Target Position:     [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        print(f"Initial Distance:    {initial_distance:.4f}")
        print(f"Success Threshold:   {self.success_threshold:.4f}")
        print("")
        
        # 执行控制循环
        start_time = time.time()
        trajectory = []
        
        while True:
            # 获取动作
            action = self.get_action(env, deterministic=True)
            
            # 执行动作
            env.step(action)
            
            # 获取当前状态
            current_ee_pos = env.robot.get_cartesian().t
            current_distance = np.linalg.norm(current_ee_pos - target_position)
            
            # 记录轨迹
            trajectory.append({
                'step': self.step_count,
                'ee_pos': current_ee_pos.copy(),
                'distance': current_distance,
                'action': action.copy()
            })
            
            # 更新最佳距离
            if current_distance < self.best_distance:
                self.best_distance = current_distance
            
            # 渲染
            if render:
                env.render()
            
            # 输出进度
            if verbose and self.step_count % 10 == 0:
                print(f"Step {self.step_count:3d}: Distance {current_distance:.4f}, "
                      f"Best {self.best_distance:.4f}")
            
            # 检查完成条件
            status = self.is_task_complete(env)
            
            if status['success']:
                print(f"\n✓ SUCCESS! Reached target in {self.step_count} steps")
                break
            elif status['failed']:
                print(f"\n✗ FAILED! Task failed after {self.step_count} steps")
                if not self._is_in_workspace(current_ee_pos):
                    print("  Reason: Out of workspace")
                if current_ee_pos[2] < 0.0:
                    print("  Reason: Collision with ground")
                if self._is_joint_limit_violated(env.robot.get_joint()):
                    print("  Reason: Joint limit violated")
                break
            elif status['timeout']:
                print(f"\n⏰ TIMEOUT! Max steps ({self.max_steps}) reached")
                break
        
        # 计算结果
        execution_time = time.time() - start_time
        final_ee_pos = env.robot.get_cartesian().t
        final_distance = np.linalg.norm(final_ee_pos - target_position)
        
        result = {
            'success': status['success'],
            'steps': self.step_count,
            'execution_time': execution_time,
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'best_distance': self.best_distance,
            'initial_ee_pos': initial_ee_pos,
            'final_ee_pos': final_ee_pos,
            'target_position': target_position,
            'trajectory': trajectory
        }
        
        # 打印总结
        print(f"\nExecution Summary:")
        print(f"  Success: {'✓' if result['success'] else '✗'}")
        print(f"  Steps: {result['steps']}")
        print(f"  Time: {result['execution_time']:.2f}s")
        print(f"  Final Distance: {result['final_distance']:.4f}")
        print(f"  Distance Improvement: {initial_distance - final_distance:.4f}")
        print(f"{'='*60}")
        
        return result
    
    def _get_observation(self, env) -> Dict[str, np.ndarray]:
        """获取PPO所需的观察"""
        # 机械臂状态
        joint_pos = env.robot.get_joint()
        ee_transform = env.robot.get_cartesian()
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
        joint_pos_norm = self._normalize_joint_pos(joint_pos)
        joint_vel_norm = np.clip(joint_vel / 10.0, -1.0, 1.0)
        ee_pos_norm = self._normalize_position(ee_pos)
        target_relative_norm = self._normalize_position(target_relative, center_zero=True)
        
        return {
            'joint_pos': joint_pos_norm.astype(np.float32),
            'joint_vel': joint_vel_norm.astype(np.float32),
            'ee_pos': ee_pos_norm.astype(np.float32),
            'target_relative': target_relative_norm.astype(np.float32),
            'gripper_state': np.array([0.0], dtype=np.float32)  # 简化处理
        }
    
    def _convert_to_env_action(self, ppo_action: np.ndarray, env) -> np.ndarray:
        """将PPO动作转换为环境动作"""
        # PPO输出的是关节增量 + 夹爪动作
        joint_increments = ppo_action[:-1]
        gripper_increment = ppo_action[-1]
        
        # 获取当前关节位置
        current_joints = env.robot.get_joint()
        
        # 计算目标关节位置
        target_joints = current_joints + joint_increments
        
        # 关节限制
        for i, (low, high) in enumerate(self.joint_limits):
            target_joints[i] = np.clip(target_joints[i], low, high)
        
        # 构造完整动作
        if self.robot_type == 'panda':
            full_action = np.zeros(8)
            full_action[:7] = target_joints
            full_action[7] = np.clip(gripper_increment * 50 + 127.5, 0, 255)  # 夹爪控制
        else:  # ur5e
            full_action = np.zeros(7)
            full_action[:6] = target_joints
            full_action[6] = np.clip(gripper_increment * 50 + 127.5, 0, 255)
            
        return full_action
    
    def _normalize_joint_pos(self, joint_pos: np.ndarray) -> np.ndarray:
        """归一化关节位置"""
        normalized = np.zeros_like(joint_pos)
        for i, (pos, (low, high)) in enumerate(zip(joint_pos, self.joint_limits)):
            normalized[i] = 2 * (pos - low) / (high - low) - 1
        return normalized
    
    def _normalize_position(self, pos: np.ndarray, center_zero=False) -> np.ndarray:
        """归一化位置"""
        if center_zero:
            return np.clip(pos / 2.0, -1.0, 1.0)
        else:
            x_norm = 2 * (pos[0] - self.workspace_bounds['x'][0]) / (self.workspace_bounds['x'][1] - self.workspace_bounds['x'][0]) - 1
            y_norm = 2 * (pos[1] - self.workspace_bounds['y'][0]) / (self.workspace_bounds['y'][1] - self.workspace_bounds['y'][0]) - 1
            z_norm = 2 * (pos[2] - self.workspace_bounds['z'][0]) / (self.workspace_bounds['z'][1] - self.workspace_bounds['z'][0]) - 1
            return np.array([x_norm, y_norm, z_norm])
    
    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        """检查是否在工作空间内"""
        return (self.workspace_bounds['x'][0] <= pos[0] <= self.workspace_bounds['x'][1] and
                self.workspace_bounds['y'][0] <= pos[1] <= self.workspace_bounds['y'][1] and
                self.workspace_bounds['z'][0] <= pos[2] <= self.workspace_bounds['z'][1])
    
    def _is_joint_limit_violated(self, joint_pos: np.ndarray) -> bool:
        """检查关节限制是否违反"""
        for i, (pos, (low, high)) in enumerate(zip(joint_pos, self.joint_limits)):
            if pos < low or pos > high:
                return True
        return False


def demo_ppo_vs_graspnet():
    """演示PPO控制器 vs 原始GraspNet方法的对比"""
    
    # 创建环境
    env = PandaGraspEnv()  # 或者 UR5GraspEnv()
    env.reset()
    
    # 让环境稳定
    for _ in range(100):
        env.step()
    
    # 定义测试目标
    test_targets = [
        np.array([1.4, 0.2, 0.9]),
        np.array([1.0, 0.3, 0.85]),
        np.array([1.2, -0.2, 0.95]),
        np.array([0.8, 0.4, 0.8])
    ]
    
    # 假设已经训练好的模型路径
    model_path = "./checkpoints/best_model.pt"
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found: {model_path}")
        print("Please train a model first using train_ppo.py")
        return
    
    # 创建PPO控制器
    ppo_controller = PPOGraspController(
        model_path=model_path,
        robot_type='panda'
    )
    
    print("Starting PPO vs GraspNet demonstration...")
    
    results = []
    
    for i, target in enumerate(test_targets):
        print(f"\n\nTest Case {i+1}/{len(test_targets)}")
        
        # 重置环境
        env.reset()
        for _ in range(100):
            env.step()
        
        # 使用PPO控制器执行抓取
        result = ppo_controller.execute_grasp_sequence(
            env=env,
            target_position=target,
            render=True,
            verbose=True
        )
        
        results.append(result)
        
        # 等待用户确认继续
        input("Press Enter to continue to next test case...")
    
    # 打印总体结果
    print(f"\n\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}")
    
    successes = [r['success'] for r in results]
    steps = [r['steps'] for r in results]
    times = [r['execution_time'] for r in results]
    improvements = [r['initial_distance'] - r['final_distance'] for r in results]
    
    print(f"Total Test Cases: {len(results)}")
    print(f"Success Rate: {np.mean(successes):.3f} ({np.sum(successes)}/{len(successes)})")
    print(f"Average Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"Average Time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
    print(f"Average Distance Improvement: {np.mean(improvements):.4f} ± {np.std(improvements):.4f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='PPO Integration Demo')
    
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pt',
                       help='Path to trained PPO model')
    parser.add_argument('--robot_type', type=str, default='panda', 
                       choices=['panda', 'ur5e'], help='Robot type')
    parser.add_argument('--target', nargs=3, type=float, 
                       default=[1.4, 0.2, 0.9], help='Target position [x, y, z]')
    parser.add_argument('--render', action='store_true', default=True,
                       help='Render execution')
    parser.add_argument('--demo', action='store_true', 
                       help='Run full demonstration')
    
    args = parser.parse_args()
    
    if args.demo:
        # 运行完整演示
        demo_ppo_vs_graspnet()
    else:
        # 单个目标测试
        # 创建环境
        if args.robot_type == 'panda':
            env = PandaGraspEnv()
        else:
            env = UR5GraspEnv()
        
        env.reset()
        for _ in range(100):
            env.step()
        
        # 创建PPO控制器
        try:
            ppo_controller = PPOGraspController(
                model_path=args.model_path,
                robot_type=args.robot_type
            )
            
            # 执行抓取
            target = np.array(args.target)
            result = ppo_controller.execute_grasp_sequence(
                env=env,
                target_position=target,
                render=args.render
            )
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please train a model first using train_ppo.py")
        
        env.close()


if __name__ == '__main__':
    main()