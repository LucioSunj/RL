"""
测试完整抓取任务的训练模型
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
import json
from typing import Dict, List
import matplotlib.pyplot as plt

# 添加路径
manipulator_grasp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp')
if manipulator_grasp_path not in sys.path:
    sys.path.insert(0, manipulator_grasp_path)

from grasp_task_env import GraspTaskEnv
from ppo_agent import PPOAgent


class GraspTaskTester:
    """完整抓取任务测试器"""
    
    def __init__(self, model_path: str, config_path: str = None, robot_type: str = 'panda'):
        self.model_path = model_path
        self.robot_type = robot_type
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)['config']
        else:
            # 默认配置
            self.config = {
                'robot_type': robot_type,
                'use_image_obs': False,
                'image_size': [84, 84],
                'max_episode_steps': 2000,
                'randomize_object': False,
                'sparse_reward': False,
                'headless': True
            }
        
        # 创建环境
        self.env = GraspTaskEnv(
            robot_type=self.config['robot_type'],
            image_size=tuple(self.config.get('image_size', [84, 84])),
            max_episode_steps=self.config['max_episode_steps'],
            use_image_obs=self.config.get('use_image_obs', False),
            randomize_object=self.config.get('randomize_object', False),
            sparse_reward=self.config.get('sparse_reward', False),
            headless=self.config.get('headless', True)
        )
        
        # 创建agent并加载模型
        self.agent = PPOAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 加载模型
        self.agent.load(model_path)
        print(f"Grasp task model loaded from: {model_path}")
        
        # 测试结果
        self.test_results = {
            'rewards': [],
            'lengths': [],
            'successes': [],
            'final_phases': [],
            'max_phases': [],
            'phase_completion_rates': {
                'approach': [],
                'grasp': [],
                'lift': [],
                'transport': [],
                'place': []
            }
        }
        
        # 阶段名称
        self.phase_names = ['Approach', 'Grasp', 'Lift', 'Transport', 'Place']
    
    def test_episodes(self, n_episodes: int = 20, deterministic: bool = True, 
                     render: bool = False, save_video: bool = False, verbose: bool = True):
        """测试多个episodes"""
        print(f"Testing {n_episodes} episodes of complete grasp task...")
        print(f"Deterministic: {deterministic}")
        print(f"Environment: {type(self.env).__name__}")
        print(f"Robot: {self.robot_type}")
        
        for episode in range(n_episodes):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"{'='*60}")
            
            # 运行episode
            result = self._run_episode(
                deterministic=deterministic, 
                render=render, 
                save_frames=save_video,
                verbose=verbose
            )
            
            # 记录结果
            self._record_result(result)
            
            # 打印结果
            if verbose:
                self._print_episode_result(result, episode + 1)
        
        # 打印总结
        self._print_summary()
        
        return self.test_results
    
    def _run_episode(self, deterministic: bool = True, render: bool = False, 
                    save_frames: bool = False, verbose: bool = True):
        """运行单个episode"""
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        frames = [] if save_frames else None
        max_phase_reached = 0
        
        phase_completion = {
            'approach': False,
            'grasp': False,
            'lift': False,
            'transport': False,
            'place': False
        }
        
        # 记录阶段转换
        phase_transitions = []
        prev_phase = -1
        
        if verbose:
            print(f"Initial state:")
            print(f"  Task phase: {self.phase_names[info['task_phase']]}")
            print(f"  EE position: {info['ee_position']}")
            print(f"  Object position: {info['object_position']}")
            print(f"  Target: {info['current_target']}")
        
        while not done and episode_length < self.config['max_episode_steps']:
            # 获取动作
            action, _, _ = self.agent.get_action(obs, deterministic=deterministic)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action[0])
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # 记录阶段信息
            current_phase = info['task_phase']
            max_phase_reached = max(max_phase_reached, current_phase)
            
            # 记录阶段转换
            if current_phase != prev_phase:
                phase_transitions.append({
                    'step': episode_length,
                    'from_phase': prev_phase,
                    'to_phase': current_phase,
                    'phase_name': self.phase_names[current_phase] if current_phase < len(self.phase_names) else f"Phase_{current_phase}"
                })
                prev_phase = current_phase
            
            # 更新阶段完成情况
            phase_names_lower = ['approach', 'grasp', 'lift', 'transport', 'place']
            for i, phase_name in enumerate(phase_names_lower):
                if current_phase > i:
                    phase_completion[phase_name] = True
            
            # 渲染和保存帧
            if render:
                self.env.render()
                time.sleep(0.02)
            
            if save_frames:
                frame = self.env.render()
                if 'image' in frame:
                    frames.append(frame['image'])
            
            # 定期输出进度
            if verbose and episode_length % 200 == 0:
                print(f"  Step {episode_length:4d}: Phase={self.phase_names[current_phase]:10s} "
                      f"Reward={reward:6.2f} Distance={info['distance_to_target']:.3f}")
            
            obs = next_obs
        
        # 最终信息
        final_info = info
        success = final_info.get('task_completed', False)
        
        if verbose:
            print(f"\nPhase transitions:")
            for transition in phase_transitions:
                from_name = self.phase_names[transition['from_phase']] if transition['from_phase'] >= 0 else "Start"
                to_name = transition['phase_name']
                print(f"  Step {transition['step']:4d}: {from_name} → {to_name}")
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'success': success,
            'final_phase': final_info['task_phase'],
            'max_phase_reached': max_phase_reached,
            'phase_completion': phase_completion,
            'phase_transitions': phase_transitions,
            'final_info': final_info,
            'frames': frames
        }
    
    def _record_result(self, result: Dict):
        """记录测试结果"""
        self.test_results['rewards'].append(result['reward'])
        self.test_results['lengths'].append(result['length'])
        self.test_results['successes'].append(result['success'])
        self.test_results['final_phases'].append(result['final_phase'])
        self.test_results['max_phases'].append(result['max_phase_reached'])
        
        # 记录阶段完成情况
        for phase, completed in result['phase_completion'].items():
            self.test_results['phase_completion_rates'][phase].append(completed)
    
    def _print_episode_result(self, result: Dict, episode_num: int):
        """打印单个episode结果"""
        print(f"\nEpisode {episode_num} Results:")
        print(f"  Total Reward: {result['reward']:8.2f}")
        print(f"  Episode Length: {result['length']:4d} steps")
        print(f"  Success: {'✅ YES' if result['success'] else '❌ NO'}")
        print(f"  Final Phase: {self.phase_names[result['final_phase']]}")
        print(f"  Max Phase Reached: {self.phase_names[result['max_phase_reached']]}")
        print(f"  Phase Completion:")
        for phase, completed in result['phase_completion'].items():
            status = '✅' if completed else '❌'
            print(f"    {phase.capitalize()}: {status}")
        
        final_info = result['final_info']
        print(f"  Final Distance to Target: {final_info['distance_to_target']:.3f}m")
        print(f"  Final Distance to Object: {final_info['distance_to_object']:.3f}m")
        print(f"  Object Grasped: {'✅' if final_info['object_grasped'] else '❌'}")
        print(f"  Object Lifted: {'✅' if final_info['object_lifted'] else '❌'}")
    
    def _print_summary(self):
        """打印测试总结"""
        rewards = np.array(self.test_results['rewards'])
        lengths = np.array(self.test_results['lengths'])
        successes = np.array(self.test_results['successes'])
        final_phases = np.array(self.test_results['final_phases'])
        max_phases = np.array(self.test_results['max_phases'])
        
        print(f"\n{'='*80}")
        print("COMPLETE GRASP TASK TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total Episodes: {len(rewards)}")
        print(f"Overall Success Rate: {np.mean(successes):.3f} ({np.sum(successes)}/{len(successes)})")
        print(f"")
        
        # 奖励统计
        print(f"Rewards:")
        print(f"  Mean: {np.mean(rewards):8.2f} ± {np.std(rewards):6.2f}")
        print(f"  Min:  {np.min(rewards):8.2f}")
        print(f"  Max:  {np.max(rewards):8.2f}")
        print(f"")
        
        # 步数统计
        print(f"Episode Lengths:")
        print(f"  Mean: {np.mean(lengths):6.1f} ± {np.std(lengths):5.1f}")
        print(f"  Min:  {np.min(lengths):6.0f}")
        print(f"  Max:  {np.max(lengths):6.0f}")
        print(f"")
        
        # 阶段统计
        print(f"Phase Analysis:")
        print(f"  Mean Final Phase: {np.mean(final_phases):.2f}")
        print(f"  Mean Max Phase: {np.mean(max_phases):.2f}")
        print(f"")
        
        print(f"Phase Completion Rates:")
        for phase, completions in self.test_results['phase_completion_rates'].items():
            completion_rate = np.mean(completions)
            print(f"  {phase.capitalize():10s}: {completion_rate:.3f} ({np.sum(completions)}/{len(completions)})")
        print(f"")
        
        # 成功episode分析
        success_indices = np.where(successes)[0]
        if len(success_indices) > 0:
            success_rewards = rewards[success_indices]
            success_lengths = lengths[success_indices]
            print(f"Successful Episodes Analysis:")
            print(f"  Count: {len(success_indices)}")
            print(f"  Average Reward: {np.mean(success_rewards):.2f}")
            print(f"  Average Length: {np.mean(success_lengths):.1f}")
        
        print(f"{'='*80}")
    
    def interactive_test(self):
        """交互式测试"""
        print("Interactive grasp task testing mode")
        print("Commands:")
        print("  'q' - quit")
        print("  'r' - reset environment")
        print("  'n' - run episode")
        print("  'd' - toggle deterministic")
        print("  's' - show statistics")
        print("  'v' - toggle verbose mode")
        
        deterministic = True
        verbose = True
        
        while True:
            cmd = input("\nEnter command: ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 'r':
                obs, info = self.env.reset()
                print("Environment reset")
                print(f"Task phase: {self.phase_names[info['task_phase']]}")
                print(f"Object position: {info['object_position']}")
                print(f"Target: {info['current_target']}")
            elif cmd == 'n':
                result = self._run_episode(
                    deterministic=deterministic, 
                    render=True, 
                    verbose=verbose
                )
                self._record_result(result)
                self._print_episode_result(result, len(self.test_results['rewards']))
            elif cmd == 'd':
                deterministic = not deterministic
                print(f"Deterministic mode: {deterministic}")
            elif cmd == 'v':
                verbose = not verbose
                print(f"Verbose mode: {verbose}")
            elif cmd == 's':
                if self.test_results['rewards']:
                    self._print_summary()
                else:
                    print("No test results yet")
            else:
                print("Unknown command")
    
    def close(self):
        """关闭环境"""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description='Test trained grasp task PPO model')
    
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('--config_path', type=str, help='Path to config file')
    parser.add_argument('--robot_type', type=str, default='panda', choices=['panda', 'ur5e'])
    parser.add_argument('--n_episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--render', action='store_true', help='Render episodes')
    parser.add_argument('--save_video', action='store_true', help='Save video of episodes')
    parser.add_argument('--interactive', action='store_true', help='Interactive testing mode')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # 创建测试器
    tester = GraspTaskTester(
        model_path=args.model_path,
        config_path=args.config_path,
        robot_type=args.robot_type
    )
    
    try:
        if args.interactive:
            # 交互式模式
            tester.interactive_test()
        else:
            # 标准测试
            tester.test_episodes(
                n_episodes=args.n_episodes,
                deterministic=args.deterministic,
                render=args.render,
                save_video=args.save_video,
                verbose=args.verbose
            )
    
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"\nTesting failed with error: {e}")
        raise
    finally:
        tester.close()


if __name__ == '__main__':
    main()