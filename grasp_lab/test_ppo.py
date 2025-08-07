import os
import argparse
import numpy as np
import torch
import time
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import cv2

from rl_grasp_env import RLGraspEnv, SimpleRLGraspEnv
from ppo_agent import PPOAgent

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

class PPOTester:
    """PPO测试器"""
    
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
                'use_simple_env': True,
                'image_size': [84, 84],
                'max_episode_steps': 500,
                'success_threshold': 0.05,
                'randomize_target': True,
                'sparse_reward': False
            }
        
        # 创建环境
        self._create_environment()
        
        # 创建agent并加载模型
        self.agent = PPOAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 加载模型
        self.agent.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        # 测试统计
        self.test_results = {
            'rewards': [],
            'lengths': [],
            'successes': [],
            'distances': [],
            'success_steps': []
        }
    
    def _create_environment(self):
        """创建测试环境"""
        if self.config.get('use_simple_env', True):
            self.env = SimpleRLGraspEnv(
                robot_type=self.config['robot_type'],
                max_episode_steps=self.config['max_episode_steps'],
                success_distance_threshold=self.config['success_threshold'],
                randomize_target=self.config.get('randomize_target', True),
                sparse_reward=self.config.get('sparse_reward', False)
            )
        else:
            self.env = RLGraspEnv(
                robot_type=self.config['robot_type'],
                image_size=tuple(self.config.get('image_size', [84, 84])),
                max_episode_steps=self.config['max_episode_steps'],
                success_distance_threshold=self.config['success_threshold'],
                use_image_obs=self.config.get('use_image_obs', False),
                randomize_target=self.config.get('randomize_target', True),
                sparse_reward=self.config.get('sparse_reward', False)
            )
    
    def test_episodes(self, n_episodes: int = 20, deterministic: bool = True, render: bool = False, save_video: bool = False):
        """测试多个episode"""
        print(f"Testing {n_episodes} episodes...")
        print(f"Deterministic: {deterministic}")
        print(f"Environment: {type(self.env).__name__}")
        
        video_frames = [] if save_video else None
        
        for episode in range(n_episodes):
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            
            # 运行单个episode
            result = self._run_episode(deterministic=deterministic, render=render, save_frames=save_video)
            
            # 记录结果
            self.test_results['rewards'].append(result['reward'])
            self.test_results['lengths'].append(result['length'])
            self.test_results['successes'].append(result['success'])
            self.test_results['distances'].append(result['final_distance'])
            
            if result['success']:
                self.test_results['success_steps'].append(result['length'])
            
            if save_video and result['frames']:
                video_frames.extend(result['frames'])
            
            # 打印结果
            print(f"  Reward: {result['reward']:8.2f}")
            print(f"  Length: {result['length']:3d}")
            print(f"  Success: {'✓' if result['success'] else '✗'}")
            print(f"  Final Distance: {result['final_distance']:.4f}")
            print(f"  Target: {result['target_pos']}")
            print(f"  Final EE: {result['final_ee_pos']}")
        
        # 保存视频
        if save_video and video_frames:
            self._save_video(video_frames, 'test_episodes.mp4')
        
        # 打印总结
        self._print_summary()
        
        return self.test_results
    
    def _run_episode(self, deterministic: bool = True, render: bool = False, save_frames: bool = False):
        """运行单个episode"""
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        frames = [] if save_frames else None
        
        # 记录初始状态
        initial_ee_pos = info['ee_position']
        target_pos = info['target_position']
        
        print(f"    Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"    Initial EE: [{initial_ee_pos[0]:.3f}, {initial_ee_pos[1]:.3f}, {initial_ee_pos[2]:.3f}]")
        print(f"    Initial Distance: {info['target_distance']:.4f}")
        
        while not done and episode_length < self.config['max_episode_steps']:
            # 获取动作
            action, _, _ = self.agent.get_action(obs, deterministic=deterministic)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action[0])
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # 渲染
            if render:
                self.env.render()
                time.sleep(0.01)  # 稍微延迟以便观察
            
            # 保存帧
            if save_frames:
                frame = self.env.render()
                if 'image' in frame:
                    frames.append(frame['image'])
            
            # 更新观察
            obs = next_obs
            
            # 每10步打印一次距离
            if episode_length % 10 == 0:
                print(f"    Step {episode_length:3d}: Distance {info['target_distance']:.4f}, Reward {reward:6.2f}")
        
        # 最终信息
        final_ee_pos = info['ee_position']
        final_distance = info['target_distance']
        success = info['success']
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'success': success,
            'final_distance': final_distance,
            'target_pos': target_pos,
            'initial_ee_pos': initial_ee_pos,
            'final_ee_pos': final_ee_pos,
            'frames': frames
        }
    
    def _print_summary(self):
        """打印测试总结"""
        rewards = np.array(self.test_results['rewards'])
        lengths = np.array(self.test_results['lengths'])
        successes = np.array(self.test_results['successes'])
        distances = np.array(self.test_results['distances'])
        
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Episodes: {len(rewards)}")
        print(f"Success Rate: {np.mean(successes):.3f} ({np.sum(successes)}/{len(successes)})")
        print(f"")
        print(f"Rewards:")
        print(f"  Mean: {np.mean(rewards):8.2f} ± {np.std(rewards):6.2f}")
        print(f"  Min:  {np.min(rewards):8.2f}")
        print(f"  Max:  {np.max(rewards):8.2f}")
        print(f"")
        print(f"Episode Lengths:")
        print(f"  Mean: {np.mean(lengths):6.1f} ± {np.std(lengths):5.1f}")
        print(f"  Min:  {np.min(lengths):6.0f}")
        print(f"  Max:  {np.max(lengths):6.0f}")
        print(f"")
        print(f"Final Distances:")
        print(f"  Mean: {np.mean(distances):6.4f} ± {np.std(distances):6.4f}")
        print(f"  Min:  {np.min(distances):6.4f}")
        print(f"  Max:  {np.max(distances):6.4f}")
        
        if len(self.test_results['success_steps']) > 0:
            success_steps = np.array(self.test_results['success_steps'])
            print(f"")
            print(f"Success Episodes Steps:")
            print(f"  Mean: {np.mean(success_steps):6.1f} ± {np.std(success_steps):5.1f}")
            print(f"  Min:  {np.min(success_steps):6.0f}")
            print(f"  Max:  {np.max(success_steps):6.0f}")
        
        print(f"{'='*60}")
    
    def _save_video(self, frames: List, filename: str, fps: int = 30):
        """保存视频"""
        if not frames:
            return
        
        print(f"Saving video: {filename}")
        
        # 确定视频尺寸
        height, width = frames[0].shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame in frames:
            # 转换颜色格式 (RGB -> BGR)
            if len(frame.shape) == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved: {filename}")
    
    def interactive_test(self):
        """交互式测试"""
        print("Interactive testing mode")
        print("Commands:")
        print("  'q' - quit")
        print("  'r' - reset environment")
        print("  'n' - run episode")
        print("  'd' - toggle deterministic")
        print("  's' - show statistics")
        
        deterministic = True
        
        while True:
            cmd = input("\nEnter command: ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 'r':
                obs, info = self.env.reset()
                print("Environment reset")
                print(f"Target: {info['target_position']}")
                print(f"Initial EE: {info['ee_position']}")
                print(f"Distance: {info['target_distance']:.4f}")
            elif cmd == 'n':
                result = self._run_episode(deterministic=deterministic, render=True)
                print(f"Episode completed:")
                print(f"  Reward: {result['reward']:.2f}")
                print(f"  Success: {result['success']}")
                print(f"  Length: {result['length']}")
            elif cmd == 'd':
                deterministic = not deterministic
                print(f"Deterministic mode: {deterministic}")
            elif cmd == 's':
                if self.test_results['rewards']:
                    self._print_summary()
                else:
                    print("No test results yet")
            else:
                print("Unknown command")
    
    def benchmark_performance(self, n_episodes: int = 100):
        """性能基准测试"""
        print(f"Running benchmark with {n_episodes} episodes...")
        
        start_time = time.time()
        results = self.test_episodes(n_episodes, deterministic=True, render=False)
        end_time = time.time()
        
        total_time = end_time - start_time
        fps = sum(results['lengths']) / total_time
        
        print(f"\nBenchmark Results:")
        print(f"Total Time: {total_time:.2f}s")
        print(f"FPS: {fps:.1f}")
        print(f"Time per Episode: {total_time / n_episodes:.3f}s")
        
        return {
            'total_time': total_time,
            'fps': fps,
            'time_per_episode': total_time / n_episodes,
            'results': results
        }
    
    def close(self):
        """关闭环境"""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description='Test trained PPO model')
    
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('--config_path', type=str, help='Path to config file')
    parser.add_argument('--robot_type', type=str, default='panda', choices=['panda', 'ur5e'])
    parser.add_argument('--n_episodes', type=int, default=20, help='Number of test episodes')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Use deterministic policy')
    parser.add_argument('--render', action='store_true', help='Render episodes')
    parser.add_argument('--save_video', action='store_true', help='Save video of episodes')
    parser.add_argument('--interactive', action='store_true', help='Interactive testing mode')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # 创建测试器
    tester = PPOTester(
        model_path=args.model_path,
        config_path=args.config_path,
        robot_type=args.robot_type
    )
    
    try:
        if args.interactive:
            # 交互式模式
            tester.interactive_test()
        elif args.benchmark:
            # 性能基准测试
            tester.benchmark_performance(args.n_episodes)
        else:
            # 标准测试
            tester.test_episodes(
                n_episodes=args.n_episodes,
                deterministic=args.deterministic,
                render=args.render,
                save_video=args.save_video
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