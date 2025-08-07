import os
import sys
import time
import argparse
import numpy as np
import torch
import wandb
from datetime import datetime
from typing import Dict, List
import json

# 重要：必须在导入其他自定义模块之前添加路径！
# 添加manipulator_grasp目录到Python路径，使其可以作为根模块导入
manipulator_grasp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp')
if manipulator_grasp_path not in sys.path:
    sys.path.insert(0, manipulator_grasp_path)

from rl_grasp_env import RLGraspEnv, SimpleRLGraspEnv
from ppo_agent import PPOAgent


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 创建环境
        if config['use_simple_env']:
            self.env = SimpleRLGraspEnv(
                robot_type=config['robot_type'],
                max_episode_steps=config['max_episode_steps'],
                success_distance_threshold=config['success_threshold'],
                randomize_target=config['randomize_target'],
                sparse_reward=config['sparse_reward']
            )
        else:
            self.env = RLGraspEnv(
                robot_type=config['robot_type'],
                image_size=tuple(config['image_size']),
                max_episode_steps=config['max_episode_steps'],
                success_distance_threshold=config['success_threshold'],
                use_image_obs=config['use_image_obs'],
                randomize_target=config['randomize_target'],
                sparse_reward=config['sparse_reward']
            )
        
        # 创建agent
        self.agent = PPOAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_epsilon=config['clip_epsilon'],
            entropy_coef=config['entropy_coef'],
            value_loss_coef=config['value_loss_coef'],
            max_grad_norm=config['max_grad_norm'],
            target_kl=config['target_kl'],
            n_epochs=config['n_epochs'],
            batch_size=config['batch_size'],
            buffer_size=config['buffer_size'],
            device=config['device']
        )
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.best_reward = float('-inf')
        self.best_success_rate = 0.0
        
        # 文件路径
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 评估统计
        self.eval_rewards = []
        self.eval_success_rates = []
        
    def train(self):
        """主训练循环"""
        print("Starting PPO training...")
        print(f"Environment: {type(self.env).__name__}")
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")
        print(f"Device: {self.agent.device}")
        
        start_time = time.time()
        global_step = 0
        episode = 0
        
        while global_step < self.config['total_timesteps']:
            # 收集经验
            episode_reward, episode_length, episode_success = self._collect_rollout()
            
            episode += 1
            global_step = self.agent.total_steps
            
            # 记录episode统计
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # 计算成功率 (最近100个episode)
            recent_episodes = min(100, len(self.episode_rewards))
            if hasattr(self, '_recent_successes'):
                self._recent_successes.append(episode_success)
                if len(self._recent_successes) > 100:
                    self._recent_successes.pop(0)
            else:
                self._recent_successes = [episode_success]
            
            current_success_rate = np.mean(self._recent_successes)
            self.success_rates.append(current_success_rate)
            
            # 更新网络
            if self.agent.buffer.size >= self.config['buffer_size']:
                obs, _ = self.env.reset()  # 获取最后观察用于价值估计
                update_stats = self.agent.update(obs)
                
                # 记录训练统计
                log_data = {
                    'episode': episode,
                    'global_step': global_step,
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'episode_success': episode_success,
                    'success_rate': current_success_rate,
                    'mean_reward_100': np.mean(self.episode_rewards[-100:]),
                    'std_reward_100': np.std(self.episode_rewards[-100:]),
                    **update_stats
                }
                
                # wandb记录
                if self.config['use_wandb']:
                    wandb.log(log_data, step=global_step)
                
                # 控制台输出
                if episode % self.config['log_interval'] == 0:
                    elapsed_time = time.time() - start_time
                    fps = global_step / elapsed_time
                    
                    print(f"\nEpisode: {episode:6d} | Step: {global_step:8d}")
                    print(f"Reward: {episode_reward:8.2f} | Success: {episode_success}")
                    print(f"Success Rate: {current_success_rate:6.3f} | Mean Reward: {log_data['mean_reward_100']:8.2f}")
                    print(f"Policy Loss: {update_stats['policy_loss']:8.4f} | Value Loss: {update_stats['value_loss']:8.4f}")
                    print(f"Entropy: {update_stats['entropy']:8.4f} | KL Div: {update_stats['kl_divergence']:8.4f}")
                    print(f"FPS: {fps:.1f} | Time: {elapsed_time/60:.1f}m")
                
                # 保存检查点
                if episode % self.config['save_interval'] == 0:
                    self._save_checkpoint(episode, global_step)
                
                # 评估
                if episode % self.config['eval_interval'] == 0:
                    eval_reward, eval_success_rate = self._evaluate()
                    self.eval_rewards.append(eval_reward)
                    self.eval_success_rates.append(eval_success_rate)
                    
                    eval_log = {
                        'eval_reward': eval_reward,
                        'eval_success_rate': eval_success_rate,
                        'eval_episode': episode
                    }
                    
                    if self.config['use_wandb']:
                        wandb.log(eval_log, step=global_step)
                    
                    print(f"\n{'='*50}")
                    print(f"EVALUATION - Episode {episode}")
                    print(f"Eval Reward: {eval_reward:.2f} | Eval Success Rate: {eval_success_rate:.3f}")
                    print(f"{'='*50}\n")
                    
                    # 保存最佳模型
                    if eval_success_rate > self.best_success_rate:
                        self.best_success_rate = eval_success_rate
                        self.best_reward = eval_reward
                        self._save_best_model(episode, global_step, eval_reward, eval_success_rate)
        
        # 训练完成
        print(f"\nTraining completed!")
        print(f"Total episodes: {episode}")
        print(f"Total timesteps: {global_step}")
        print(f"Best success rate: {self.best_success_rate:.3f}")
        print(f"Best reward: {self.best_reward:.2f}")
        
        # 最终保存
        self._save_checkpoint(episode, global_step, final=True)
        
        # 关闭环境
        self.env.close()
    
    def _collect_rollout(self):
        """收集一个episode的经验"""
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < self.config['max_episode_steps']:
            # 获取动作
            action, log_prob, value = self.agent.get_action(obs)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action[0])
            done = terminated or truncated
            
            # 存储经验
            self.agent.store_transition(obs, action[0], reward, value, log_prob, done)
            
            # 更新
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        episode_success = info.get('success', False)
        return episode_reward, episode_length, episode_success
    
    def _evaluate(self, n_episodes: int = 10):
        """评估当前策略"""
        print("Evaluating...")
        rewards = []
        successes = []
        
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < self.config['max_episode_steps']:
                # 确定性动作
                action, _, _ = self.agent.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action[0])
                done = terminated or truncated
                episode_reward += reward
                steps += 1
            
            rewards.append(episode_reward)
            successes.append(info.get('success', False))
        
        mean_reward = np.mean(rewards)
        success_rate = np.mean(successes)
        
        return mean_reward, success_rate
    
    def _save_checkpoint(self, episode: int, global_step: int, final: bool = False):
        """保存检查点"""
        suffix = 'final' if final else f'ep{episode}'
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{suffix}.pt')
        
        self.agent.save(checkpoint_path)
        
        # 保存训练统计
        stats_path = os.path.join(self.checkpoint_dir, f'stats_{suffix}.json')
        stats = {
            'episode': episode,
            'global_step': global_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'eval_rewards': self.eval_rewards,
            'eval_success_rates': self.eval_success_rates,
            'best_reward': self.best_reward,
            'best_success_rate': self.best_success_rate,
            'config': self.config
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_best_model(self, episode: int, global_step: int, reward: float, success_rate: float):
        """保存最佳模型"""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        self.agent.save(best_path)
        
        # 保存最佳模型信息
        best_info = {
            'episode': episode,
            'global_step': global_step,
            'reward': reward,
            'success_rate': success_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        info_path = os.path.join(self.checkpoint_dir, 'best_model_info.json')
        with open(info_path, 'w') as f:
            json.dump(best_info, f, indent=2)
        
        print(f"New best model saved! Success rate: {success_rate:.3f}, Reward: {reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Train PPO for robot grasping')
    
    # 环境参数
    parser.add_argument('--robot_type', type=str, default='panda', choices=['panda', 'ur5e'])
    parser.add_argument('--use_image_obs', action='store_true', help='Use image observations')
    parser.add_argument('--use_simple_env', action='store_true', help='Use simple environment (state-only)')
    parser.add_argument('--image_size', nargs=2, type=int, default=[84, 84])
    parser.add_argument('--max_episode_steps', type=int, default=500)
    parser.add_argument('--success_threshold', type=float, default=0.05)
    parser.add_argument('--randomize_target', action='store_true', default=True)
    parser.add_argument('--sparse_reward', action='store_true', help='Use sparse reward')
    
    # PPO参数
    parser.add_argument('--lr_actor', type=float, default=3e-4)
    parser.add_argument('--lr_critic', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=2048)
    
    # 训练参数
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto')
    
    # 文件路径
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--experiment_name', type=str, default=None)
    
    # wandb参数
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='robot_grasping_ppo')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_offline', action='store_true', default=True, help='Run wandb in offline mode')
    parser.add_argument('--wandb_online', action='store_true', help='Run wandb in online mode (overrides offline)')
    
    args = parser.parse_args()
    
    # 创建配置字典
    config = vars(args)
    
    # 设置实验名称
    if config['experiment_name'] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['experiment_name'] = f"ppo_{config['robot_type']}_{timestamp}"
    
    # 更新检查点目录
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], config['experiment_name'])
    
    # 初始化wandb
    if config['use_wandb']:
        # 确定wandb模式
        if config['wandb_online']:
            wandb_mode = "online"
        elif config['wandb_offline']:
            wandb_mode = "offline"
        else:
            wandb_mode = "online"  # 默认在线模式
            
        print(f"Wandb mode: {wandb_mode}")
        
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            name=config['experiment_name'],
            config=config,
            save_code=True,
            mode=wandb_mode
        )
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建训练器并开始训练
    trainer = PPOTrainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        if config['use_wandb']:
            wandb.finish()


if __name__ == '__main__':
    main()