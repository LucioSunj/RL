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
manipulator_grasp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp')
if manipulator_grasp_path not in sys.path:
    sys.path.insert(0, manipulator_grasp_path)

from grasp_task_env import GraspTaskEnv
from ppo_agent import PPOAgent


class GraspTaskTrainer:
    """完整抓取任务的PPO训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 创建完整抓取任务环境
        self.env = GraspTaskEnv(
            robot_type=config['robot_type'],
            image_size=tuple(config['image_size']),
            max_episode_steps=config['max_episode_steps'],
            use_image_obs=config['use_image_obs'],
            randomize_object=config['randomize_object'],
            sparse_reward=config['sparse_reward'],
            headless=config['headless']
        )
        
        # 创建PPO agent
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
        self.phase_completion_rates = {
            'approach': [],
            'grasp': [],
            'lift': [],
            'transport': [],
            'place': []
        }
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
        print("Starting Grasp Task PPO training...")
        print(f"Environment: {type(self.env).__name__}")
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")
        print(f"Device: {self.agent.device}")
        print(f"Task: Pick and Place with {self.config['robot_type']} robot")
        
        start_time = time.time()
        global_step = 0
        episode = 0
        
        while global_step < self.config['total_timesteps']:
            # 收集经验
            episode_stats = self._collect_rollout()
            
            episode += 1
            global_step = self.agent.total_steps
            
            # 记录episode统计
            self.episode_rewards.append(episode_stats['reward'])
            self.episode_lengths.append(episode_stats['length'])
            
            # 记录阶段完成情况
            for phase, completed in episode_stats['phase_completion'].items():
                self.phase_completion_rates[phase].append(completed)
            
            # 计算成功率 (最近100个episode)
            if hasattr(self, '_recent_successes'):
                self._recent_successes.append(episode_stats['success'])
                if len(self._recent_successes) > 100:
                    self._recent_successes.pop(0)
            else:
                self._recent_successes = [episode_stats['success']]
            
            current_success_rate = np.mean(self._recent_successes)
            self.success_rates.append(current_success_rate)
            
            # 更新网络
            if self.agent.buffer.size >= self.config['buffer_size']:
                obs, _ = self.env.reset()  # 获取最后观察用于价值估计
                update_stats = self.agent.update(obs)
                
                # 计算阶段完成率（最近100个episode）
                recent_phase_rates = {}
                for phase, rates in self.phase_completion_rates.items():
                    recent_rates = rates[-100:] if len(rates) >= 100 else rates
                    recent_phase_rates[f'{phase}_completion_rate'] = np.mean(recent_rates) if recent_rates else 0.0
                
                # 记录训练统计
                log_data = {
                    'episode': episode,
                    'global_step': global_step,
                    'episode_reward': episode_stats['reward'],
                    'episode_length': episode_stats['length'],
                    'episode_success': episode_stats['success'],
                    'success_rate': current_success_rate,
                    'final_phase': episode_stats['final_phase'],
                    'max_phase_reached': episode_stats['max_phase_reached'],
                    'mean_reward_100': np.mean(self.episode_rewards[-100:]),
                    'std_reward_100': np.std(self.episode_rewards[-100:]),
                    **recent_phase_rates,
                    **update_stats
                }
                
                # wandb记录
                if self.config['use_wandb']:
                    wandb.log(log_data, step=global_step)
                
                # 控制台输出
                if episode % self.config['log_interval'] == 0:
                    elapsed_time = time.time() - start_time
                    fps = global_step / elapsed_time
                    
                    print(f"\n{'='*80}")
                    print(f"Episode: {episode:6d} | Step: {global_step:8d}")
                    print(f"Reward: {episode_stats['reward']:8.2f} | Success: {episode_stats['success']}")
                    print(f"Success Rate: {current_success_rate:6.3f} | Mean Reward: {log_data['mean_reward_100']:8.2f}")
                    print(f"Final Phase: {episode_stats['final_phase']} | Max Phase: {episode_stats['max_phase_reached']}")
                    print(f"Phase Completion Rates:")
                    for phase, rate in recent_phase_rates.items():
                        print(f"  {phase}: {rate:.3f}")
                    print(f"Policy Loss: {update_stats['policy_loss']:8.4f} | Value Loss: {update_stats['value_loss']:8.4f}")
                    print(f"Entropy: {update_stats['entropy']:8.4f} | KL Div: {update_stats['kl_divergence']:8.4f}")
                    print(f"FPS: {fps:.1f} | Time: {elapsed_time/60:.1f}m")
                
                # 保存检查点
                if episode % self.config['save_interval'] == 0:
                    self._save_checkpoint(episode, global_step)
                
                # 评估
                if episode % self.config['eval_interval'] == 0:
                    eval_stats = self._evaluate()
                    self.eval_rewards.append(eval_stats['mean_reward'])
                    self.eval_success_rates.append(eval_stats['success_rate'])
                    
                    eval_log = {
                        'eval_reward': eval_stats['mean_reward'],
                        'eval_success_rate': eval_stats['success_rate'],
                        'eval_episode': episode,
                        'eval_mean_final_phase': eval_stats['mean_final_phase'],
                        'eval_max_phase_rate': eval_stats['max_phase_rate']
                    }
                    
                    if self.config['use_wandb']:
                        wandb.log(eval_log, step=global_step)
                    
                    print(f"\n{'='*50}")
                    print(f"EVALUATION - Episode {episode}")
                    print(f"Eval Reward: {eval_stats['mean_reward']:.2f}")
                    print(f"Eval Success Rate: {eval_stats['success_rate']:.3f}")
                    print(f"Eval Mean Final Phase: {eval_stats['mean_final_phase']:.1f}")
                    print(f"Eval Max Phase Rate: {eval_stats['max_phase_rate']:.3f}")
                    print(f"{'='*50}\n")
                    
                    # 保存最佳模型
                    if eval_stats['success_rate'] > self.best_success_rate:
                        self.best_success_rate = eval_stats['success_rate']
                        self.best_reward = eval_stats['mean_reward']
                        self._save_best_model(episode, global_step, 
                                            eval_stats['mean_reward'], eval_stats['success_rate'])
        
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
        max_phase_reached = 0
        phase_completion = {
            'approach': False,
            'grasp': False,
            'lift': False,
            'transport': False,
            'place': False
        }
        
        while not done and episode_length < self.config['max_episode_steps']:
            # 获取动作
            action, log_prob, value = self.agent.get_action(obs)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action[0])
            done = terminated or truncated
            
            # 存储经验
            self.agent.store_transition(obs, action[0], reward, value, log_prob, done)
            
            # 更新统计
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # 记录最高达到的阶段
            current_phase = info['task_phase']
            max_phase_reached = max(max_phase_reached, current_phase)
            
            # 记录阶段完成情况
            phase_names = ['approach', 'grasp', 'lift', 'transport', 'place']
            for i, phase_name in enumerate(phase_names):
                if current_phase > i:
                    phase_completion[phase_name] = True
        
        episode_success = info.get('task_completed', False)
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'success': episode_success,
            'final_phase': info['task_phase'],
            'max_phase_reached': max_phase_reached,
            'phase_completion': phase_completion
        }
    
    def _evaluate(self, n_episodes: int = 10):
        """评估当前策略"""
        print("Evaluating...")
        rewards = []
        successes = []
        final_phases = []
        max_phases = []
        
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            max_phase_reached = 0
            
            while not done and steps < self.config['max_episode_steps']:
                # 确定性动作
                action, _, _ = self.agent.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action[0])
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                
                # 记录最高达到的阶段
                max_phase_reached = max(max_phase_reached, info['task_phase'])
            
            rewards.append(episode_reward)
            successes.append(info.get('task_completed', False))
            final_phases.append(info['task_phase'])
            max_phases.append(max_phase_reached)
        
        return {
            'mean_reward': np.mean(rewards),
            'success_rate': np.mean(successes),
            'mean_final_phase': np.mean(final_phases),
            'max_phase_rate': np.mean([phase >= 4 for phase in max_phases])  # 达到放置阶段的比例
        }
    
    def _save_checkpoint(self, episode: int, global_step: int, final: bool = False):
        """保存检查点"""
        suffix = 'final' if final else f'ep{episode}'
        checkpoint_path = os.path.join(self.checkpoint_dir, f'grasp_task_checkpoint_{suffix}.pt')
        
        self.agent.save(checkpoint_path)
        
        # 保存训练统计
        stats_path = os.path.join(self.checkpoint_dir, f'grasp_task_stats_{suffix}.json')
        stats = {
            'episode': episode,
            'global_step': global_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'phase_completion_rates': self.phase_completion_rates,
            'eval_rewards': self.eval_rewards,
            'eval_success_rates': self.eval_success_rates,
            'best_reward': self.best_reward,
            'best_success_rate': self.best_success_rate,
            'config': self.config
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Grasp task checkpoint saved: {checkpoint_path}")
    
    def _save_best_model(self, episode: int, global_step: int, reward: float, success_rate: float):
        """保存最佳模型"""
        best_path = os.path.join(self.checkpoint_dir, 'best_grasp_task_model.pt')
        self.agent.save(best_path)
        
        # 保存最佳模型信息
        best_info = {
            'episode': episode,
            'global_step': global_step,
            'reward': reward,
            'success_rate': success_rate,
            'timestamp': datetime.now().isoformat(),
            'task_type': 'grasp_task'
        }
        
        info_path = os.path.join(self.checkpoint_dir, 'best_grasp_task_model_info.json')
        with open(info_path, 'w') as f:
            json.dump(best_info, f, indent=2)
        
        print(f"New best grasp task model saved! Success rate: {success_rate:.3f}, Reward: {reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Train PPO for complete grasping task')
    
    # 环境参数
    parser.add_argument('--robot_type', type=str, default='panda', choices=['panda', 'ur5e'])
    parser.add_argument('--use_image_obs', action='store_true', help='Use image observations')
    parser.add_argument('--image_size', nargs=2, type=int, default=[84, 84])
    parser.add_argument('--max_episode_steps', type=int, default=2000, help='Max steps per episode for complete task')
    parser.add_argument('--randomize_object', action='store_true', help='Randomize object position')
    parser.add_argument('--sparse_reward', action='store_true', help='Use sparse reward')
    parser.add_argument('--headless', action='store_true', default=True, help='Run in headless mode')
    parser.add_argument('--with_display', action='store_true', help='Run with display')
    
    # PPO参数 (调整为适合复杂任务)
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='Lower LR for complex task')
    parser.add_argument('--lr_critic', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.02, help='Higher entropy for exploration')
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--target_kl', type=float, default=0.015)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32, help='Smaller batch for complex obs')
    parser.add_argument('--buffer_size', type=int, default=4096, help='Larger buffer for long episodes')
    
    # 训练参数
    parser.add_argument('--total_timesteps', type=int, default=2000000, help='More timesteps for complex task')
    parser.add_argument('--log_interval', type=int, default=5, help='More frequent logging')
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=25)
    parser.add_argument('--device', type=str, default='auto')
    
    # 文件路径
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--experiment_name', type=str, default=None)
    
    # wandb参数
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='robot_grasp_task_ppo')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_offline', action='store_true', default=True)
    parser.add_argument('--wandb_online', action='store_true', help='Run wandb online')
    
    args = parser.parse_args()
    
    # 创建配置字典
    config = vars(args)
    
    # 处理headless模式设置
    if config['with_display']:
        config['headless'] = False
        print("Display mode enabled - rendering will be available")
    elif config['headless']:
        print("Headless mode enabled - no rendering for better performance")
    
    # 设置实验名称
    if config['experiment_name'] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['experiment_name'] = f"grasp_task_{config['robot_type']}_{timestamp}"
    
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
            wandb_mode = "online"
            
        print(f"Wandb mode: {wandb_mode}")
        
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            name=config['experiment_name'],
            config=config,
            save_code=True,
            mode=wandb_mode
        )
    
    print("=" * 80)
    print("COMPLETE GRASP TASK TRAINING")
    print("=" * 80)
    print("Task: Pick up cube and place it in target zone")
    print("Phases: Approach → Grasp → Lift → Transport → Place")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # 创建训练器并开始训练
    trainer = GraspTaskTrainer(config)
    
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