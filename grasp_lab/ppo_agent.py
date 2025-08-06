import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import copy

from network import ActorCritic


class RolloutBuffer:
    """经验缓冲区"""
    
    def __init__(self, buffer_size: int, observation_space, action_space, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # 初始化缓冲区
        self.observations = {}
        for key, space in observation_space.spaces.items():
            self.observations[key] = torch.zeros(
                (buffer_size, *space.shape), dtype=torch.float32, device=device
            )
        
        self.actions = torch.zeros((buffer_size, action_space.shape[0]), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        
    def add(self, obs: Dict, action: torch.Tensor, reward: float, value: torch.Tensor, 
            log_prob: torch.Tensor, done: bool):
        """添加经验"""
        for key, value_tensor in obs.items():
            if isinstance(value_tensor, np.ndarray):
                value_tensor = torch.FloatTensor(value_tensor)
            self.observations[key][self.ptr] = value_tensor.to(self.device)
        
        self.actions[self.ptr] = action.to(self.device)
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value.to(self.device)
        self.log_probs[self.ptr] = log_prob.to(self.device)
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_advantages_and_returns(self, last_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        """计算优势函数和回报"""
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        gae = 0
        next_value = last_value
        
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - self.dones[step].float()
            else:
                next_non_terminal = 1.0 - self.dones[step + 1].float()
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = advantages[step] + self.values[step]
            next_value = self.values[step]
        
        self.advantages = advantages
        self.returns = returns
        
        # 标准化优势函数
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def get_batch(self, batch_size: int):
        """获取批次数据"""
        indices = torch.randperm(self.size, device=self.device)[:batch_size]
        
        batch_obs = {}
        for key in self.observations:
            batch_obs[key] = self.observations[key][indices]
        
        return (
            batch_obs,
            self.actions[indices],
            self.values[indices],
            self.log_probs[indices],
            self.advantages[indices],
            self.returns[indices]
        )
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """PPO算法实现"""
    
    def __init__(self,
                 observation_space,
                 action_space,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: float = 0.01,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 buffer_size: int = 2048,
                 device: str = 'auto'):
        
        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"PPO Agent using device: {self.device}")
        
        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # 网络
        self.actor_critic = ActorCritic(observation_space, action_space).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.actor.parameters(), 'lr': lr_actor},
            {'params': self.actor_critic.critic.parameters(), 'lr': lr_critic}
        ])
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # 经验缓冲区
        self.buffer = RolloutBuffer(buffer_size, observation_space, action_space, self.device)
        
        # 统计信息
        self.total_steps = 0
        self.episode_count = 0
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100),
            'explained_variance': deque(maxlen=100),
            'clipfrac': deque(maxlen=100)
        }
        
    def get_action(self, observation: Dict, deterministic: bool = False):
        """获取动作"""
        with torch.no_grad():
            # 转换观察为tensor
            obs_tensor = self._obs_to_tensor(observation)
            
            if deterministic:
                action = self.actor_critic.get_action(obs_tensor, deterministic=True)
                return action.cpu().numpy(), None, None
            else:
                action, log_prob = self.actor_critic.get_action(obs_tensor, deterministic=False)
                value = self.actor_critic.get_value(obs_tensor)
                return action.cpu().numpy(), log_prob.cpu(), value.cpu()
    
    def store_transition(self, obs: Dict, action: np.ndarray, reward: float, 
                        value: torch.Tensor, log_prob: torch.Tensor, done: bool):
        """存储转换"""
        action_tensor = torch.FloatTensor(action)
        self.buffer.add(obs, action_tensor, reward, value, log_prob, done)
        self.total_steps += 1
    
    def update(self, last_observation: Dict) -> Dict:
        """更新网络"""
        # 计算最后一个状态的价值
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(last_observation)
            last_value = self.actor_critic.get_value(obs_tensor).cpu()
        
        # 计算优势函数和回报
        self.buffer.compute_advantages_and_returns(last_value, self.gamma, self.gae_lambda)
        
        # 训练
        update_stats = self._train_networks()
        
        # 清空缓冲区
        self.buffer.clear()
        
        return update_stats
    
    def _train_networks(self) -> Dict:
        """训练网络"""
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divergences = []
        clipfracs = []
        
        # 多轮更新
        for epoch in range(self.n_epochs):
            # 获取批次数据
            obs_batch, actions_batch, old_values_batch, old_log_probs_batch, \
            advantages_batch, returns_batch = self.buffer.get_batch(self.batch_size)
            
            # 当前网络输出
            values, log_probs, entropy = self.actor_critic.evaluate(obs_batch, actions_batch)
            
            # 计算比率
            ratio = torch.exp(log_probs - old_log_probs_batch)
            
            # 计算策略损失 (PPO-Clip)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            if self.value_loss_coef > 0:
                value_pred_clipped = old_values_batch + torch.clamp(
                    values - old_values_batch, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss1 = (values - returns_batch).pow(2)
                value_loss2 = (value_pred_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
            else:
                value_loss = 0.5 * (values - returns_batch).pow(2).mean()
            
            # 熵损失
            entropy_loss = -entropy.mean()
            
            # 总损失
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # 统计信息
            with torch.no_grad():
                kl_div = 0.5 * (log_probs - old_log_probs_batch).pow(2).mean()
                clipfrac = ((ratio - 1).abs() > self.clip_epsilon).float().mean()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(-entropy_loss.item())
                kl_divergences.append(kl_div.item())
                clipfracs.append(clipfrac.item())
            
            # 早停机制
            if kl_div > self.target_kl:
                print(f"Early stopping at epoch {epoch} due to reaching max KL: {kl_div:.4f}")
                break
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算解释方差
        explained_var = self._explained_variance(
            self.buffer.values[:self.buffer.size].cpu().numpy(),
            self.buffer.returns[:self.buffer.size].cpu().numpy()
        )
        
        # 更新统计信息
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'kl_divergence': np.mean(kl_divergences),
            'explained_variance': explained_var,
            'clipfrac': np.mean(clipfracs),
            'learning_rate_actor': self.optimizer.param_groups[0]['lr'],
            'learning_rate_critic': self.optimizer.param_groups[1]['lr']
        }
        
        # 更新历史统计
        for key, value in stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        return stats
    
    def _obs_to_tensor(self, observation: Dict) -> Dict:
        """转换观察为tensor"""
        obs_tensor = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
            else:
                obs_tensor[key] = torch.FloatTensor([value]).to(self.device)
        return obs_tensor
    
    def _explained_variance(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算解释方差"""
        var_y = np.var(y_true)
        return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'training_stats': dict(self.training_stats)
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        if 'training_stats' in checkpoint:
            for key, values in checkpoint['training_stats'].items():
                self.training_stats[key] = deque(values, maxlen=100)
        print(f"Model loaded from {filepath}")
    
    def get_stats(self) -> Dict:
        """获取训练统计信息"""
        stats = {}
        for key, values in self.training_stats.items():
            if len(values) > 0:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
        return stats


if __name__ == '__main__':
    # 测试PPO Agent
    from rl_grasp_env import RLGraspEnv
    
    print("Testing PPO Agent...")
    
    # 创建环境
    env = RLGraspEnv(robot_type='panda', use_image_obs=False)  # 先测试不带图像的版本
    
    # 创建agent
    agent = PPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        batch_size=32,
        buffer_size=128
    )
    
    print("Agent created successfully!")
    
    # 测试一个episode
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(50):
        action, log_prob, value = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action[0])
        
        agent.store_transition(obs, action[0], reward, value, log_prob, terminated)
        
        obs = next_obs
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode completed with total reward: {total_reward:.3f}")
    
    # 测试更新
    if agent.buffer.size > 0:
        update_stats = agent.update(obs)
        print(f"Update stats: {update_stats}")
    
    env.close()
    print("PPO Agent test completed!")