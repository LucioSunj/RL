import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import numpy as np


class CNN(nn.Module):
    """卷积神经网络用于图像特征提取"""
    
    def __init__(self, input_channels=3, output_dim=512):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.output_layer = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.output_layer(x)
        return x


class MLPExtractor(nn.Module):
    """MLP特征提取器"""
    
    def __init__(self, input_dim: int, hidden_dims=[256, 256], output_dim=256):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)


class MultiInputExtractor(nn.Module):
    """多输入特征提取器，处理图像和状态信息"""
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__()
        
        self.features_dim = features_dim
        extractors = {}
        
        total_concat_size = 0
        
        for key, subspace in observation_space.spaces.items():
            if key in ['image']:
                # 图像输入使用CNN
                extractors[key] = CNN(
                    input_channels=subspace.shape[0] if len(subspace.shape) == 3 else subspace.shape[2], 
                    output_dim=256
                )
                total_concat_size += 256
                
            elif key in ['depth']:
                # 深度图像使用简化CNN
                extractors[key] = CNN(
                    input_channels=1, 
                    output_dim=128
                )
                total_concat_size += 128
                
            else:
                # 其他状态信息使用MLP
                if len(subspace.shape) == 1:
                    input_dim = subspace.shape[0]
                else:
                    input_dim = np.prod(subspace.shape)
                    
                extractors[key] = MLPExtractor(
                    input_dim=input_dim,
                    hidden_dims=[64, 64],
                    output_dim=32
                )
                total_concat_size += 32
        
        self.extractors = nn.ModuleDict(extractors)
        
        # 最终融合层
        self.fusion = nn.Sequential(
            nn.Linear(total_concat_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_tensor_list = []
        
        for key, extractor in self.extractors.items():
            obs = observations[key]
            
            # 处理图像输入的维度
            if key in ['image', 'depth']:
                if len(obs.shape) == 3:  # (H, W, C) -> (C, H, W)
                    obs = obs.permute(2, 0, 1).unsqueeze(0)
                elif len(obs.shape) == 4:  # (B, H, W, C) -> (B, C, H, W)
                    obs = obs.permute(0, 3, 1, 2)
                    
                if key == 'depth':
                    obs = obs.unsqueeze(1) if len(obs.shape) == 3 else obs
                    
                # 归一化图像
                obs = obs.float() / 255.0
                
            encoded_tensor_list.append(extractor(obs))
        
        # 拼接所有特征
        concatenated = torch.cat(encoded_tensor_list, dim=-1)
        return self.fusion(concatenated)


class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络"""
    
    def __init__(self, 
                 observation_space: spaces.Dict,
                 action_space: spaces.Box,
                 features_dim: int = 512,
                 hidden_dims=[256, 256]):
        super().__init__()
        
        self.action_dim = action_space.shape[0]
        
        # 特征提取器
        self.feature_extractor = MultiInputExtractor(observation_space, features_dim)
        
        # 策略头
        policy_layers = []
        prev_dim = features_dim
        
        for hidden_dim in hidden_dims:
            policy_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        self.policy_net = nn.Sequential(*policy_layers)
        
        # 输出层
        self.mean_layer = nn.Linear(prev_dim, self.action_dim)
        self.log_std_layer = nn.Linear(prev_dim, self.action_dim)
        
        # 动作范围
        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(observations)
        policy_features = self.policy_net(features)
        
        mean = self.mean_layer(policy_features)
        log_std = self.log_std_layer(policy_features)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        
        return mean, log_std
    
    def sample(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(observations)
        std = log_std.exp()
        
        # 重参数化技巧
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        
        # Tanh变换到动作空间
        action = torch.tanh(x_t)
        
        # 计算log概率
        log_prob = normal.log_prob(x_t)
        # 修正tanh变换的雅可比行列式
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        # 缩放到实际动作空间
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob, mean
    
    def log_prob(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(observations)
        std = log_std.exp()
        
        # 反缩放动作
        actions_normalized = (actions - self.action_bias) / self.action_scale
        
        # 反tanh变换
        x_t = torch.atanh(torch.clamp(actions_normalized, -0.999, 0.999))
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(x_t)
        
        # 修正tanh变换的雅可比行列式
        log_prob -= torch.log(1 - actions_normalized.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return log_prob


class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络"""
    
    def __init__(self, 
                 observation_space: spaces.Dict,
                 features_dim: int = 512,
                 hidden_dims=[256, 256]):
        super().__init__()
        
        # 特征提取器
        self.feature_extractor = MultiInputExtractor(observation_space, features_dim)
        
        # 价值头
        value_layers = []
        prev_dim = features_dim
        
        for hidden_dim in hidden_dims:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        value_layers.append(nn.Linear(prev_dim, 1))
        
        self.value_net = nn.Sequential(*value_layers)
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.feature_extractor(observations)
        value = self.value_net(features)
        return value


class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, 
                 observation_space: spaces.Dict,
                 action_space: spaces.Box,
                 features_dim: int = 512,
                 hidden_dims=[256, 256]):
        super().__init__()
        
        self.actor = ActorNetwork(observation_space, action_space, features_dim, hidden_dims)
        self.critic = CriticNetwork(observation_space, features_dim, hidden_dims)
        
    def get_action(self, observations: Dict[str, torch.Tensor], deterministic: bool = False):
        if deterministic:
            mean, _ = self.actor.forward(observations)
            action = torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
            return action
        else:
            action, log_prob, _ = self.actor.sample(observations)
            return action, log_prob
    
    def get_value(self, observations: Dict[str, torch.Tensor]):
        return self.critic(observations)
    
    def evaluate(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor):
        value = self.critic(observations)
        log_prob = self.actor.log_prob(observations, actions)
        
        # 计算熵
        mean, log_std = self.actor.forward(observations)
        std = log_std.exp()
        entropy = (0.5 * (1 + 2 * log_std + np.log(2 * np.pi))).sum(-1, keepdim=True)
        
        return value, log_prob, entropy


def create_networks(observation_space: spaces.Dict, 
                   action_space: spaces.Box, 
                   features_dim: int = 512,
                   hidden_dims=[256, 256]) -> ActorCritic:
    """创建Actor-Critic网络"""
    return ActorCritic(observation_space, action_space, features_dim, hidden_dims)


if __name__ == '__main__':
    # 测试网络
    from rl_grasp_env import RLGraspEnv
    
    print("Testing networks...")
    
    # 创建测试环境
    env = RLGraspEnv(robot_type='panda', use_image_obs=True)
    
    # 创建网络
    net = create_networks(env.observation_space, env.action_space)
    
    print(f"Network created successfully!")
    print(f"Actor parameters: {sum(p.numel() for p in net.actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in net.critic.parameters()):,}")
    
    # 测试前向传播
    obs, _ = env.reset()
    
    # 转换为tensor
    obs_tensor = {}
    for key, value in obs.items():
        obs_tensor[key] = torch.FloatTensor(value).unsqueeze(0)
    
    # 测试网络
    with torch.no_grad():
        action, log_prob = net.get_action(obs_tensor)
        value = net.get_value(obs_tensor)
        
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Value shape: {value.shape}")
    print("Network test completed!")
    
    env.close()