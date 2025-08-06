"""
工具函数和辅助脚本
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import wandb.apis.public as wandb_api


def setup_directories():
    """设置必要的目录结构"""
    dirs = ['checkpoints', 'logs', 'videos', 'plots']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Directory '{dir_name}' ready")


def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = [
        'torch', 'torchvision', 'gymnasium', 'numpy', 
        'opencv-python', 'matplotlib', 'wandb'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("All dependencies are installed ✓")
        return True


def load_training_stats(stats_path: str) -> Dict:
    """加载训练统计信息"""
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats


def plot_training_curves(stats_path: str, save_path: Optional[str] = None):
    """绘制训练曲线"""
    stats = load_training_stats(stats_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PPO Training Curves', fontsize=16)
    
    # 奖励曲线
    axes[0, 0].plot(stats['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # 成功率曲线
    axes[0, 1].plot(stats['success_rates'])
    axes[0, 1].set_title('Success Rate')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].grid(True)
    
    # Episode长度
    axes[0, 2].plot(stats['episode_lengths'])
    axes[0, 2].set_title('Episode Lengths')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].grid(True)
    
    # 评估奖励
    if stats.get('eval_rewards'):
        eval_episodes = np.arange(0, len(stats['eval_rewards'])) * 50  # 假设每50episode评估一次
        axes[1, 0].plot(eval_episodes, stats['eval_rewards'])
        axes[1, 0].set_title('Evaluation Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Eval Reward')
        axes[1, 0].grid(True)
    
    # 评估成功率
    if stats.get('eval_success_rates'):
        eval_episodes = np.arange(0, len(stats['eval_success_rates'])) * 50
        axes[1, 1].plot(eval_episodes, stats['eval_success_rates'])
        axes[1, 1].set_title('Evaluation Success Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Eval Success Rate')
        axes[1, 1].grid(True)
    
    # 奖励分布
    axes[1, 2].hist(stats['episode_rewards'], bins=50, alpha=0.7)
    axes[1, 2].set_title('Reward Distribution')
    axes[1, 2].set_xlabel('Reward')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def analyze_model_performance(checkpoint_dir: str):
    """分析模型性能"""
    # 查找最新的统计文件
    stats_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('stats_') and f.endswith('.json')]
    
    if not stats_files:
        print(f"No stats files found in {checkpoint_dir}")
        return
    
    # 找到最新的文件
    latest_stats = max(stats_files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    stats_path = os.path.join(checkpoint_dir, latest_stats)
    
    print(f"Analyzing: {stats_path}")
    
    stats = load_training_stats(stats_path)
    
    # 基本统计
    rewards = np.array(stats['episode_rewards'])
    success_rates = np.array(stats['success_rates'])
    lengths = np.array(stats['episode_lengths'])
    
    print(f"\n{'='*50}")
    print("TRAINING ANALYSIS")
    print(f"{'='*50}")
    print(f"Total Episodes: {len(rewards)}")
    print(f"Total Steps: {stats['global_step']}")
    print(f"")
    print(f"Final Success Rate: {success_rates[-1]:.3f}")
    print(f"Best Success Rate: {np.max(success_rates):.3f}")
    print(f"")
    print(f"Final Reward (last 10 episodes): {np.mean(rewards[-10:]):.2f}")
    print(f"Best Reward (max): {np.max(rewards):.2f}")
    print(f"")
    print(f"Average Episode Length: {np.mean(lengths):.1f}")
    print(f"Final Episode Length (last 10): {np.mean(lengths[-10:]):.1f}")
    
    # 学习进度分析
    if len(rewards) >= 100:
        first_100_reward = np.mean(rewards[:100])
        last_100_reward = np.mean(rewards[-100:])
        improvement = last_100_reward - first_100_reward
        
        first_100_success = np.mean(success_rates[:100])
        last_100_success = np.mean(success_rates[-100:])
        success_improvement = last_100_success - first_100_success
        
        print(f"")
        print(f"Learning Progress:")
        print(f"  Reward Improvement: {improvement:+.2f}")
        print(f"  Success Rate Improvement: {success_improvement:+.3f}")
    
    # 生成图表
    plot_path = os.path.join(checkpoint_dir, 'training_curves.png')
    plot_training_curves(stats_path, plot_path)


def compare_models(model_paths: List[str], model_names: List[str] = None):
    """比较多个模型的性能"""
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(model_paths))]
    
    print(f"Comparing {len(model_paths)} models...")
    
    # 这里可以实现模型比较逻辑
    # 加载每个模型并在相同环境中测试
    pass


def download_wandb_runs(project_name: str, entity: str = None, save_dir: str = "./wandb_downloads"):
    """从wandb下载训练记录"""
    try:
        api = wandb_api.Api()
        
        if entity:
            runs = api.runs(f"{entity}/{project_name}")
        else:
            runs = api.runs(project_name)
        
        os.makedirs(save_dir, exist_ok=True)
        
        for run in runs:
            print(f"Downloading run: {run.name}")
            
            # 下载历史数据
            history = run.scan_history()
            data = []
            for row in history:
                data.append(row)
            
            # 保存数据
            save_path = os.path.join(save_dir, f"{run.name}_history.json")
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"  Saved to: {save_path}")
    
    except Exception as e:
        print(f"Error downloading from wandb: {e}")


def create_training_config(robot_type: str = 'panda', 
                          difficulty: str = 'easy',
                          use_image: bool = False) -> Dict:
    """创建训练配置"""
    
    base_config = {
        'robot_type': robot_type,
        'use_simple_env': not use_image,
        'use_image_obs': use_image,
        'randomize_target': True,
        'max_episode_steps': 500,
        'success_threshold': 0.05,
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'n_epochs': 10,
        'batch_size': 64,
        'buffer_size': 2048,
        'use_wandb': True
    }
    
    if difficulty == 'easy':
        base_config.update({
            'total_timesteps': 200000,
            'success_threshold': 0.08,
            'max_episode_steps': 300,
            'randomize_target': False
        })
    elif difficulty == 'medium':
        base_config.update({
            'total_timesteps': 500000,
            'success_threshold': 0.05,
            'max_episode_steps': 500
        })
    elif difficulty == 'hard':
        base_config.update({
            'total_timesteps': 1000000,
            'success_threshold': 0.03,
            'max_episode_steps': 800,
            'sparse_reward': True
        })
    
    if use_image:
        base_config.update({
            'batch_size': 32,
            'buffer_size': 4096,
            'total_timesteps': base_config['total_timesteps'] * 2
        })
    
    return base_config


def generate_training_command(config: Dict) -> str:
    """生成训练命令"""
    cmd = "python train_ppo.py"
    
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{key}"
        else:
            cmd += f" --{key} {value}"
    
    return cmd


def quick_setup(robot_type: str = 'panda', difficulty: str = 'easy', use_image: bool = False):
    """快速设置训练环境"""
    print(f"Setting up PPO training for {robot_type} robot...")
    print(f"Difficulty: {difficulty}")
    print(f"Use images: {use_image}")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 设置目录
    setup_directories()
    
    # 生成配置
    config = create_training_config(robot_type, difficulty, use_image)
    
    # 生成命令
    command = generate_training_command(config)
    
    print(f"\nGenerated training command:")
    print(command)
    
    # 保存配置
    config_path = f"config_{robot_type}_{difficulty}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_path}")
    print(f"\nTo start training, run:")
    print(command)


def main():
    """主函数 - 命令行工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO Training Utils')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 快速设置
    setup_parser = subparsers.add_parser('setup', help='Quick setup')
    setup_parser.add_argument('--robot_type', default='panda', choices=['panda', 'ur5e'])
    setup_parser.add_argument('--difficulty', default='easy', choices=['easy', 'medium', 'hard'])
    setup_parser.add_argument('--use_image', action='store_true')
    
    # 分析模型
    analyze_parser = subparsers.add_parser('analyze', help='Analyze model performance')
    analyze_parser.add_argument('checkpoint_dir', help='Checkpoint directory')
    
    # 绘制曲线
    plot_parser = subparsers.add_parser('plot', help='Plot training curves')
    plot_parser.add_argument('stats_file', help='Stats JSON file')
    plot_parser.add_argument('--save', help='Save path for plot')
    
    # 检查依赖
    subparsers.add_parser('check', help='Check dependencies')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        quick_setup(args.robot_type, args.difficulty, args.use_image)
    elif args.command == 'analyze':
        analyze_model_performance(args.checkpoint_dir)
    elif args.command == 'plot':
        plot_training_curves(args.stats_file, args.save)
    elif args.command == 'check':
        check_dependencies()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()