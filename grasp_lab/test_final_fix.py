#!/usr/bin/env python3
"""
最终修复验证测试
"""

import os
import sys

# 设置环境变量以模拟服务器环境
os.environ['MUJOCO_GL'] = 'none'
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']

# 添加路径
manipulator_grasp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp')
if manipulator_grasp_path not in sys.path:
    sys.path.insert(0, manipulator_grasp_path)

def test_environment_creation():
    """测试环境创建"""
    print("Testing environment creation...")
    
    try:
        from improved_grasp_env import ImprovedGraspEnv
        
        # 创建环境
        env = ImprovedGraspEnv(robot_type='panda', difficulty='easy', headless=True)
        print("✓ Environment created successfully")
        
        # 重置环境
        obs, info = env.reset()
        print("✓ Environment reset successfully")
        
        # 测试几步
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✓ Step {step}: reward={reward:.2f}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("✓ Environment closed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script():
    """测试训练脚本"""
    print("\nTesting training script...")
    
    try:
        from train_improved_ppo import ImprovedPPOTrainer
        
        # 创建配置
        config = {
            'robot_type': 'panda',
            'difficulty': 'easy',
            'max_episode_steps': 100,
            'use_image_obs': False,
            'headless': True,
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.05,
            'value_loss_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': 0.02,
            'n_epochs': 8,
            'batch_size': 64,
            'buffer_size': 2048,
            'total_timesteps': 1000,
            'log_interval': 10,
            'save_interval': 100,
            'eval_interval': 50,
            'device': 'auto',
            'checkpoint_dir': './test_checkpoints',
            'use_wandb': False,
            'disable_video': True,
            'video_config': {'interval': 999999, 'max_per_eval': 0, 'fps': 30}
        }
        
        trainer = ImprovedPPOTrainer(config)
        print("✓ Trainer created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Training script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Final Fix Verification Test")
    print("=" * 40)
    
    # 检查环境变量
    print(f"MUJOCO_GL: {os.environ.get('MUJOCO_GL', 'Not set')}")
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'Not set')}")
    
    # 运行测试
    test1_result = test_environment_creation()
    test2_result = test_training_script()
    
    # 总结
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"  Environment creation: {'PASS' if test1_result else 'FAIL'}")
    print(f"  Training script: {'PASS' if test2_result else 'FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All tests passed! The fix is working.")
        print("You can now run training with: bash run_improved_training.sh")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    return test1_result and test2_result

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 