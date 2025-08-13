#!/usr/bin/env python3
"""
测试服务器环境下的训练
"""

import os
import sys

# 添加路径
manipulator_grasp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp')
if manipulator_grasp_path not in sys.path:
    sys.path.insert(0, manipulator_grasp_path)

def test_server_environment():
    """测试服务器环境"""
    print("Testing server environment...")
    
    # 检查环境变量
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'Not set')}")
    print(f"XDG_SESSION_TYPE: {os.environ.get('XDG_SESSION_TYPE', 'Not set')}")
    print(f"SSH_CONNECTION: {os.environ.get('SSH_CONNECTION', 'Not set')}")
    print(f"TERM: {os.environ.get('TERM', 'Not set')}")
    print(f"MUJOCO_GL: {os.environ.get('MUJOCO_GL', 'Not set')}")
    
    try:
        from improved_grasp_env import ImprovedGraspEnv
        print("✓ Successfully imported ImprovedGraspEnv")
        
        # 创建环境
        env = ImprovedGraspEnv(robot_type='panda', difficulty='easy', headless=True)
        print("✓ Successfully created environment")
        
        # 重置环境
        obs, info = env.reset()
        print("✓ Successfully reset environment")
        
        # 测试几步
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✓ Step {step}: reward={reward:.2f}, phase={info.get('phase', 0)}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        env.close()
        print("✓ Environment closed successfully")
        
        print("\n🎉 Server environment test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_import():
    """测试训练脚本导入"""
    print("\nTesting training script import...")
    
    try:
        from train_improved_ppo import ImprovedPPOTrainer
        print("✓ Successfully imported ImprovedPPOTrainer")
        
        # 创建简单的配置
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
            'disable_video': True
        }
        
        trainer = ImprovedPPOTrainer(config)
        print("✓ Successfully created trainer")
        
        print("\n🎉 Training script test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Training script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Server Environment Test Suite")
    print("=" * 40)
    
    # 运行测试
    test1_result = test_server_environment()
    test2_result = test_training_import()
    
    # 总结
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"  Server environment: {'PASS' if test1_result else 'FAIL'}")
    print(f"  Training script: {'PASS' if test2_result else 'FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All tests passed! Ready for training.")
        print("You can now run: bash run_improved_training.sh")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    return test1_result and test2_result

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 