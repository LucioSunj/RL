#!/usr/bin/env python3
"""
简单的导入测试 - 验证修复是否有效
"""

import os
import sys

# 添加路径
manipulator_grasp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp')
if manipulator_grasp_path not in sys.path:
    sys.path.insert(0, manipulator_grasp_path)

def test_import():
    """测试导入"""
    try:
        print("Testing import...")
        from improved_grasp_env import ImprovedGraspEnv
        print("✓ Successfully imported ImprovedGraspEnv")
        
        print("Creating environment...")
        env = ImprovedGraspEnv(robot_type='panda', difficulty='easy', headless=True)
        print("✓ Successfully created environment")
        
        print("Resetting environment...")
        obs, info = env.reset()
        print("✓ Successfully reset environment")
        
        print("Testing render...")
        render_output = env.render()
        print(f"✓ Render output: {type(render_output)}")
        
        print("Testing step...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step successful, reward: {reward:.2f}")
        
        env.close()
        print("✓ Environment closed successfully")
        
        print("\n🎉 All tests passed! The fix is working.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_import()
    sys.exit(0 if success else 1) 