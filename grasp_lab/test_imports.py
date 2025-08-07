#!/usr/bin/env python3
"""
测试导入是否正常的脚本
"""

def test_imports():
    """测试关键模块的导入"""
    print("🧪 开始测试导入...")
    
    try:
        print("1. 测试 manipulator_grasp.env.panda_grasp_env...")
        from manipulator_grasp.env.panda_grasp_env import PandaGraspEnv
        print("   ✅ PandaGraspEnv 导入成功")
    except Exception as e:
        print(f"   ❌ PandaGraspEnv 导入失败: {e}")
        return False
    
    try:
        print("2. 测试 manipulator_grasp.env.ur5_grasp_env...")
        from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
        print("   ✅ UR5GraspEnv 导入成功")
    except Exception as e:
        print(f"   ❌ UR5GraspEnv 导入失败: {e}")
        return False
    
    try:
        print("3. 测试 manipulator_grasp.arm.robot...")
        from manipulator_grasp.arm.robot import Robot, Panda, UR5e
        print("   ✅ Robot 类导入成功")
    except Exception as e:
        print(f"   ❌ Robot 类导入失败: {e}")
        return False
    
    try:
        print("4. 测试 rl_grasp_env...")
        from rl_grasp_env import RLGraspEnv
        print("   ✅ RLGraspEnv 导入成功")
    except Exception as e:
        print(f"   ❌ RLGraspEnv 导入失败: {e}")
        return False
    
    print("\n🎉 所有导入测试通过！")
    return True

if __name__ == "__main__":
    success = test_imports()
    if not success:
        print("\n⚠️  还有一些导入问题需要修复")
        exit(1)
    else:
        print("\n✅ 导入测试完成，可以开始训练了！")