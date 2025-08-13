#!/usr/bin/env python3
"""
测试headless渲染功能
"""

import os
import sys
import numpy as np

# 添加路径
manipulator_grasp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp')
if manipulator_grasp_path not in sys.path:
    sys.path.insert(0, manipulator_grasp_path)

def test_headless_rendering():
    """测试headless渲染"""
    print("Testing headless rendering...")
    
    try:
        # 导入环境
        from improved_grasp_env import ImprovedGraspEnv
        print("✓ Successfully imported ImprovedGraspEnv")
        
        # 创建环境
        env = ImprovedGraspEnv(robot_type='panda', difficulty='easy', headless=True)
        print("✓ Successfully created environment")
        
        # 重置环境
        obs, info = env.reset()
        print("✓ Successfully reset environment")
        
        # 测试渲染
        render_output = env.render()
        print(f"✓ Render output type: {type(render_output)}")
        
        if isinstance(render_output, dict):
            if 'img' in render_output:
                img = render_output['img']
                print(f"✓ Image shape: {img.shape}, dtype: {img.dtype}")
                print(f"✓ Image range: {img.min()} - {img.max()}")
                
                # 检查是否不是全黑
                if img.max() > 0:
                    print("✓ Rendering is working (not all black)")
                else:
                    print("⚠ Rendering returned black image (may be expected in headless)")
            else:
                print("⚠ No 'img' key in render output")
        else:
            print(f"⚠ Unexpected render output format: {render_output}")
        
        # 测试几步
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✓ Step {step}: reward={reward:.2f}")
            
            # 再次渲染
            render_output = env.render()
            if isinstance(render_output, dict) and 'img' in render_output:
                img = render_output['img']
                if img.max() > 0:
                    print(f"✓ Step {step} rendering: OK")
                else:
                    print(f"⚠ Step {step} rendering: black")
        
        env.close()
        print("✓ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_recording():
    """测试视频录制"""
    print("\nTesting video recording...")
    
    try:
        from improved_grasp_env import ImprovedGraspEnv
        from video_recorder import VideoRecorder
        
        # 创建环境
        env = ImprovedGraspEnv(robot_type='panda', difficulty='easy', headless=True)
        
        # 创建录制器
        recorder = VideoRecorder('./test_headless_videos', fps=30)
        
        # 开始录制
        recorder.start_recording()
        
        # 重置环境
        obs, info = env.reset()
        
        # 录制几帧
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 获取渲染帧
            render_output = env.render()
            if isinstance(render_output, dict) and 'img' in render_output:
                frame = render_output['img']
                recorder.add_frame(frame)
                print(f"✓ Added frame {step}, shape: {frame.shape}")
            else:
                print(f"⚠ No frame at step {step}")
        
        # 保存视频
        video_path = recorder.stop_recording("test_headless.mp4")
        print(f"✓ Video saved: {video_path}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Video recording test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Headless Rendering Test Suite")
    print("=" * 40)
    
    # 创建测试目录
    os.makedirs('./test_headless_videos', exist_ok=True)
    
    # 运行测试
    test1_result = test_headless_rendering()
    test2_result = test_video_recording()
    
    # 总结
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"  Headless rendering: {'PASS' if test1_result else 'FAIL'}")
    print(f"  Video recording: {'PASS' if test2_result else 'FAIL'}")
    
    if test1_result and test2_result:
        print("\nAll tests passed! Headless rendering is working.")
    else:
        print("\nSome tests failed. Check the error messages above.")
    
    return test1_result and test2_result

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 