#!/usr/bin/env python3
"""
测试视频录制功能
"""

import os
import sys
import numpy as np

# 添加路径
manipulator_grasp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manipulator_grasp')
if manipulator_grasp_path not in sys.path:
    sys.path.insert(0, manipulator_grasp_path)

from improved_grasp_env import ImprovedGraspEnv
from video_recorder import VideoRecorder, EvaluationVideoManager


def test_basic_video_recording():
    """测试基础视频录制功能"""
    print("Testing basic video recording...")
    
    # 创建环境
    env = ImprovedGraspEnv(robot_type='panda', difficulty='easy', headless=False)
    
    # 创建视频录制器
    recorder = VideoRecorder('./test_videos', fps=30)
    
    try:
        # 开始录制
        recorder.start_recording()
        
        # 运行一个简短的episode
        obs, info = env.reset()
        
        for step in range(50):  # 录制50帧
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 获取渲染帧
            try:
                render_output = env.render()
                if isinstance(render_output, dict):
                    frame = render_output.get('img', None)
                else:
                    frame = render_output
                    
                if frame is not None:
                    recorder.add_frame(frame)
                    if step % 10 == 0:
                        print(f"  Recorded frame {step}, shape: {frame.shape}")
                else:
                    print(f"  Warning: No frame at step {step}")
                    
            except Exception as e:
                print(f"  Error capturing frame at step {step}: {e}")
            
            if terminated or truncated:
                break
        
        # 保存视频
        video_path = recorder.stop_recording("test_basic.mp4")
        print(f"Video saved: {video_path}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        env.close()


def test_evaluation_video_manager():
    """测试evaluation视频管理器"""
    print("\nTesting evaluation video manager...")
    
    # 创建管理器
    manager = EvaluationVideoManager('./test_videos', video_interval=2, max_videos_per_eval=2)
    
    # 创建环境
    env = ImprovedGraspEnv(robot_type='panda', difficulty='easy', headless=False)
    
    try:
        # 模拟多次evaluation
        for eval_num in range(5):
            print(f"\nEvaluation {eval_num + 1}:")
            should_record = manager.start_evaluation()
            print(f"  Should record: {should_record}")
            
            if should_record:
                # 录制几个episodes
                for ep in range(3):
                    if manager.start_episode_recording(ep):
                        print(f"    Recording episode {ep}")
                        
                        # 运行短episode
                        obs, info = env.reset()
                        for step in range(20):
                            action = env.action_space.sample()
                            obs, reward, terminated, truncated, info = env.step(action)
                            
                            # 录制帧
                            try:
                                render_output = env.render()
                                if isinstance(render_output, dict):
                                    frame = render_output.get('img', None)
                                else:
                                    frame = render_output
                                    
                                if frame is not None:
                                    manager.add_frame(frame)
                            except Exception as e:
                                print(f"      Frame capture error: {e}")
                            
                            if terminated or truncated:
                                break
                        
                        # 完成录制
                        episode_info = {
                            'success': ep == 0,  # 第一个成功
                            'reward': 50.0 if ep == 0 else -10.0,
                            'length': step + 1
                        }
                        video_path = manager.finish_episode_recording(episode_info)
                        if video_path:
                            print(f"      Saved: {video_path}")
        
        # 显示摘要
        summary = manager.get_evaluation_summary()
        print(f"\nSummary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        env.close()


def main():
    """主测试函数"""
    print("Video Recording Test Suite")
    print("=" * 40)
    
    # 创建测试目录
    os.makedirs('./test_videos', exist_ok=True)
    
    # 运行测试
    test1_result = test_basic_video_recording()
    test2_result = test_evaluation_video_manager()
    
    # 总结
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"  Basic video recording: {'PASS' if test1_result else 'FAIL'}")
    print(f"  Evaluation manager: {'PASS' if test2_result else 'FAIL'}")
    
    if test1_result and test2_result:
        print("\nAll tests passed! Video recording is ready.")
        print("Check ./test_videos/ for generated test videos.")
    else:
        print("\nSome tests failed. Check the error messages above.")
    
    return test1_result and test2_result


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)