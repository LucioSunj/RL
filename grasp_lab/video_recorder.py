"""
视频录制工具类 - 用于evaluation期间的视频输出
"""

import cv2
import os
import numpy as np
from typing import List, Optional
from datetime import datetime


class VideoRecorder:
    """视频录制器 - 用于记录evaluation episodes"""
    
    def __init__(self, output_dir: str = './evaluation_videos', fps: int = 30):
        self.output_dir = output_dir
        self.fps = fps
        self.frames = []
        self.recording = False
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def start_recording(self):
        """开始录制"""
        self.frames = []
        self.recording = True
        
    def add_frame(self, frame: np.ndarray):
        """添加帧到录制中"""
        if self.recording and frame is not None:
            # 确保frame是uint8格式
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            self.frames.append(frame)
    
    def stop_recording(self, filename: Optional[str] = None) -> str:
        """停止录制并保存视频"""
        if not self.recording or not self.frames:
            return ""
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.mp4"
            
        filepath = os.path.join(self.output_dir, filename)
        success = self._save_video(self.frames, filepath)
        
        self.recording = False
        self.frames = []
        
        return filepath if success else ""
    
    def _save_video(self, frames: List[np.ndarray], filepath: str) -> bool:
        """保存视频文件"""
        if not frames:
            return False
            
        try:
            # 获取帧尺寸
            height, width = frames[0].shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
            
            for frame in frames:
                # 处理颜色通道
                if len(frame.shape) == 3:
                    # RGB -> BGR (OpenCV格式)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)
            
            out.release()
            print(f"Video saved: {filepath} ({len(frames)} frames)")
            return True
            
        except Exception as e:
            print(f"Error saving video: {e}")
            return False


class EvaluationVideoManager:
    """Evaluation视频管理器 - 控制何时录制视频"""
    
    def __init__(self, 
                 output_dir: str = './evaluation_videos',
                 video_interval: int = 5,  # 每5次evaluation录制一次
                 max_videos_per_eval: int = 3,  # 每次evaluation最多录制3个episodes
                 fps: int = 30):
        
        self.video_interval = video_interval
        self.max_videos_per_eval = max_videos_per_eval
        self.evaluation_count = 0
        
        # 创建视频录制器
        self.recorder = VideoRecorder(output_dir, fps)
        
        # 创建子目录
        self.base_output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def should_record_video(self) -> bool:
        """判断当前evaluation是否应该录制视频"""
        return (self.evaluation_count % self.video_interval) == 0
    
    def start_evaluation(self) -> bool:
        """开始新的evaluation，返回是否需要录制视频"""
        self.evaluation_count += 1
        should_record = self.should_record_video()
        
        if should_record:
            # 为这次evaluation创建专门的目录
            eval_dir = os.path.join(
                self.base_output_dir, 
                f"eval_{self.evaluation_count:04d}"
            )
            os.makedirs(eval_dir, exist_ok=True)
            self.current_eval_dir = eval_dir
            print(f"Evaluation {self.evaluation_count}: Recording videos to {eval_dir}")
        
        return should_record
    
    def start_episode_recording(self, episode_idx: int) -> bool:
        """开始录制episode，返回是否成功开始"""
        if not self.should_record_video():
            return False
            
        if episode_idx >= self.max_videos_per_eval:
            return False
            
        self.recorder.start_recording()
        self.current_episode_idx = episode_idx
        return True
    
    def add_frame(self, frame: np.ndarray):
        """添加帧"""
        self.recorder.add_frame(frame)
    
    def finish_episode_recording(self, episode_info: dict) -> str:
        """完成episode录制"""
        if not self.recorder.recording:
            return ""
            
        # 生成文件名
        success = episode_info.get('success', False)
        reward = episode_info.get('reward', 0.0)
        length = episode_info.get('length', 0)
        
        filename = f"ep_{self.current_episode_idx:02d}_" \
                  f"{'success' if success else 'fail'}_" \
                  f"r{reward:.1f}_len{length}.mp4"
        
        # 保存到当前evaluation目录
        self.recorder.output_dir = self.current_eval_dir
        filepath = self.recorder.stop_recording(filename)
        
        return filepath
    
    def get_evaluation_summary(self) -> dict:
        """获取evaluation录制摘要"""
        return {
            'evaluation_count': self.evaluation_count,
            'video_interval': self.video_interval,
            'last_video_eval': self.evaluation_count if self.should_record_video() else 
                              (self.evaluation_count // self.video_interval) * self.video_interval,
            'next_video_eval': ((self.evaluation_count // self.video_interval) + 1) * self.video_interval
        }


def create_evaluation_video_manager(config: dict) -> EvaluationVideoManager:
    """创建evaluation视频管理器"""
    
    # 从配置中获取参数
    video_config = config.get('video_config', {})
    
    output_dir = os.path.join(
        config.get('checkpoint_dir', './checkpoints'),
        'evaluation_videos'
    )
    
    return EvaluationVideoManager(
        output_dir=output_dir,
        video_interval=video_config.get('interval', 5),
        max_videos_per_eval=video_config.get('max_per_eval', 3),
        fps=video_config.get('fps', 30)
    )


if __name__ == '__main__':
    # 测试视频管理器
    manager = EvaluationVideoManager('./test_videos', video_interval=2)
    
    # 模拟多次evaluation
    for eval_num in range(8):
        should_record = manager.start_evaluation()
        print(f"Evaluation {eval_num + 1}: Record video = {should_record}")
        
        if should_record:
            # 模拟录制几个episodes
            for ep in range(3):
                if manager.start_episode_recording(ep):
                    print(f"  Recording episode {ep}")
                    # 模拟添加一些帧
                    for frame_idx in range(10):
                        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                        manager.add_frame(dummy_frame)
                    
                    # 完成录制
                    episode_info = {
                        'success': ep < 2,  # 前两个成功
                        'reward': 100.0 if ep < 2 else -10.0,
                        'length': 50 + ep * 10
                    }
                    filepath = manager.finish_episode_recording(episode_info)
                    print(f"    Saved: {filepath}")
    
    print("\nSummary:", manager.get_evaluation_summary())