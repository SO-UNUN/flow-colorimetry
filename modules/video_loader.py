"""
modules/video_loader.py
Video loading and frame extraction using OpenCV.

Author: University of Phayao | School of Science
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional


class VideoLoader:
    """โหลดไฟล์วิดีโอและ extract frames ตาม sampling rate ที่กำหนด"""

    def __init__(self, video_path: str):
        """
        Args:
            video_path: path ไปยังไฟล์วิดีโอ (.mp4, .mov, .avi)
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # Read metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0

    def get_metadata(self) -> dict:
        """คืนข้อมูล metadata ของวิดีโอ"""
        return {
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration_sec": round(self.duration_sec, 2),
            "duration_min": round(self.duration_sec / 60, 2),
            "filename": self.video_path.name,
        }

    def get_first_frame(self) -> Optional[np.ndarray]:
        """ดึง frame แรกของวิดีโอเพื่อใช้เป็น preview สำหรับเลือก ROI
        คืนเป็น RGB (ไม่ใช่ BGR ตาม OpenCV default)"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            return None
        # แปลง BGR → RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame_at(self, time_sec: float) -> Optional[np.ndarray]:
        """ดึง frame ที่เวลา time_sec (วินาที) คืน RGB"""
        if time_sec < 0 or time_sec > self.duration_sec:
            return None
        frame_idx = int(time_sec * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def iter_frames(
        self, sampling_fps: float = 1.0
    ) -> Iterator[Tuple[float, np.ndarray]]:
        """Generator ที่ yield (timestamp_sec, frame_rgb) ตาม sampling rate

        Args:
            sampling_fps: จำนวน frames ที่ต้องการต่อวินาที
                          เช่น 1.0 = 1 frame/sec (เหมาะกับ flow ที่เปลี่ยนช้า)
                          0.5 = 1 frame ทุก 2 วินาที (ลด data ลง)
        Yields:
            (timestamp ในวินาที, frame array RGB)
        """
        if sampling_fps <= 0:
            raise ValueError("sampling_fps must be > 0")

        # คำนวณ frame interval
        frame_skip = max(1, int(round(self.fps / sampling_fps)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                timestamp = frame_idx / self.fps
                yield timestamp, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_idx += 1

    def estimate_total_samples(self, sampling_fps: float) -> int:
        """ประมาณจำนวน frames ที่จะได้จาก iter_frames() — ใช้สำหรับ progress bar"""
        if sampling_fps <= 0 or self.fps <= 0:
            return 0
        return int(self.duration_sec * sampling_fps)

    def close(self):
        """ปิดไฟล์วิดีโอ"""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def get_supported_formats() -> list:
    """คืน list ของ extensions ที่ Streamlit file_uploader ควรยอมรับ"""
    return ["mp4", "mov", "avi", "mkv", "webm"]
