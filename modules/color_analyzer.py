"""
modules/color_analyzer.py
Extract mean RGB + HSV values from ROI in each frame.

Author: University of Phayao | School of Science
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator, Dict
from dataclasses import dataclass


@dataclass
class ROI:
    """Region of Interest (rectangle)"""
    x: int        # left
    y: int        # top
    w: int        # width
    h: int        # height
    label: str = "ROI"

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """ตัดพื้นที่ ROI ออกจาก frame"""
        h_img, w_img = frame.shape[:2]
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(w_img, self.x + self.w)
        y2 = min(h_img, self.y + self.h)
        return frame[y1:y2, x1:x2]

    def is_valid(self, frame_shape: Tuple[int, int]) -> bool:
        """ตรวจว่า ROI อยู่ในกรอบ frame หรือไม่"""
        h_img, w_img = frame_shape[:2]
        return (
            self.w > 0 and self.h > 0
            and self.x >= 0 and self.y >= 0
            and self.x + self.w <= w_img
            and self.y + self.h <= h_img
        )

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "label": self.label}


def extract_mean_colors(
    frame_rgb: np.ndarray,
    roi: ROI,
    include_hsv: bool = True,
) -> Dict[str, float]:
    """ดึงค่าเฉลี่ยสีจาก ROI ใน frame

    Args:
        frame_rgb: เฟรมในระบบสี RGB (numpy array uint8)
        roi: พื้นที่ที่จะวิเคราะห์
        include_hsv: ถ้า True จะคำนวณ HSV เพิ่มเติม

    Returns:
        dict ที่มี R, G, B (และ H, S, V ถ้า include_hsv=True)
    """
    patch = roi.crop(frame_rgb)
    if patch.size == 0:
        # ROI invalid → คืน NaN
        result = {"R": np.nan, "G": np.nan, "B": np.nan}
        if include_hsv:
            result.update({"H": np.nan, "S": np.nan, "V": np.nan})
        return result

    # Mean RGB (per channel)
    mean_rgb = patch.reshape(-1, 3).mean(axis=0)
    result = {
        "R": float(mean_rgb[0]),
        "G": float(mean_rgb[1]),
        "B": float(mean_rgb[2]),
    }

    if include_hsv:
        # OpenCV ใช้ BGR → ต้องแปลง
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        # OpenCV: H=0-179, S=0-255, V=0-255
        # แปลง H ให้เป็น 0-360 (มาตรฐาน) เพื่อให้นักศึกษาเข้าใจง่าย
        mean_hsv = patch_hsv.reshape(-1, 3).mean(axis=0)
        result.update({
            "H": float(mean_hsv[0]) * 2.0,    # 0-360
            "S": float(mean_hsv[1]) / 255.0 * 100.0,  # 0-100 %
            "V": float(mean_hsv[2]) / 255.0 * 100.0,  # 0-100 %
        })

    return result


def analyze_video_frames(
    frame_iterator: Iterator[Tuple[float, np.ndarray]],
    roi: ROI,
    include_hsv: bool = True,
    progress_callback=None,
    total_estimate: int = 0,
) -> pd.DataFrame:
    """วิเคราะห์ทุก frame ที่ถูก yield จาก video_loader

    Args:
        frame_iterator: iterator จาก VideoLoader.iter_frames()
        roi: พื้นที่ที่จะวิเคราะห์
        include_hsv: รวม HSV ด้วยหรือไม่
        progress_callback: function(current, total) ที่จะเรียกทุก frame
                           สำหรับ Streamlit progress bar
        total_estimate: จำนวน frames ที่คาดว่าจะมี (ไว้ส่งให้ callback)

    Returns:
        DataFrame ที่มี columns: time_sec, R, G, B (และ H, S, V ถ้า include_hsv)
    """
    rows = []
    count = 0

    for timestamp, frame in frame_iterator:
        colors = extract_mean_colors(frame, roi, include_hsv=include_hsv)
        row = {"time_sec": timestamp, **colors}
        rows.append(row)
        count += 1

        if progress_callback:
            progress_callback(count, total_estimate)

    df = pd.DataFrame(rows)
    return df


def analyze_multi_roi(
    frame_iterator: Iterator[Tuple[float, np.ndarray]],
    rois: List[ROI],
    include_hsv: bool = True,
    progress_callback=None,
    total_estimate: int = 0,
) -> pd.DataFrame:
    """Advanced mode: วิเคราะห์หลาย ROI พร้อมกัน

    Returns:
        DataFrame ที่มี columns: time_sec, [label]_R, [label]_G, [label]_B, ...
        เช่น cell1_R, cell1_G, cell1_B, cell2_R, ...
    """
    rows = []
    count = 0

    for timestamp, frame in frame_iterator:
        row = {"time_sec": timestamp}
        for roi in rois:
            colors = extract_mean_colors(frame, roi, include_hsv=include_hsv)
            for key, val in colors.items():
                row[f"{roi.label}_{key}"] = val
        rows.append(row)
        count += 1

        if progress_callback:
            progress_callback(count, total_estimate)

    return pd.DataFrame(rows)


def rgb_to_absorbance(
    df: pd.DataFrame,
    baseline_time_range: Tuple[float, float] = None,
    baseline_indices: Tuple[int, int] = None,
    channels: List[str] = ["R", "G", "B"],
) -> pd.DataFrame:
    """แปลง RGB intensity เป็น Absorbance ตาม Beer-Lambert: A = -log10(I/I0)

    Args:
        df: DataFrame ที่ได้จาก analyze_video_frames()
        baseline_time_range: (t_start, t_end) ใช้คำนวณ I0 (เฉลี่ยใน range นี้)
        baseline_indices: (idx_start, idx_end) — ใช้แทน time_range ก็ได้
        channels: list ของ channel ที่จะแปลง เช่น ["R", "G", "B"]

    Returns:
        DataFrame ที่มี columns เพิ่ม: A_R, A_G, A_B
    """
    df = df.copy()

    # หา baseline I0 จาก time range หรือ indices
    if baseline_time_range is not None:
        t_start, t_end = baseline_time_range
        mask = (df["time_sec"] >= t_start) & (df["time_sec"] <= t_end)
        if not mask.any():
            raise ValueError(f"No frames in baseline range {baseline_time_range}")
        baseline = df.loc[mask, channels].mean()
    elif baseline_indices is not None:
        i_start, i_end = baseline_indices
        baseline = df.iloc[i_start:i_end][channels].mean()
    else:
        # default: ใช้ 5 frames แรก
        baseline = df.iloc[:5][channels].mean()

    # Beer-Lambert: A = -log10(I / I0)
    for ch in channels:
        I0 = baseline[ch]
        if I0 <= 0:
            df[f"A_{ch}"] = np.nan
            continue
        # หลีกเลี่ยง log(0): clip I ที่ค่าน้อยมากๆ
        I = df[ch].clip(lower=0.1)
        df[f"A_{ch}"] = -np.log10(I / I0)

    return df


def compute_complementary_channel(df: pd.DataFrame) -> pd.DataFrame:
    """คำนวณ "channel ที่ดีที่สุด" สำหรับวิเคราะห์ — ใช้ Absorbance สูงสุด
    เพื่อช่วยนักศึกษาตัดสินใจว่า R, G, หรือ B ตอบสนองต่อสารละลายดีที่สุด"""
    df = df.copy()
    if all(col in df.columns for col in ["A_R", "A_G", "A_B"]):
        # ใช้ค่าสุดท้ายเป็นตัวตัดสิน
        last_a = df.iloc[-1][["A_R", "A_G", "A_B"]]
        best = last_a.idxmax()  # "A_R", "A_G", หรือ "A_B"
        df.attrs["best_channel"] = best
    return df
