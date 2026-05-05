"""
modules/signal_processor.py
Signal smoothing and plateau detection for flow colorimetry data.

Author: University of Phayao | School of Science
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from typing import List, Tuple, Dict


def smooth_savgol(
    series: pd.Series,
    window_size: int = 11,
    polyorder: int = 3,
) -> pd.Series:
    """Smooth ข้อมูลด้วย Savitzky-Golay filter

    เหมาะกับ time series ที่มี noise — preserves peak shape ได้ดีกว่า moving average

    Args:
        series: pandas Series (เช่น df["R"])
        window_size: จำนวน points ที่ใช้ smooth (ต้องเป็นเลขคี่)
        polyorder: order ของ polynomial (ปกติ 2 หรือ 3)

    Returns:
        Series ที่ smooth แล้ว
    """
    n = len(series)
    if n < window_size:
        # data น้อยเกินไป ใช้ moving average แทน
        return series.rolling(window=min(5, n), center=True, min_periods=1).mean()

    # window_size ต้องเป็นเลขคี่
    if window_size % 2 == 0:
        window_size += 1

    if polyorder >= window_size:
        polyorder = window_size - 1

    smoothed = savgol_filter(series.values, window_size, polyorder)
    return pd.Series(smoothed, index=series.index, name=series.name)


def smooth_dataframe(
    df: pd.DataFrame,
    columns: List[str] = None,
    window_size: int = 11,
    polyorder: int = 3,
    suffix: str = "_smooth",
) -> pd.DataFrame:
    """Smooth หลาย columns พร้อมกัน → เพิ่ม columns ใหม่ด้วย suffix

    Args:
        df: DataFrame ต้นฉบับ
        columns: list ของ columns ที่จะ smooth (default: R, G, B และ H, S, V ถ้ามี)
        window_size, polyorder: ส่งให้ smooth_savgol()
        suffix: ต่อท้ายชื่อ column เช่น "R" → "R_smooth"

    Returns:
        DataFrame ที่มี columns เดิม + columns smooth
    """
    df = df.copy()
    if columns is None:
        # auto-detect: smooth ทุก channel ที่เป็นตัวเลข ยกเว้น time
        candidates = ["R", "G", "B", "H", "S", "V", "A_R", "A_G", "A_B"]
        columns = [c for c in candidates if c in df.columns]

    for col in columns:
        if col in df.columns:
            df[f"{col}{suffix}"] = smooth_savgol(df[col], window_size, polyorder)

    return df


def detect_plateaus(
    series: pd.Series,
    time: pd.Series,
    derivative_threshold: float = 0.5,
    min_duration_sec: float = 30.0,
    smooth_window: int = 21,
) -> List[Dict]:
    """ตรวจจับช่วง plateau (ความเข้มข้นคงที่) ในสัญญาณ

    หลักการ: คำนวณอนุพันธ์ของสัญญาณ → ช่วงที่อนุพันธ์ใกล้ 0 = plateau

    Args:
        series: ค่าสัญญาณ (เช่น df["R"])
        time: เวลา (วินาที) — ขนาดเท่ากับ series
        derivative_threshold: |dy/dt| < threshold ถือว่าเป็น plateau
                              ค่าน้อย = เข้มงวด, ค่ามาก = หลวม
        min_duration_sec: plateau ต้องยาวอย่างน้อยกี่วินาที
        smooth_window: window size สำหรับ smooth ก่อนหา derivative

    Returns:
        List ของ dict: [{"start_time": ..., "end_time": ..., "mean": ..., "std": ...}, ...]
    """
    if len(series) < smooth_window + 5:
        return []

    # 1. Smooth signal ก่อน เพื่อลด noise ในการหา derivative
    smoothed = smooth_savgol(series, smooth_window, 3)

    # 2. คำนวณ derivative (dy/dt) ด้วย finite difference
    dt = np.gradient(time.values)
    dy = np.gradient(smoothed.values)
    derivative = np.abs(dy / np.where(dt > 0, dt, 1e-9))

    # 3. หา indices ที่ derivative ต่ำกว่า threshold (= plateau candidate)
    is_plateau = derivative < derivative_threshold

    # 4. หา contiguous regions
    plateaus = []
    in_plateau = False
    start_idx = 0

    for i, p in enumerate(is_plateau):
        if p and not in_plateau:
            in_plateau = True
            start_idx = i
        elif not p and in_plateau:
            in_plateau = False
            duration = time.iloc[i - 1] - time.iloc[start_idx]
            if duration >= min_duration_sec:
                segment = series.iloc[start_idx:i]
                plateaus.append({
                    "start_time": float(time.iloc[start_idx]),
                    "end_time": float(time.iloc[i - 1]),
                    "duration_sec": float(duration),
                    "mean": float(segment.mean()),
                    "std": float(segment.std()),
                    "n_points": int(i - start_idx),
                })

    # ปิดท้ายถ้า plateau ขยายไปจนสุดข้อมูล
    if in_plateau:
        duration = time.iloc[-1] - time.iloc[start_idx]
        if duration >= min_duration_sec:
            segment = series.iloc[start_idx:]
            plateaus.append({
                "start_time": float(time.iloc[start_idx]),
                "end_time": float(time.iloc[-1]),
                "duration_sec": float(duration),
                "mean": float(segment.mean()),
                "std": float(segment.std()),
                "n_points": int(len(series) - start_idx),
            })

    return plateaus


def auto_segment_by_plateaus(
    df: pd.DataFrame,
    target_column: str = "R",
    derivative_threshold: float = 0.5,
    min_duration_sec: float = 30.0,
    smooth_window: int = 21,
) -> pd.DataFrame:
    """แบ่ง DataFrame ออกเป็น segments ตาม plateau ที่ตรวจพบ

    Returns:
        DataFrame ที่มี column "segment" (1, 2, 3, ...) และ "is_plateau" (bool)
        plateau ใหม่ = segment ใหม่
    """
    df = df.copy()
    plateaus = detect_plateaus(
        df[target_column],
        df["time_sec"],
        derivative_threshold=derivative_threshold,
        min_duration_sec=min_duration_sec,
        smooth_window=smooth_window,
    )

    df["segment"] = 0
    df["is_plateau"] = False

    for i, p in enumerate(plateaus, start=1):
        mask = (df["time_sec"] >= p["start_time"]) & (df["time_sec"] <= p["end_time"])
        df.loc[mask, "segment"] = i
        df.loc[mask, "is_plateau"] = True

    return df


def summarize_plateaus(plateaus: List[Dict]) -> pd.DataFrame:
    """แปลง list ของ plateau เป็น DataFrame เพื่อแสดงเป็นตาราง"""
    if not plateaus:
        return pd.DataFrame(columns=[
            "Segment", "Start (s)", "End (s)", "Duration (s)", "Mean", "Std", "N points"
        ])

    rows = []
    for i, p in enumerate(plateaus, start=1):
        rows.append({
            "Segment": i,
            "Start (s)": round(p["start_time"], 2),
            "End (s)": round(p["end_time"], 2),
            "Duration (s)": round(p["duration_sec"], 2),
            "Mean": round(p["mean"], 3),
            "Std": round(p["std"], 3),
            "N points": p["n_points"],
        })

    return pd.DataFrame(rows)
