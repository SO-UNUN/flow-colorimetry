"""
modules/calibration.py
Linear regression for calibration curve (Beer-Lambert: A = ε·l·c).
Advanced mode: predict unknown concentration from absorbance.

Author: University of Phayao | School of Science
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """ผลลัพธ์ของการทำ calibration curve"""
    slope: float          # ε·l (จากสมการ A = ε·l·c)
    intercept: float      # baseline offset
    r_squared: float      # R² coefficient of determination
    n_points: int         # จำนวนจุด calibration
    channel: str          # ช่องสีที่ใช้ (เช่น "A_G")
    concentrations: List[float]   # ค่า x (known concentrations)
    absorbances: List[float]      # ค่า y (measured absorbances)

    def predict(self, absorbance: float) -> float:
        """คำนวณ concentration จาก absorbance ที่วัดได้"""
        if self.slope == 0:
            return float("nan")
        return (absorbance - self.intercept) / self.slope

    def predict_array(self, absorbances: np.ndarray) -> np.ndarray:
        """คำนวณ concentration จาก array ของ absorbances"""
        if self.slope == 0:
            return np.full_like(absorbances, np.nan)
        return (absorbances - self.intercept) / self.slope

    def equation_str(self) -> str:
        """สมการ calibration ในรูป string สำหรับแสดงใน UI"""
        sign = "+" if self.intercept >= 0 else "-"
        return f"A = {self.slope:.4f} · C {sign} {abs(self.intercept):.4f}"

    def to_dict(self) -> Dict:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "n_points": self.n_points,
            "channel": self.channel,
            "equation": self.equation_str(),
        }


def linear_regression(
    concentrations: List[float],
    absorbances: List[float],
    channel: str = "A_G",
) -> CalibrationResult:
    """ทำ linear regression แบบ least squares: A = slope·C + intercept

    Args:
        concentrations: ค่าความเข้มข้นที่ทราบ (independent variable, x)
        absorbances: ค่า absorbance ที่วัดได้ (dependent variable, y)
        channel: ชื่อ channel ที่ใช้ (สำหรับ label เท่านั้น)

    Returns:
        CalibrationResult — มี slope, intercept, R², และ predict() method
    """
    if len(concentrations) != len(absorbances):
        raise ValueError("concentrations และ absorbances ต้องยาวเท่ากัน")
    if len(concentrations) < 2:
        raise ValueError("ต้องมีอย่างน้อย 2 จุดสำหรับ linear regression")

    x = np.array(concentrations, dtype=float)
    y = np.array(absorbances, dtype=float)
    n = len(x)

    # Least squares
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x * x).sum()

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-12:
        # x ทุกค่าเหมือนกัน → fit ไม่ได้
        return CalibrationResult(
            slope=0.0, intercept=float(y.mean()), r_squared=0.0,
            n_points=n, channel=channel,
            concentrations=concentrations, absorbances=absorbances,
        )

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R² calculation
    y_mean = y.mean()
    y_pred = slope * x + intercept
    ss_total = ((y - y_mean) ** 2).sum()
    ss_residual = ((y - y_pred) ** 2).sum()
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 0.0

    return CalibrationResult(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        n_points=n,
        channel=channel,
        concentrations=list(concentrations),
        absorbances=list(absorbances),
    )


def get_calibration_quality(r_squared: float) -> Tuple[str, str]:
    """แปล R² เป็นข้อความเชิงคุณภาพ + สี (สำหรับ UI)

    Returns:
        (text, color_emoji)
    """
    if r_squared >= 0.99:
        return ("ดีเยี่ยม (Excellent)", "🟢")
    elif r_squared >= 0.95:
        return ("ดี (Good)", "🔵")
    elif r_squared >= 0.90:
        return ("พอใช้ (Fair)", "🟡")
    else:
        return ("แย่ (Poor) — ควรทำซ้ำ", "🔴")


def predict_unknown(
    cal: CalibrationResult,
    df: pd.DataFrame,
    abs_column: Optional[str] = None,
) -> pd.DataFrame:
    """ใช้ calibration curve คำนวณความเข้มข้นตลอดทั้ง time series

    Args:
        cal: CalibrationResult จาก linear_regression()
        df: DataFrame ที่มี column absorbance (เช่น "A_R", "A_G", "A_B")
        abs_column: ชื่อ column ที่จะใช้ (default: cal.channel)

    Returns:
        DataFrame ที่มี column "concentration" เพิ่มขึ้น
    """
    df = df.copy()
    col = abs_column or cal.channel
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    df["concentration"] = cal.predict_array(df[col].values)
    return df


def plateau_to_calibration_points(
    plateaus: List[Dict],
    known_concentrations: List[float],
    df: pd.DataFrame,
    abs_column: str = "A_G",
) -> Tuple[List[float], List[float]]:
    """แปลง plateau ที่ตรวจพบ → จุด calibration

    หลักการ: แต่ละ plateau = ความเข้มข้นหนึ่งค่า (ที่นักศึกษาทราบล่วงหน้า)
    เช่น flow ใส่ standard 0.1 M → 0.2 M → 0.3 M → ... ตามลำดับเวลา

    Args:
        plateaus: list จาก signal_processor.detect_plateaus()
        known_concentrations: ค่าความเข้มข้นที่ทราบ ตามลำดับเวลา
                              ต้องมีจำนวนเท่ากับ plateaus
        df: DataFrame ต้นฉบับ ที่มี column absorbance
        abs_column: column ที่จะใช้ดึงค่า A เฉลี่ยในแต่ละ plateau

    Returns:
        (concentrations, absorbances) — พร้อมส่งให้ linear_regression()
    """
    if len(plateaus) != len(known_concentrations):
        raise ValueError(
            f"จำนวน plateaus ({len(plateaus)}) ไม่ตรงกับ "
            f"known_concentrations ({len(known_concentrations)})"
        )

    absorbances = []
    for p in plateaus:
        mask = (df["time_sec"] >= p["start_time"]) & (df["time_sec"] <= p["end_time"])
        if not mask.any() or abs_column not in df.columns:
            absorbances.append(float("nan"))
        else:
            absorbances.append(float(df.loc[mask, abs_column].mean()))

    return list(known_concentrations), absorbances
