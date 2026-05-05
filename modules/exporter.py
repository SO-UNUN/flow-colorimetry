"""
modules/exporter.py
Export utilities for analysis results (CSV, PNG, JSON).

Author: University of Phayao | School of Science
"""

import io
import json
import pandas as pd
from typing import Dict, Any
from datetime import datetime


def df_to_csv_bytes(df: pd.DataFrame, filename_hint: str = "data") -> bytes:
    """แปลง DataFrame เป็น CSV bytes สำหรับ Streamlit download_button"""
    return df.to_csv(index=False).encode("utf-8-sig")  # utf-8-sig = ใช้กับ Excel ภาษาไทยได้


def make_filename(prefix: str, ext: str = "csv") -> str:
    """สร้างชื่อไฟล์พร้อม timestamp เพื่อไม่ทับกัน"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{ext}"


def figure_to_png_bytes(fig) -> bytes:
    """แปลง Plotly figure เป็น PNG bytes (ต้องใช้ kaleido)
    ถ้า kaleido ไม่ติดตั้ง จะคืน HTML แทน"""
    try:
        return fig.to_image(format="png", width=1200, height=600, scale=2)
    except Exception:
        # fallback: HTML
        return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


def figure_to_html_bytes(fig) -> bytes:
    """แปลง Plotly figure เป็น standalone HTML"""
    return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


def make_metadata_json(
    video_meta: dict,
    roi_info: dict,
    sampling_fps: float,
    settings: Dict[str, Any] = None,
) -> bytes:
    """สร้างไฟล์ metadata JSON เพื่อบันทึกค่าที่ใช้ในการวิเคราะห์
    มีประโยชน์สำหรับนักศึกษาบันทึกในรายงาน lab"""
    metadata = {
        "analysis_timestamp": datetime.now().isoformat(),
        "video": video_meta,
        "roi": roi_info,
        "sampling_fps": sampling_fps,
        "settings": settings or {},
        "tool": "Flow Cell Colorimetry — University of Phayao",
    }
    return json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")


def build_summary_text(
    video_meta: dict,
    df: pd.DataFrame,
    plateaus: list = None,
) -> str:
    """สร้าง summary text สำหรับแสดงในแอป"""
    lines = []
    lines.append(f"Video: {video_meta.get('filename', 'N/A')}")
    lines.append(f"Duration: {video_meta.get('duration_min', 0):.2f} min "
                 f"({video_meta.get('duration_sec', 0):.1f} sec)")
    lines.append(f"FPS: {video_meta.get('fps', 0)}")
    lines.append(f"Resolution: {video_meta.get('width', 0)} × {video_meta.get('height', 0)}")
    lines.append(f"Total frames analyzed: {len(df)}")

    if "R" in df.columns:
        lines.append("")
        lines.append("RGB summary:")
        for ch in ["R", "G", "B"]:
            if ch in df.columns:
                lines.append(f"  {ch}: mean={df[ch].mean():.1f}, "
                           f"min={df[ch].min():.1f}, max={df[ch].max():.1f}")

    if plateaus:
        lines.append("")
        lines.append(f"Plateaus detected: {len(plateaus)}")

    return "\n".join(lines)
