"""
app.py
Flow Cell Colorimetry — Web App (Streamlit)

Analyze color changes in flowing solutions from smartphone videos.
University of Phayao | School of Science |

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import tempfile
import os

from modules.video_loader import VideoLoader, get_supported_formats
from modules.color_analyzer import (
    ROI, analyze_video_frames, analyze_multi_roi,
    rgb_to_absorbance, compute_complementary_channel,
)
from modules.signal_processor import (
    smooth_dataframe, detect_plateaus, summarize_plateaus,
)
from modules.calibration import (
    linear_regression, get_calibration_quality,
    predict_unknown, plateau_to_calibration_points,
)
from modules.exporter import (
    df_to_csv_bytes, make_filename, make_metadata_json, build_summary_text,
)

# ----------------------------------------------------------------
# Page config
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Flow Cell Colorimetry — UP",
    page_icon="🧪",
    layout="wide",
)

# ----------------------------------------------------------------
# CSS — UP Navy & Gold theme
# ----------------------------------------------------------------
st.markdown("""
<style>
.main-title {
    background: linear-gradient(135deg, #1a2745 0%, #2d4373 100%);
    color: #e6c870;
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #e6c870;
    margin-bottom: 1.2rem;
}
.main-title h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 600;
}
.main-title p {
    margin: 0.3rem 0 0 0;
    color: #b8c5dc;
    font-size: 0.9rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 18px;
    border-radius: 6px 6px 0 0;
}
div[data-testid="metric-container"] {
    background: rgba(230, 200, 112, 0.05);
    border: 1px solid rgba(230, 200, 112, 0.2);
    border-radius: 8px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# Header
# ----------------------------------------------------------------
st.markdown("""
<div class="main-title">
  <h1>🧪 Flow Cell Colorimetry</h1>
  <p>Smartphone-based RGB/HSV video analysis · University of Phayao · School of Science</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------
ss = st.session_state
for key, default in [
    ("video_path", None),
    ("video_meta", None),
    ("first_frame", None),
    ("roi", None),
    ("df_raw", None),
    ("df_processed", None),
    ("plateaus", None),
    ("calibration", None),
    ("analysis_done", False),
]:
    if key not in ss:
        ss[key] = default

# ----------------------------------------------------------------
# SIDEBAR — Mode + Settings
# ----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    mode = st.radio(
        "Analysis Mode",
        ["Core", "Core + Plus", "Advanced"],
        index=0,
        help=(
            "**Core** — RGB/HSV plot ตามเวลา (เร็ว, ใช้งานทั่วไป)\n\n"
            "**Core + Plus** — เพิ่ม smoothing + plateau detection\n\n"
            "**Advanced** — เพิ่ม Beer-Lambert + Calibration curve"
        ),
    )
    is_plus = mode in ("Core + Plus", "Advanced")
    is_advanced = mode == "Advanced"

    st.divider()
    st.subheader("Sampling")
    sampling_fps = st.select_slider(
        "Frames per second to analyze",
        options=[0.5, 1.0, 2.0, 5.0, 6.0, 10.0, 12.0, 15.0],
        value=5.0,
        help="Flow ที่เปลี่ยนช้า ใช้ 1 fps ก็เพียงพอ — ลด data, เพิ่มความเร็ว",
    )

    include_hsv = st.checkbox(
        "Analyze HSV channels",
        value=True,
        help="Hue ช่วยแยกสีของสารละลายชัดเจนกว่า RGB ในบางกรณี",
    )

    if is_plus:
        st.divider()
        st.subheader("Smoothing (Plus)")
        smooth_window = st.slider("Savitzky-Golay window", 5, 51, 11, step=2)
        smooth_polyorder = st.slider("Polynomial order", 1, 5, 3)

        st.divider()
        st.subheader("Plateau Detection")
        deriv_threshold = st.number_input(
            "Derivative threshold",
            min_value=0.01, max_value=10.0, value=0.5, step=0.1,
            help="|dy/dt| ต่ำกว่านี้ = plateau",
        )
        min_plateau_sec = st.slider(
            "Minimum plateau duration (sec)", 5, 300, 30, step=5,
        )
        plateau_channel = st.selectbox(
            "Channel for plateau detection",
            ["R", "G", "B", "H", "S", "V"],
            index=1,  # G default
        )

# ----------------------------------------------------------------
# MAIN — Tabs
# ----------------------------------------------------------------
tab_upload, tab_analyze, tab_results = st.tabs([
    "📤 1. Upload & ROI",
    "🔬 2. Analyze",
    "📊 3. Results",
])

# ================================================================
# TAB 1 — Upload + ROI selection
# ================================================================
with tab_upload:
    col_up, col_meta = st.columns([2, 1])

    with col_up:
        st.subheader("Upload video file")
        uploaded = st.file_uploader(
            "เลือกไฟล์วิดีโอ (.mp4, .mov, .avi)",
            type=get_supported_formats(),
            help="ไฟล์จะถูกประมวลผลในเครื่อง ไม่อัปโหลดที่อื่น",
        )

        if uploaded is not None:
            # Save to temp file
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                ss.video_path = tmp.name

            # Load metadata + first frame
            try:
                with VideoLoader(ss.video_path) as loader:
                    ss.video_meta = loader.get_metadata()
                    ss.first_frame = loader.get_first_frame()
                ss.video_meta["filename"] = uploaded.name
                st.success(f"✅ Loaded: {uploaded.name}")
            except Exception as e:
                st.error(f"Cannot load video: {e}")

    with col_meta:
        st.subheader("Video info")
        if ss.video_meta:
            m = ss.video_meta
            st.metric("Duration", f"{m['duration_min']:.2f} min")
            st.metric("FPS", f"{m['fps']}")
            st.metric("Resolution", f"{m['width']} × {m['height']}")
            st.metric("Total frames", f"{m['total_frames']:,}")

            est_samples = int(m['duration_sec'] * sampling_fps)
            st.info(f"จะวิเคราะห์ ~{est_samples:,} frames ที่ {sampling_fps} fps")
        else:
            st.info("⬅ อัปโหลดวิดีโอเพื่อดูข้อมูล")

    st.divider()

    # ROI Selection
    st.subheader("📐 Select Region of Interest (ROI)")

    if ss.first_frame is None:
        st.info("⬆ อัปโหลดวิดีโอก่อนเลือก ROI")
    else:
        st.caption("ลากเมาส์บนภาพเพื่อเลือกพื้นที่ที่จะวิเคราะห์ (ปกติเลือกพื้นที่กลาง flow cell)")

        try:
            from streamlit_drawable_canvas import st_canvas

            # Resize for display if too large
            img = Image.fromarray(ss.first_frame)
            max_w = 700
            disp_w = min(max_w, img.width)
            disp_h = int(img.height * disp_w / img.width)

            canvas_result = st_canvas(
                fill_color="rgba(230, 200, 112, 0.2)",
                stroke_width=2,
                stroke_color="#e6c870",
                background_image=img,
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                drawing_mode="rect",
                key="canvas_roi",
            )

            # Scale factor (canvas → original)
            scale_x = img.width / disp_w
            scale_y = img.height / disp_h

            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                last_rect = canvas_result.json_data["objects"][-1]
                x = int(last_rect["left"] * scale_x)
                y = int(last_rect["top"] * scale_y)
                w = int(last_rect["width"] * scale_x)
                h = int(last_rect["height"] * scale_y)

                ss.roi = ROI(x=x, y=y, w=w, h=h, label="cell")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("X", x)
                col_b.metric("Y", y)
                col_c.metric("Width", w)
                col_d.metric("Height", h)
                st.success(f"✅ ROI selected: ({x}, {y}, {w}×{h})")

        except ImportError:
            st.warning(
                "📦 streamlit-drawable-canvas ยังไม่ติดตั้ง — ใช้ manual input แทน"
            )
            col_a, col_b, col_c, col_d = st.columns(4)
            x = col_a.number_input("X", 0, ss.first_frame.shape[1], 100)
            y = col_b.number_input("Y", 0, ss.first_frame.shape[0], 100)
            w = col_c.number_input("Width", 10, ss.first_frame.shape[1], 200)
            h = col_d.number_input("Height", 10, ss.first_frame.shape[0], 200)
            ss.roi = ROI(x=int(x), y=int(y), w=int(w), h=int(h), label="cell")
            st.image(ss.first_frame, caption="First frame", use_column_width=True)


# ================================================================
# TAB 2 — Analyze
# ================================================================
with tab_analyze:
    if ss.video_path is None:
        st.info("⬆ Upload video first")
    elif ss.roi is None:
        st.info("⬆ Select ROI first")
    else:
        st.subheader("🔬 Run Analysis")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.write(f"**Mode:** {mode}")
            st.write(f"**Sampling:** {sampling_fps} fps")
            st.write(f"**ROI:** ({ss.roi.x}, {ss.roi.y}, {ss.roi.w}×{ss.roi.h})")
            st.write(f"**HSV:** {'Yes' if include_hsv else 'No'}")

        with col_right:
            run_btn = st.button("🚀 START ANALYSIS", type="primary",
                                use_container_width=True)

        if run_btn:
            progress = st.progress(0, text="กำลังวิเคราะห์...")
            status = st.empty()

            with VideoLoader(ss.video_path) as loader:
                total_est = loader.estimate_total_samples(sampling_fps)

                def cb(current, total):
                    if total > 0:
                        pct = min(1.0, current / total)
                        progress.progress(pct, text=f"Frame {current}/{total}")

                df = analyze_video_frames(
                    loader.iter_frames(sampling_fps=sampling_fps),
                    ss.roi,
                    include_hsv=include_hsv,
                    progress_callback=cb,
                    total_estimate=total_est,
                )

            progress.progress(1.0, text="✅ Done!")
            ss.df_raw = df
            status.success(f"วิเคราะห์เสร็จสิ้น — {len(df)} frames")

            # Plus: smoothing + plateau detection
            if is_plus:
                df = smooth_dataframe(df, window_size=smooth_window,
                                       polyorder=smooth_polyorder)
                if plateau_channel in df.columns:
                    plateaus = detect_plateaus(
                        df[plateau_channel],
                        df["time_sec"],
                        derivative_threshold=deriv_threshold,
                        min_duration_sec=min_plateau_sec,
                        smooth_window=smooth_window,
                    )
                    ss.plateaus = plateaus
                else:
                    ss.plateaus = []

            # Advanced: Beer-Lambert (auto baseline = first 5 frames)
            if is_advanced:
                df = rgb_to_absorbance(
                    df,
                    baseline_indices=(0, 5),
                    channels=["R", "G", "B"],
                )
                df = compute_complementary_channel(df)

            ss.df_processed = df
            ss.analysis_done = True
            st.rerun()

        # Show preview if done
        if ss.analysis_done and ss.df_processed is not None:
            st.divider()
            st.success("✅ Analysis complete — go to **Results** tab")
            st.dataframe(ss.df_processed.head(10), use_container_width=True)


# ================================================================
# TAB 3 — Results
# ================================================================
with tab_results:
    if not ss.analysis_done or ss.df_processed is None:
        st.info("⬅ Run analysis first")
    else:
        df = ss.df_processed

        # ------------- Plot RGB ----------
        st.subheader("📈 RGB over Time")
        fig_rgb = go.Figure()
        for ch, color in [("R", "#ef4444"), ("G", "#22c55e"), ("B", "#3b82f6")]:
            if ch in df.columns:
                fig_rgb.add_trace(go.Scatter(
                    x=df["time_sec"], y=df[ch],
                    name=ch, line=dict(color=color, width=1.5),
                    opacity=0.5 if f"{ch}_smooth" in df.columns else 1.0,
                ))
            if f"{ch}_smooth" in df.columns:
                fig_rgb.add_trace(go.Scatter(
                    x=df["time_sec"], y=df[f"{ch}_smooth"],
                    name=f"{ch} (smooth)", line=dict(color=color, width=2.5),
                ))
        fig_rgb.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Intensity (0-255)",
            template="plotly_white",
            height=420,
            hovermode="x unified",
        )
        # Highlight plateaus
        if is_plus and ss.plateaus:
            for i, p in enumerate(ss.plateaus, start=1):
                fig_rgb.add_vrect(
                    x0=p["start_time"], x1=p["end_time"],
                    fillcolor="rgba(230, 200, 112, 0.15)",
                    line_width=0,
                    annotation_text=f"P{i}",
                    annotation_position="top left",
                )
        st.plotly_chart(fig_rgb, use_container_width=True)

        # ------------- Plot HSV ----------
        if include_hsv and "H" in df.columns:
            st.subheader("📈 HSV over Time")
            fig_hsv = go.Figure()
            for ch, color, ax in [
                ("H", "#a855f7", "y1"),
                ("S", "#f59e0b", "y2"),
                ("V", "#06b6d4", "y2"),
            ]:
                if ch in df.columns:
                    fig_hsv.add_trace(go.Scatter(
                        x=df["time_sec"], y=df[ch],
                        name=ch, line=dict(color=color, width=2),
                        yaxis=ax,
                    ))
            fig_hsv.update_layout(
                xaxis_title="Time (s)",
                yaxis=dict(title="Hue (0-360°)", side="left"),
                yaxis2=dict(title="Saturation / Value (0-100%)", overlaying="y", side="right"),
                template="plotly_white",
                height=380,
                hovermode="x unified",
            )
            st.plotly_chart(fig_hsv, use_container_width=True)

        # ------------- Advanced: Absorbance plot ----------
        if is_advanced and "A_R" in df.columns:
            st.subheader("📈 Absorbance (Beer-Lambert)")
            fig_a = go.Figure()
            for ch, color in [("A_R", "#ef4444"), ("A_G", "#22c55e"), ("A_B", "#3b82f6")]:
                if ch in df.columns:
                    fig_a.add_trace(go.Scatter(
                        x=df["time_sec"], y=df[ch],
                        name=ch, line=dict(color=color, width=2),
                    ))
            fig_a.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Absorbance (A = -log₁₀(I/I₀))",
                template="plotly_white",
                height=380,
                hovermode="x unified",
            )
            st.plotly_chart(fig_a, use_container_width=True)

            best = df.attrs.get("best_channel", "A_G")
            st.info(f"💡 **Best channel for analysis:** `{best}` "
                    f"(highest absorbance change) — แนะนำใช้ channel นี้ใน calibration")

        # ------------- Plus: Plateau table ----------
        if is_plus and ss.plateaus:
            st.subheader(f"📋 Detected Plateaus ({len(ss.plateaus)})")
            tbl = summarize_plateaus(ss.plateaus)
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        elif is_plus:
            st.warning("ไม่พบ plateau — ลองปรับ threshold หรือ duration ใน sidebar")

        # ------------- Advanced: Calibration ----------
        if is_advanced and ss.plateaus and len(ss.plateaus) >= 2:
            st.divider()
            st.subheader("🎯 Calibration Curve (Advanced)")
            st.caption(
                "ป้อนค่าความเข้มข้นที่ทราบของแต่ละ plateau ตามลำดับเวลา → "
                "โปรแกรมจะคำนวณ linear regression"
            )

            n_plat = len(ss.plateaus)
            cols = st.columns(min(n_plat, 5))
            known_concs = []
            for i, p in enumerate(ss.plateaus):
                col = cols[i % len(cols)]
                val = col.number_input(
                    f"Plateau {i+1} (t={p['start_time']:.1f}s)",
                    min_value=0.0, value=float(i + 1) * 0.1, step=0.01,
                    format="%.4f", key=f"conc_{i}",
                )
                known_concs.append(val)

            cal_channel = st.selectbox(
                "Channel for calibration",
                [c for c in ["A_R", "A_G", "A_B"] if c in df.columns],
                index=1 if "A_G" in df.columns else 0,
            )

            if st.button("🔬 Build Calibration Curve"):
                concs, abss = plateau_to_calibration_points(
                    ss.plateaus, known_concs, df, abs_column=cal_channel,
                )
                cal = linear_regression(concs, abss, channel=cal_channel)
                ss.calibration = cal

            if ss.calibration is not None:
                cal = ss.calibration
                col1, col2, col3 = st.columns(3)
                col1.metric("Slope (ε·l)", f"{cal.slope:.4f}")
                col2.metric("Intercept", f"{cal.intercept:.4f}")
                quality, emoji = get_calibration_quality(cal.r_squared)
                col3.metric(f"R² {emoji}", f"{cal.r_squared:.4f}",
                            help=quality)

                st.code(cal.equation_str(), language="text")

                # Plot calibration
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=cal.concentrations, y=cal.absorbances,
                    mode="markers", name="Data",
                    marker=dict(color="#e6c870", size=12, line=dict(color="#1a2745", width=2)),
                ))
                # Fit line
                x_fit = np.linspace(min(cal.concentrations), max(cal.concentrations), 100)
                y_fit = cal.slope * x_fit + cal.intercept
                fig_cal.add_trace(go.Scatter(
                    x=x_fit, y=y_fit,
                    mode="lines", name=f"Fit (R²={cal.r_squared:.4f})",
                    line=dict(color="#1a2745", width=2, dash="dash"),
                ))
                fig_cal.update_layout(
                    xaxis_title="Concentration",
                    yaxis_title=f"Absorbance ({cal.channel})",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig_cal, use_container_width=True)

                # Predict unknown
                st.subheader("🔍 Predict Unknown Concentration")
                unk_a = st.number_input(
                    "Enter measured Absorbance:",
                    value=0.5, step=0.01, format="%.4f",
                )
                pred_c = cal.predict(unk_a)
                st.success(f"**Predicted concentration: {pred_c:.4f}**")

        # ------------- Export ----------
        st.divider()
        st.subheader("💾 Export Results")
        col_csv, col_meta = st.columns(2)
        with col_csv:
            st.download_button(
                "📥 Download CSV (data)",
                data=df_to_csv_bytes(df),
                file_name=make_filename("colorimetry_data", "csv"),
                mime="text/csv",
                use_container_width=True,
            )
        with col_meta:
            meta_json = make_metadata_json(
                video_meta=ss.video_meta,
                roi_info=ss.roi.to_dict() if ss.roi else {},
                sampling_fps=sampling_fps,
                settings={
                    "mode": mode,
                    "include_hsv": include_hsv,
                    "smooth_window": smooth_window if is_plus else None,
                },
            )
            st.download_button(
                "📥 Download metadata (JSON)",
                data=meta_json,
                file_name=make_filename("metadata", "json"),
                mime="application/json",
                use_container_width=True,
            )

        # Summary
        with st.expander("📄 Analysis Summary"):
            st.text(build_summary_text(ss.video_meta, df, ss.plateaus))

# ----------------------------------------------------------------
# Footer
# ----------------------------------------------------------------
st.divider()
st.caption(
    "🧪 **Flow Cell Colorimetry** — Smartphone-based digital image colorimetry. "
    "Developed by School of Science, University of Phayao."
)
