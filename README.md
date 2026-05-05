# Flow Cell Colorimetry — Web App

วิเคราะห์การเปลี่ยนแปลงสีของสารละลายใน flow cell จากวิดีโอที่ถ่ายด้วยมือถือ

**University of Phayao | Faculty of Science **

---

## คุณสมบัติ

- 📤 อัปโหลดวิดีโอ (.mp4, .mov, .avi, .mkv, .webm)
- 📐 เลือก ROI ด้วยการลากเมาส์
- 🎨 วิเคราะห์ค่า RGB + HSV ทุกช่วงเวลา
- 📊 กราฟ interactive (Plotly) — zoom, pan, hover ได้
- 🔬 3 โหมดการใช้งาน:
  - **Core** — RGB/HSV vs time
  - **Core + Plus** — เพิ่ม smoothing + plateau detection
  - **Advanced** — เพิ่ม Beer-Lambert + Calibration curve
- 💾 Export CSV และ metadata JSON

---

## การติดตั้ง

### 1. สร้าง Python virtual environment (แนะนำ)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

### 3. รันแอป
```bash
streamlit run app.py
```

แอปจะเปิดในเบราว์เซอร์ที่ `http://localhost:8501`

---

## โครงสร้างไฟล์

```
flow_colorimetry/
├── app.py                       # Streamlit main app
├── requirements.txt             # Python dependencies
├── README.md                    # ไฟล์นี้
└── modules/
    ├── __init__.py
    ├── video_loader.py          # โหลด + extract frames
    ├── color_analyzer.py        # RGB + HSV extraction
    ├── signal_processor.py      # Smoothing + plateau detection
    ├── calibration.py           # Linear regression
    └── exporter.py              # CSV + JSON export
```

---

## วิธีใช้งานโดยย่อ

1. **Tab 1 — Upload & ROI**
   - อัปโหลดไฟล์วิดีโอ
   - ลากเมาส์เลือกพื้นที่ (ROI) ที่จะวิเคราะห์ บนเฟรมแรกของวิดีโอ
2. **Tab 2 — Analyze**
   - เลือก mode ใน sidebar (Core / Plus / Advanced)
   - กด **START ANALYSIS**
3. **Tab 3 — Results**
   - ดูกราฟ RGB, HSV (และ Absorbance ใน Advanced mode)
   - Export CSV/JSON

---

## ข้อแนะนำสำหรับการถ่ายวิดีโอ

- ✅ **ล็อค exposure + white balance** ในแอปกล้องมือถือ (ใช้ ProCam, Open Camera)
- ✅ ใช้ **light box / กล่องปิดมิด** ที่มีแสง LED คงที่
- ✅ วาง flow cell และมือถือใน**ตำแหน่งคงที่** ตลอดการทดลอง
- ✅ ความละเอียดวิดีโอ 720p–1080p เพียงพอ (ไม่ต้อง 4K — ไฟล์ใหญ่เกินจำเป็น)
- ❌ หลีกเลี่ยงแสงแวดล้อมที่เปลี่ยน (ไม่ใช้ในที่มีแสงแดดเปลี่ยน)

---

## Troubleshooting

**`streamlit-drawable-canvas` ติดตั้งไม่ได้**
```bash
pip install streamlit-drawable-canvas --no-cache-dir
```
ถ้ายังไม่ได้ แอปจะ fallback ใช้ manual input (ใส่ X, Y, W, H ด้วยตัวเอง)

**OpenCV error เมื่ออ่านวิดีโอ**
- ตรวจว่าไฟล์ไม่เสียหาย (ลอง play ในมือถือ/คอม)
- ถ้าเป็น `.mov` จากมือถือ Apple อาจต้อง convert เป็น `.mp4` ก่อนด้วย ffmpeg

**RAM ไม่พอ**
- ลด sampling rate (เช่น 0.5 fps แทน 1 fps)
- ใช้วิดีโอความละเอียดต่ำกว่า

---

## License

For academic and educational use at University of Phayao.
