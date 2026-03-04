import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import pathlib
import time
import pandas as pd

pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(
    page_title="ClearSight",
    layout="centered",
    page_icon="🔍",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700;800&display=swap');

[data-testid="stSidebarNav"] {display: none;}

.cs-header {
    padding: 16px 0 32px 0;
    border-bottom: 1px solid #30363d;
    margin-bottom: 32px;
}
.cs-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 76px;
    font-weight: 800;
    color: #00ffcc;
    letter-spacing: -3px;
    line-height: 1;
    margin: 0;
}
.cs-tagline {
    color: #8b949e;
    font-size: 16px;
    margin: 12px 0 0 0;
    letter-spacing: 0.1px;
}

.upload-hint {
    text-align: center;
    padding: 56px 24px;
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 14px;
    color: #8b949e;
    margin: 16px 0;
}
.upload-hint-icon { font-size: 44px; margin-bottom: 14px; }
.upload-hint-title { font-size: 17px; font-weight: 600; color: #e6edf3; margin-bottom: 6px; }
.upload-hint-sub { font-size: 13px; }

.metric-row { display: flex; gap: 12px; margin: 20px 0; }
.metric-card {
    flex: 1;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 12px;
    text-align: center;
}
.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #00ffcc;
    line-height: 1;
}
.metric-label {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 6px;
}

.section-heading {
    font-size: 12px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 24px 0 8px 0;
}

.stButton > button {
    background: #161b22 !important;
    border: 1px solid #00ffcc !important;
    color: #00ffcc !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    width: 100% !important;
    padding: 10px 0 !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #00ffcc !important;
    color: #0d1117 !important;
}

.cs-footer {
    text-align: center;
    color: #484f58;
    font-size: 12px;
    padding: 32px 0 8px 0;
    border-top: 1px solid #21262d;
    margin-top: 48px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cs-header">
    <p class="cs-logo">ClearSight</p>
    <p class="cs-tagline">AI-powered object detection for adverse driving conditions</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05, max_value=0.95, value=0.25, step=0.05,
    help="Lower = detect more objects. Higher = only confident detections.",
)

# ── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    m.iou = 0.45
    return m

model = load_model()
model.conf = conf_threshold


def class_color(name: str):
    """Deterministic BGR color per class name."""
    hue = (hash(name) * 37) % 180
    hsv = np.array([[[hue, 210, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_boxes(img_rgb: np.ndarray, detections, names) -> np.ndarray:
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        name = names[int(cls_id)]
        color = class_color(name)
        label = f"{name}  {conf * 100:.0f}%"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(out, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop a driving image to analyze",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if not uploaded:
    st.markdown("""
    <div class="upload-hint">
        <div class="upload-hint-icon">📷</div>
        <div class="upload-hint-title">Drop an image to get started</div>
        <div class="upload-hint-sub">Supports JPG &amp; PNG &nbsp;·&nbsp; Works best with driving scenes</div>
    </div>
    """, unsafe_allow_html=True)
else:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)

    col_img, col_ctrl = st.columns([3, 1])
    with col_img:
        st.markdown('<p class="section-heading">Uploaded Image</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with col_ctrl:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        run = st.button("Run Detection")

    if run:
        with st.spinner("Running inference..."):
            t0 = time.time()
            results = model(img_array)
            elapsed = time.time() - t0

        detections = results.xyxy[0].cpu().numpy()
        names = model.names

        # Metrics
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{len(detections)}</div>
                <div class="metric-label">Objects Detected</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{elapsed:.2f}s</div>
                <div class="metric-label">Inference Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{int(conf_threshold * 100)}%</div>
                <div class="metric-label">Min Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if len(detections) > 0:
            annotated = draw_boxes(img_array, detections, names)
            st.markdown('<p class="section-heading">Detection Results</p>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)

            rows = [
                {"Class": names[int(d[5])], "Confidence": f"{d[4] * 100:.1f}%", "_conf": d[4]}
                for d in detections
            ]
            df = (pd.DataFrame(rows)
                  .sort_values("_conf", ascending=False)
                  .drop(columns=["_conf"])
                  .reset_index(drop=True))
            df.index += 1

            st.markdown('<p class="section-heading">Detections</p>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning(
                f"No objects detected at {int(conf_threshold * 100)}% confidence. "
                "Try lowering the threshold in the sidebar."
            )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cs-footer">
    ClearSight &nbsp;·&nbsp; YOLOv5s (COCO pretrained) &nbsp;·&nbsp; Built by Arnav Madavaram
</div>
""", unsafe_allow_html=True)
