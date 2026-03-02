import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import pathlib
import time

# Fix for models saved on Windows being loaded on macOS/Linux
pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="🚗 Blind Spot Detector", layout="centered", page_icon="🌫️")
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>🚗 Blind Spot Object Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a driving image to detect vehicles and pedestrians using YOLOv5.</p>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("⚙️ Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.05, 0.95, 0.25, 0.05,
    help="Lower = detect more objects. Higher = only show confident detections."
)

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.iou = 0.45
    return model

model = load_model()
model.conf = conf_threshold

uploaded_file = st.file_uploader("📁 Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    np_image = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Run Detection"):
        with st.spinner("Running detection..."):
            start = time.time()

            results = model(np_image)
            detections = results.xyxy[0]

            total_detected = len(detections)
            detected_classes = []
            img0 = np_image.copy()

            if total_detected > 0:
                names = model.names
                for *xyxy, conf, cls in detections:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    detected_classes.append(names[int(cls)])
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img0, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                end = time.time()
                st.success(f"✅ {total_detected} object(s) detected in {end - start:.2f} seconds.")
                st.info("📦 Classes: " + ", ".join(set(detected_classes)))
                st.image(img0, caption="Detection Result", use_container_width=True)
            else:
                st.warning(f"⚠️ No objects detected at {int(conf_threshold*100)}% confidence. Try lowering the threshold.")
