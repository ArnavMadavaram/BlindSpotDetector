import streamlit as st
import base64

# --- Set Background ---
def set_background(jpg_file):
    with open(jpg_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    .title-text {{
        font-size: 50px;
        font-weight: bold;
        color: #000;
        text-align: center;
        padding-top: 20px;
    }}
    .section-title {{
        font-size: 30px;
        font-weight: bold;
        color: #000;
        margin-top: 40px;
    }}
    .section-content {{
        font-size: 17px;
        color: #111;
        background-color: rgba(255,255,255,0.85);
        padding: 20px;
        border-radius: 10px;
        text-align: justify;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set the same black-and-white background
set_background("image.jpg")

# ------------------------------
st.markdown('<div class="title-text">🧠 ML-Based Object Detection (Software)</div>', unsafe_allow_html=True)
st.markdown('[👉 View GitHub Repository](https://github.com/ArnavMadavaram/BlindSpotDetector/tree/main)', unsafe_allow_html=True)

# ------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("1. Carla Simulation"):
        st.markdown('<div class="section-title">CARLA Simulation for Foggy Weather</div>', unsafe_allow_html=True)
        st.image("CARLA.png", use_container_width=True)
        st.markdown("""
        <div class="section-content">
        In the software workflow, we started with CARLA to simulate adverse weather conditions. Dense fog and rain parameters were configured using CARLA’s weather API. The simulation mimicked real-world low-visibility traffic scenarios.
        
        During this phase:
        
        - Vehicles were spawned in urban maps.
        - RGB and Semantic Segmentation cameras were mounted.
        - Sensor data was collected frame-by-frame in foggy and rainy conditions.
        
        These images provided essential raw data for our deep learning model.
        </div>
        """, unsafe_allow_html=True)

with col2:
    if st.button("2. Model Training"):
        st.markdown('<div class="section-title">YOLOv5 Object Detection Training</div>', unsafe_allow_html=True)
        st.image("Final Output1.png", use_container_width=True)
        st.markdown("""
        <div class="section-content">
        We employed the YOLOv5 framework to train an object detection model focused on foggy/rainy conditions.
        
        Highlights:
        - Model used: `YOLOv5s` (lightweight for faster training)
        - Optimized for multiple classes from semantic segmentation
        - Trained for several epochs using our processed dataset
        - Evaluated using precision, recall, and mAP metrics
        
        Despite synthetic data, YOLOv5 showed promising detection accuracy even under adverse conditions.
        </div>
        """, unsafe_allow_html=True)

with col1:
    if st.button("3. Data Preparation"):
        st.markdown('<div class="section-title">Preprocessing RGB & Semantic Segmentation</div>', unsafe_allow_html=True)
        st.image("Data Processing1.png", use_container_width=True)
        st.markdown("""
        <div class="section-content">
        We processed CARLA-generated RGB images and semantic segmentation maps to align with YOLOv5's training requirements.
        
        Key Steps:
        - Resized and converted RGB images into YOLO-compatible `.jpg` format.
        - Extracted object class labels from semantic `.npy` masks.
        - Converted label maps into `.txt` bounding box format for YOLO.
        - Split the dataset into training, validation, and test sets.
        
        This step ensured that our simulation data could be effectively used by YOLO for object detection learning.
        </div>
        """, unsafe_allow_html=True)

with col2:
    if st.button("4. Streamlit Web App"):
        st.markdown('<div class="section-title">User Interface with Streamlit</div>', unsafe_allow_html=True)
        st.image("part4.png", use_container_width=True)
        st.markdown("""
        <div class="section-content">
        The final step was to deploy the trained model in a user-friendly interface using Streamlit.
        
        Features of our UI:
        - Upload image or select from foggy dataset
        - Run inference in real-time
        - Display bounding boxes on detected objects
        
        Due to time constraints, we were unable to deploy the site publicly, but the local UI demonstrated full pipeline functionality end-to-end.
        </div>
        """, unsafe_allow_html=True)
