import streamlit as st
import base64
import os

# --- Background Setter ---
def set_background(jpg_file):
    if not os.path.exists(jpg_file):
        return
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

# Set background
set_background("image.jpg")

# Title
st.markdown('<div class="title-text">🔧 Hardware Implementation</div>', unsafe_allow_html=True)
st.markdown('[👉 View GitHub Repository](https://github.com/ArnavMadavaram/BlindSpotDetector/tree/main)', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)

with col1:
    if st.button("CARLA"):
        st.markdown('<div class="section-title">CARLA Simulation Environment</div>', unsafe_allow_html=True)
        if os.path.exists("CARLA.png"):
            st.image("CARLA.png", use_container_width=True)
        st.markdown("""
        <div class="section-content">
        The CARLA simulator was the backbone of our hardware simulation setup. We leveraged CARLA 0.10.0 to simulate a realistic driving environment in complex urban layouts. Key objectives in this module were:

        - Deploy an <strong>ego vehicle</strong> in a structured city environment.
        - Spawn <strong>NPC vehicles</strong> and pedestrians to reflect realistic traffic density.
        - Define <strong>sensor placements</strong> (left/right mirrors, rear bumper) to accurately mimic blind spot zones.
        - Generate synchronized sensor data for machine learning pipelines.
        </div>
        """, unsafe_allow_html=True)

with col2:
    if st.button("Data Processing"):
        st.markdown('<div class="section-title">Sensor Fusion & Preprocessing</div>', unsafe_allow_html=True)
        if os.path.exists("Data Processing.png"):
            st.image("Data Processing.png", use_container_width=True)
        st.markdown("""
        <div class="section-content">
        Our pipeline merged raw data from 4 sensor streams into one <strong>(256x256x6)</strong> tensor:

        - RGB: 3 normalized channels (0–1)
        - Depth: single channel from <code>.npy</code>
        - Semantic Segmentation: class label as int (raw)
        - Radar: converted <code>.csv</code> to pixel-mapped data

        Key preprocessing tasks:

        - <strong>Timestamps aligned</strong> via frame number.
        - <strong>Normalization</strong> done using known min-max values.
        - <strong>Semantic segmentation</strong> optionally one-hot encoded.

        Example tensor point:
        <code>Center → RGB: 0.75,0.75,0.78 | Depth: 0.00 | Radar: 0.00 | Class: 3</code>
        </div>
        """, unsafe_allow_html=True)

with col1:
    if st.button("Simulation"):
        st.markdown('<div class="section-title">Sensor Setup & Weather Simulation</div>', unsafe_allow_html=True)
        if os.path.exists("Simulation.png"):
            st.image("Simulation.png", use_container_width=True)
        st.markdown("""
        <div class="section-content">
        Our simulation goal was to replicate <strong>adverse driving conditions</strong> like fog and rain where typical camera-based detection fails. We activated heavy weather using the CARLA weather engine, adjusting parameters such as:

        - Fog density and distance
        - Precipitation intensity
        - Road wetness and puddles

        We also equipped the ego vehicle with:

        - 📷 <strong>RGB Camera</strong>: Captures the visual field for detection tasks.
        - 🌊 <strong>Depth Camera</strong>: Measures the distance to each pixel, useful for spatial reasoning.
        - 🧠 <strong>Semantic Segmentation Camera</strong>: Identifies and labels road objects.
        - 📡 <strong>Radar</strong>: Gives velocity, azimuth, altitude, and depth data — highly robust in fog.
        </div>
        """, unsafe_allow_html=True)

with col2:
    if st.button("Final Output"):
        st.markdown('<div class="section-title">Blind Spot Detection Inference</div>', unsafe_allow_html=True)
        if os.path.exists("Final Output.png"):
            st.image("Final Output.png", use_container_width=True)
        st.markdown("""
        <div class="section-content">
        RGB frames captured by the CARLA simulation are passed directly into our trained <strong>YOLOv5</strong> model for object detection.

        The model outputs bounding boxes with class labels and confidence scores, identifying vehicles and pedestrians even in dense fog and rain.

        <strong>Output example:</strong>
        <pre>
        car 0.87
        person 0.74
        car 0.61
        </pre>

        This allows real-time detection of hazards in poor visibility conditions, enabling safety warnings in real-world use cases.
        </div>
        """, unsafe_allow_html=True)
