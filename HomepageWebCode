import streamlit as st
import base64

# Set background
def set_background(jpg_file):
    with open(jpg_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
        }}
        .title {{
            font-size: 60px;
            font-weight: bold;
            color: #000;
            text-align: center;
            padding-top: 2rem;
        }}
        .subtitle {{
            font-size: 20px;
            text-align: center;
            color: #222;
            margin-bottom: 2rem;
        }}
        .section {{
            background-color: rgba(255,255,255,0.9);
            padding: 30px;
            border-radius: 12px;
            margin: 2rem auto;
            max-width: 800px;
            font-size: 18px;
            color: #111;
            line-height: 1.6;
            box-shadow: 0 0 12px rgba(0,0,0,0.2);
        }}
        .button-container {{
            display: flex;
            justify-content: center;
            gap: 3rem;
            margin-top: 30px;
        }}
        .custom-button {{
            padding: 1rem 2rem;
            font-size: 20px;
            font-weight: bold;
            color: white;
            background-color: #111;
            border-radius: 10px;
            border: none;
            cursor: pointer;
        }}
        </style>
        """, unsafe_allow_html=True
    )

# Apply background
set_background("image.jpg")

# --- Header ---
st.markdown('<div class="title">IEEE Hackathon 2025</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A unique interactive site-based project report</div>', unsafe_allow_html=True)

# --- Topics Section ---
st.markdown("""
<div class="section">
    <h4>Topics:</h4>
    <ul>
        <li><strong>Hardware:</strong> Sensor Fusion and Virtual Data Acquisition System</li>
        <li><strong>Software:</strong> Machine Learning-Based Object Detection in Foggy and Rainy Conditions</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --- Navigation Buttons ---
st.markdown("""
<div class="button-container">
    <a href="/Hardware" target="_self"><button class="custom-button">Hardware</button></a>
    <a href="/Software" target="_self"><button class="custom-button">Software</button></a>
</div>
""", unsafe_allow_html=True)
