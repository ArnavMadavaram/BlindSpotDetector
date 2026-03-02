# BlindSpotDetector
**Team Dopamine Junkies — IEEE Hackathon 2025**

An AI-based blind spot detection system using CARLA simulation, YOLOv5 object detection, and a Streamlit web interface.

---

## Project Overview

This system detects vehicles and pedestrians in driving scenarios affected by fog and rain. The pipeline:
1. **CARLA Simulator** generates synthetic training data in adverse weather (fog, rain)
2. **YOLOv5** is trained on the CARLA data and used for real-time object detection
3. **Streamlit** provides an interactive multi-page web interface for demos

---

## Project Structure

```
BlindSpotDetector/
│
├── Home.py                          # Main entry point — run this to launch the web app
│
├── pages/
│   ├── 1_Hardware_Overview.py       # Hardware: CARLA simulation and sensor setup
│   ├── 2_Software_Pipeline.py       # Software: YOLOv5 training and data pipeline
│   └── 3_Object_Detector.py         # Live detector — upload an image and run inference
│
├── fogDetector_ui/
│   ├── app.py                       # Standalone version of the detector
│   ├── Trained_Model_2.pt           # YOLOv5 model trained on CARLA foggy data
│   └── requirements.txt             # Dependencies for standalone app
│
├── blindspot_simulation/
│   └── car_sim.py                   # CARLA simulation script — generates training data
│
├── train.py                         # YOLOv5 training script
├── requirements.txt                 # Dependencies for the multi-page web app
└── README.md
```

---

## How to Run

### Multi-page Web App (Recommended)

```bash
pip install -r requirements.txt
streamlit run Home.py
```

Navigate between pages using the Streamlit sidebar:
- **Hardware Overview** — CARLA simulation setup and sensor configuration
- **Software Pipeline** — YOLOv5 training and data preparation steps
- **Object Detector** — upload a driving image and detect objects in real time

### Standalone Detector Only

```bash
cd fogDetector_ui
pip install -r requirements.txt
streamlit run app.py
```

---

## How It Works

### 1. Data Collection — CARLA Simulation
`blindspot_simulation/car_sim.py` connects to a running CARLA instance and:
- Spawns an ego vehicle + 25 NPC vehicles in an urban map
- Applies heavy adverse weather (85% fog density, 80% precipitation)
- Records 60 seconds of multi-sensor data:
  - RGB camera frames
  - Depth maps (PNG + NumPy arrays)
  - Semantic segmentation masks
  - Radar data (CSV)
- Outputs a crash sequence at the end for edge case data

### 2. Model Training
`train.py` fine-tunes a **YOLOv5s** model on the CARLA RGB frames.
Trained weights: `fogDetector_ui/Trained_Model_2.pt`

### 3. Inference
The web app uses **YOLOv5s pretrained on COCO** (80 real-world classes) for detection on any driving image. Confidence threshold is adjustable via the sidebar.

---

## Requirements

- Python 3.8+
- PyTorch
- Streamlit
- OpenCV
- Pillow

```bash
pip install -r requirements.txt
```

> **macOS note:** If you get an SSL certificate error on first run, fix it with:
> ```bash
> open /Applications/Python\ 3.12/Install\ Certificates.command
> ```

---

## Tech Stack

| Component | Technology |
|---|---|
| Simulation | CARLA 0.10.0 |
| Object Detection | YOLOv5s (Ultralytics) |
| Web Interface | Streamlit |
| Deep Learning | PyTorch |
| Image Processing | OpenCV, Pillow |

---

## Team

- Arnav Madavaram
- Swagath Srinivasan
- Rishigesh Rajendrakumar
- Srikar Lanka
