# BlindSpotDetector
Blind Spot Detector – Dopamine Junkies
An AI-based Blind Spot Detection System using simulated sensors, weather environments, and real-time deep learning inference.

Project Overview:
This project is built for a hackathon and simulates blind spot detection using CARLA-generated data, radar sensors, and crash prediction logic. It integrates multiple sensor modalities (including radar, RGB, and possibly DVS/segmentation in future updates) and uses YOLO-based object detection to identify potential hazards in blind zones.

Repository Structure:
.idea/ – IDE settings (can be ignored)
blindspot_simulation/ – Core blind spot simulation logic using CARLA
fogDetector_ui/ – Streamlit-based UI for object detection and visualization
.gitattributes – Git LFS support for large dataset files
.gitignore – Ignores datasets, logs, and output files
Decision converter – Script or module to convert detection results into actionable decisions
Pipelined_process_Q1 – Represents the decision or data flow of the detection system
README.md – This file
Trained_Model_2.pt – Trained PyTorch model (YOLOv5 format)
best.pt – Possibly the best checkpoint model (based on validation performance)
blindspot_simulation.zip – Dataset or simulation environment archive
train.py – Training script for YOLO or custom model
Setup Instructions

Clone the repository:
git clone https://github.com/ArnavMadavaram/BlindSpotDetector.git
cd BlindSpotDetector

Install dependencies:
Make sure you have Python 3.10 or higher installed. Then run:
pip install -r requirements.txt

If requirements.txt is not available, create it by running:
pip freeze > requirements.txt

Run the simulation:
Navigate to the blindspot_simulation folder and run the appropriate CARLA script. Replace with the actual filename once confirmed.

Launch the UI
Navigate to the fogDetector_ui folder and run:
streamlit run app.py

Model Details:
Model: YOLOv5 (trained on a custom dataset with up to 16 classes)
Input: Simulated foggy or rainy images
Output: Bounding boxes and decision output via the Decision converter module

Future Enhancements:
Add DVS and Semantic Segmentation support
Improve crash prediction using velocity and trajectory estimation
Add ROS or real-vehicle integration
Implement blind spot warning logic for vehicle control systems

Team Dopamine Junkies:
Swagath Srinivasan 
Rishigesh Rajendrakumar
Srikar Lanka
Arnav Madavaram

Expand to real-world datasets
Implement LIDAR sensor emulation
Add vehicle control & warning logic
Host full app online with Docker support
License:

MIT License. See LICENSE file for more information.

Feedback:

Feel free to open an Issue or submit a Pull Request if you'd like to contribute.
