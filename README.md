# 🧠 AI-Based Gait Analysis and Fall Risk Prediction Using Deep Learning and MediaPipe Pose Estimation

## 📌 Overview

This project presents a real-time AI-powered gait analysis and fall-risk prediction system using Deep Learning and MediaPipe Pose Estimation.

The system analyzes human walking patterns using only a standard RGB webcam and predicts:

* Normal gait
* Slightly irregular gait
* High fall-risk gait patterns

The project combines:

* Computer Vision
* Pose Estimation
* Temporal Deep Learning
* Real-Time Analytics

to create a scalable, low-cost, and non-invasive healthcare monitoring solution.

---

# 🚀 Features


- ✅ Real-time gait analysis using webcam
- ✅ MediaPipe-based pose landmark extraction
- ✅ Deep Learning temporal sequence learning
- ✅ Fall-risk prediction system
- ✅ Live AI analytics dashboard
- ✅ Pose landmark visualization
- ✅ Real-time anomaly scoring
- ✅ Stability and symmetry analysis
- ✅ Lightweight and deployable system

---

# 🧠 Technologies Used

## Computer Vision

* OpenCV
* MediaPipe Pose Estimation

## Deep Learning

* TensorFlow
* Keras

## Models Implemented

* LSTM
* GRU
* CNN1D
* BiLSTM
* CNN-LSTM

## Frontend Dashboard

* React + Vite
* TailwindCSS
* Framer Motion
* Recharts

## Backend

* Flask API

---

# 📷 System Workflow

```text
Webcam Video
      ↓
MediaPipe Pose Estimation
      ↓
33 Body Landmark Extraction
      ↓
Sequence Generation (30 Frames)
      ↓
BiLSTM Deep Learning Model
      ↓
Reconstruction Error Analysis
      ↓
Gait Classification
      ↓
Fall Risk Prediction
      ↓
Live Analytics Dashboard
```

---

# 🧠 Pose Estimation

The system uses MediaPipe Pose Estimation to extract:

* 33 body landmarks
* X, Y, Z coordinates
* Total 99 features per frame

These landmarks represent:

* Joint positions
* Body posture
* Walking symmetry
* Leg movement patterns
* Temporal gait motion

---

# 🧠 Deep Learning Architecture

The project compares 5 Deep Learning architectures:

| Model    | Purpose                              |
| -------- | ------------------------------------ |
| LSTM     | Temporal gait learning               |
| GRU      | Lightweight sequence learning        |
| CNN1D    | Spatial-temporal feature extraction  |
| BiLSTM   | Bidirectional temporal gait learning |
| CNN-LSTM | Hybrid spatiotemporal learning       |

---

# 🏆 Best Performing Model

## Bidirectional LSTM (BiLSTM)

The BiLSTM model achieved the best performance because gait motion is:

* temporal
* cyclic
* sequential
* bidirectional

BiLSTM learns:

* past motion dependencies
* future motion dependencies
* gait rhythm
* walking symmetry

simultaneously.

---

# 📊 Final Evaluation Metrics

| Model     | Accuracy   | Precision  | Recall     | F1 Score   | AUC        |
| --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| LSTM      | 88.00%     | 96.42%     | 88.89%     | 92.50%     | 93.72%     |
| GRU       | 87.72%     | 96.58%     | 88.38%     | 92.30%     | 93.43%     |
| CNN1D     | 88.78%     | 96.55%     | 89.73%     | 93.02%     | 93.93%     |
| 🏆 BiLSTM | **92.35%** | **96.79%** | **93.93%** | **95.34%** | **96.07%** |
| CNN-LSTM  | 87.56%     | 96.28%     | 88.48%     | 92.21%     | 93.16%     |

---

# 📈 Real-Time Analytics

The live dashboard displays:

* Gait status
* Fall-risk estimation
* Anomaly score
* Stability score
* Symmetry score
* Gait confidence
* Pose landmark visualization

---

# 📂 Dataset

Custom gait datasets were created using:

* Normal walking videos
* Irregular gait videos
* Webcam-based recordings

The system was trained using:

* Sliding temporal sequences
* Pose landmark sequences
* Real-world gait motion patterns

---

# 🧪 Training Configuration

| Parameter       | Value                    |
| --------------- | ------------------------ |
| Sequence Length | 30 Frames                |
| Epochs          | 35                       |
| Batch Size      | 32                       |
| Optimizer       | Adam                     |
| Loss Function   | Mean Squared Error (MSE) |

---

# 🖥️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Gait-Analysis-System.git
cd AI-Gait-Analysis-System
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Project

## Run Backend

```bash
python app.py
```

## Run Live Inference

```bash
python test.py
```

---

# 🧠 Applications

* Elderly fall-risk monitoring
* Rehabilitation systems
* Physiotherapy monitoring
* Smart healthcare systems
* AI surveillance systems
* Neurological gait analysis
* Hospital patient monitoring
* Sports biomechanics

---

# ✅ Advantages

* Low-cost deployment
* Non-invasive system
* Real-time processing
* Webcam-based analysis
* Portable and scalable
* AI-driven analytics

---

# ⚠️ Limitations

* Sensitive to lighting variations
* Camera-angle dependency
* Requires full-body visibility
* Limited dataset diversity
* Environment-dependent pose estimation

---

# 🔮 Future Scope

* Cloud deployment
* Mobile app integration
* Multi-person gait analysis
* Transformer-based architectures
* Smart hospital integration
* IoT healthcare systems

---

# 👨‍💻 Authors

* Sashank Abburu
* Samvrudha RR
* Dheepak K

---

# 📚 References

1. K. Nishizawa et al., "Evaluation of the Clinical Utility of a Gait Analysis System Using Pose Estimation Techniques in Physical Therapy," IEEE, 2024.

2. S. Balamurugan et al., "A Deep Learning Model for Video-Based Human Fall Detection Using CRNN and EfficientNet," IEEE, 2025.

3. S. G and N. Meenakshisundaram, "A Robust Development of Human Posture Recognition Model by using Artificial Intelligence based Learning Scheme," IEEE, 2025.

4. C. N. Kumar and D. Bujji Babu, "An Extensive Analysis of Deep Learning-Based Human Activity Detection Techniques," IEEE, 2024.
