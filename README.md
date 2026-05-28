# AI Gait Analysis Dashboard

Complete full-stack dashboard around the existing trained BiLSTM gait model.

This project does not retrain, modify, overwrite, or delete the existing model/training files. The Flask backend loads only:

- `best_pose_gait_model.keras`
- `best_pose_threshold.npy`

## Features

- Live webcam gait analysis with OpenCV
- MediaPipe Pose estimation with 33 landmarks
- 99-feature pose vectors: `x`, `y`, `z` for each landmark
- 30-frame sequence buffer
- TensorFlow/Keras reconstruction-error inference
- Moving-average anomaly score smoothing
- Corrected thresholds:
  - score `< 0.03`: `NORMAL`, `LOW FALL RISK`
  - score `< 0.06`: `SLIGHTLY IRREGULAR`, `MEDIUM FALL RISK`
  - otherwise: `HIGH FALL RISK`
- Real-time pose overlay, metric cards, charts, and risk gauge

## Folder Structure

```text
backend/
  app.py
  inference.py
  routes/
  utils/
  models/
    pose_landmarker_lite.task
frontend/
  package.json
  src/
requirements.txt
README.md
```

## Backend Setup

From this project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python backend\app.py
```

The API runs at:

```text
http://localhost:5000
```

## Frontend Setup

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The dashboard runs at:

```text
http://localhost:5173
```

## API Endpoints

- `GET /api/health`
- `POST /api/camera/start`
- `POST /api/camera/stop`
- `POST /api/reset`
- `GET /api/prediction`
- `GET /api/history`
- `GET /api/video_feed`

## Notes

- Keep your full body visible in the webcam frame for reliable landmark extraction.
- The backend owns the webcam connection, runs MediaPipe, performs model inference, and streams annotated frames.
- The frontend consumes REST metrics and the MJPEG webcam stream from Flask.
- On Python 3.13, MediaPipe uses the Tasks API. The dashboard includes `backend/models/pose_landmarker_lite.task` for pose landmark detection.
- The saved gait model file is loaded as-is. If Keras 3 cannot read the `.keras` extension directly, the backend falls back to legacy HDF5 weight loading without modifying the model file.
