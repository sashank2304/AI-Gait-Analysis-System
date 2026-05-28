# ============================================================
# LIVE TEST USING SAVED POSE LANDMARK MODEL
# ============================================================

import cv2
import mediapipe as mp
import numpy as np

from collections import deque

from tensorflow.keras.models import load_model

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "best_pose_gait_model.keras"

THRESHOLD_PATH = "best_pose_threshold.npy"

SEQUENCE_LENGTH = 30

# ============================================================
# LOAD MODEL
# ============================================================

print("\n===================================")
print("LOADING BEST MODEL")
print("===================================")

model = load_model(MODEL_PATH)

threshold = np.load(THRESHOLD_PATH)

print("MODEL LOADED SUCCESSFULLY!")

# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(

    static_image_mode=False,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)

# ============================================================
# EXTRACT LANDMARKS
# ============================================================

def extract_landmarks(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(rgb)

    if results.pose_landmarks:

        landmarks = []

        for lm in results.pose_landmarks.landmark:

            landmarks.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        return np.array(
            landmarks,
            dtype=np.float32
        )

    return None

# ============================================================
# LIVE ANALYTICS
# ============================================================

print("\n===================================")
print("LIVE AI GAIT ANALYTICS STARTED")
print("ESC = EXIT")
print("===================================")

cap = cv2.VideoCapture(0)

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

anomaly_history = deque(
    maxlen=15
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    display_frame = frame.copy()

    landmarks = extract_landmarks(frame)

    if landmarks is not None:

        sequence_buffer.append(
            landmarks
        )

    # ========================================================
    # WAIT FOR FULL SEQUENCE
    # ========================================================

    if len(sequence_buffer) < SEQUENCE_LENGTH:

        cv2.putText(

            display_frame,

            "COLLECTING WALK DATA...",

            (20,40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (0,255,255),

            2
        )

        cv2.imshow(
            "AI Gait Analytics",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

        continue

    # ========================================================
    # MODEL INPUT
    # ========================================================

    sample = np.expand_dims(

        np.array(sequence_buffer),

        axis=0
    )

    reconstructed = model.predict(
        sample,
        verbose=0
    )

    reconstruction_error = np.mean(

        np.square(
            sample[:, -1, :] -
            reconstructed
        )
    )

    anomaly_score = reconstruction_error

    anomaly_history.append(
        anomaly_score
    )

    smoothed_score = np.mean(
        anomaly_history
    )

    # ========================================================
    # STATUS
    # ========================================================

    # ========================================================
# STATUS
# ========================================================

    if smoothed_score < 0.02:
        status = "NORMAL"
        color = (0,255,0)
        fall_risk = "LOW"

    elif smoothed_score < 0.05:
        status = "SLIGHTLY IRREGULAR"
        color = (0,255,255)
        fall_risk = "MEDIUM"

    else:
        status = "HIGH FALL RISK"
        color = (0,0,255)
        fall_risk = "HIGH"

    # ========================================================
    # ANALYTICS
    # ========================================================

    movement_variability = np.std(
        anomaly_history
    )

    stability_score = max(
        0,
        100 - movement_variability
    )

    gait_confidence = max(
        0,
        100 - (smoothed_score / 3)
    )

    symmetry_score = max(
        0,
        100 - movement_variability * 2
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.putText(
        display_frame,
        f"STATUS: {status}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    cv2.putText(
        display_frame,
        f"Fall Risk: {fall_risk}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        display_frame,
        f"Anomaly Score: {smoothed_score:.2f}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        display_frame,
        f"Stability Score: {stability_score:.1f}%",
        (20,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.putText(
        display_frame,
        f"Gait Confidence: {gait_confidence:.1f}%",
        (20,200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    cv2.putText(
        display_frame,
        f"Symmetry Score: {symmetry_score:.1f}%",
        (20,240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,0,255),
        2
    )

    cv2.imshow(
        "AI Gait Analytics",
        display_frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()