# ============================================================
# AI GAIT ANALYTICS USING MEDIAPIPE POSE LANDMARKS
# ============================================================

import os
import cv2
import mediapipe as mp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from collections import deque

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras.models import (
    Sequential,
    load_model
)

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    GRU,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Bidirectional
)

from tensorflow.keras.callbacks import ReduceLROnPlateau

# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "gait_video_dataset"

NORMAL_VIDEO_PATH = os.path.join(
    DATASET_PATH,
    "normal_videos"
)

IRREGULAR_VIDEO_PATH = os.path.join(
    DATASET_PATH,
    "irregular_videos"
)

BEST_MODEL_PATH = "best_pose_gait_model.keras"

THRESHOLD_PATH = "best_pose_threshold.npy"

IMG_SIZE = 64

SEQUENCE_LENGTH = 30

EPOCHS = 35

BATCH_SIZE = 32

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
# STORAGE
# ============================================================

X_normal = []

X_irregular = []

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
# EXTRACT SEQUENCES
# ============================================================

def extract_sequences(

    video_path,

    target_list
):

    cap = cv2.VideoCapture(video_path)

    sequence = []

    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        landmarks = extract_landmarks(frame)

        if landmarks is not None:

            sequence.append(landmarks)

            if len(sequence) == SEQUENCE_LENGTH:

                target_list.append(
                    np.array(sequence)
                )

                sequence.pop(0)

                count += 1

    cap.release()

    print(
        f"{os.path.basename(video_path)} "
        f"-> {count} sequences"
    )

# ============================================================
# LOAD NORMAL
# ============================================================

print("\n===================================")
print("LOADING NORMAL VIDEOS")
print("===================================")

for file in os.listdir(NORMAL_VIDEO_PATH):

    if file.lower().endswith((
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    )):

        extract_sequences(

            os.path.join(
                NORMAL_VIDEO_PATH,
                file
            ),

            X_normal
        )

# ============================================================
# LOAD IRREGULAR
# ============================================================

print("\n===================================")
print("LOADING IRREGULAR VIDEOS")
print("===================================")

for file in os.listdir(IRREGULAR_VIDEO_PATH):

    if file.lower().endswith((
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    )):

        extract_sequences(

            os.path.join(
                IRREGULAR_VIDEO_PATH,
                file
            ),

            X_irregular
        )

# ============================================================
# NUMPY
# ============================================================

X_normal = np.array(X_normal)

X_irregular = np.array(X_irregular)

# ============================================================
# SPLIT
# ============================================================

train_idx, normal_test_idx = train_test_split(

    np.arange(len(X_normal)),

    test_size=0.2,

    random_state=42,

    shuffle=True
)

X_train = X_normal[train_idx]

X_test_normal = X_normal[normal_test_idx]

X_test = np.concatenate([
    X_test_normal,
    X_irregular
])

y_test = np.concatenate([
    np.zeros(len(X_test_normal)),
    np.ones(len(X_irregular))
])

# ============================================================
# SUMMARY
# ============================================================

print("\n===================================")
print("DATASET SUMMARY")
print("===================================")

print("Train Samples:", len(X_train))
print("Test Samples :", len(X_test))
print("Input Shape  :", X_train.shape)

# ============================================================
# MODELS
# ============================================================

def lstm_model():

    model = Sequential([

        LSTM(
            128,
            return_sequences=True,
            input_shape=(SEQUENCE_LENGTH, 99)
        ),

        Dropout(0.3),

        LSTM(64),

        Dropout(0.3),

        Dense(99)
    ])

    return model

def gru_model():

    model = Sequential([

        GRU(
            128,
            return_sequences=True,
            input_shape=(SEQUENCE_LENGTH, 99)
        ),

        Dropout(0.3),

        GRU(64),

        Dropout(0.3),

        Dense(99)
    ])

    return model

def cnn1d_model():

    model = Sequential([

        Conv1D(
            128,
            3,
            activation='relu',
            input_shape=(SEQUENCE_LENGTH, 99)
        ),

        MaxPooling1D(2),

        Conv1D(
            64,
            3,
            activation='relu'
        ),

        Flatten(),

        Dense(
            128,
            activation='relu'
        ),

        Dropout(0.3),

        Dense(99)
    ])

    return model

def bidirectional_lstm():

    model = Sequential([

        Bidirectional(

            LSTM(
                128,
                return_sequences=True
            ),

            input_shape=(SEQUENCE_LENGTH, 99)
        ),

        Dropout(0.3),

        Bidirectional(
            LSTM(64)
        ),

        Dropout(0.3),

        Dense(99)
    ])

    return model

def hybrid_cnn_lstm():

    model = Sequential([

        Conv1D(
            128,
            3,
            activation='relu',
            input_shape=(SEQUENCE_LENGTH, 99)
        ),

        MaxPooling1D(2),

        LSTM(64),

        Dropout(0.3),

        Dense(99)
    ])

    return model

# ============================================================
# MODEL LIST
# ============================================================

models = [

    ("LSTM", lstm_model),

    ("GRU", gru_model),

    ("CNN1D", cnn1d_model),

    ("BiLSTM", bidirectional_lstm),

    ("CNN-LSTM", hybrid_cnn_lstm)
]

# ============================================================
# LOAD SAVED MODEL
# ============================================================

if os.path.exists(BEST_MODEL_PATH) and os.path.exists(THRESHOLD_PATH):

    print("\n===================================")
    print("LOADING SAVED MODEL")
    print("===================================")

    best_model = load_model(
        BEST_MODEL_PATH
    )

    best_threshold = np.load(
        THRESHOLD_PATH
    )

# ============================================================
# TRAIN MODELS
# ============================================================

else:

    best_f1 = -1

    for model_name, builder in models:

        print("\n===================================")
        print(f"TRAINING {model_name}")
        print("===================================")

        model = builder()

        model.compile(

            optimizer='adam',

            loss='mse'
        )

        reduce_lr = ReduceLROnPlateau(

            monitor='val_loss',

            factor=0.5,

            patience=3,

            verbose=1
        )

        model.fit(

            X_train,
            X_train[:, -1, :],

            validation_split=0.1,

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            callbacks=[reduce_lr],

            verbose=1
        )

        # ====================================================
        # THRESHOLD
        # ====================================================

        normal_reconstructed = model.predict(
            X_train,
            verbose=0
        )

        normal_errors = np.mean(

            np.square(
                X_train[:, -1, :] -
                normal_reconstructed
            ),

            axis=1
        )

        threshold = np.percentile(
            normal_errors,
            85
        )

        # ====================================================
        # TEST
        # ====================================================

        reconstructed = model.predict(
            X_test,
            verbose=0
        )

        reconstruction_errors = np.mean(

            np.square(
                X_test[:, -1, :] -
                reconstructed
            ),

            axis=1
        )

        y_pred = (

            reconstruction_errors > threshold

        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        try:

            auc = roc_auc_score(
                y_test,
                reconstruction_errors
            )

        except:

            auc = 0.0

        print("\n===================================")
        print(f"{model_name} RESULTS")
        print("===================================")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"AUC      : {auc:.4f}")

        if f1 > best_f1:

            best_f1 = f1

            best_model = model

            best_threshold = threshold

            model.save(
                BEST_MODEL_PATH
            )

            np.save(
                THRESHOLD_PATH,
                threshold
            )

            print("\nBEST MODEL UPDATED!")

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

    if len(sequence_buffer) < SEQUENCE_LENGTH:

        cv2.imshow(
            "AI Gait Analytics",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

        continue

    sample = np.expand_dims(
        np.array(sequence_buffer),
        axis=0
    )

    reconstructed = best_model.predict(
        sample,
        verbose=0
    )

    reconstruction_error = np.mean(

        np.square(
            sample[:, -1, :] -
            reconstructed
        )
    )

    anomaly_score = (
        reconstruction_error /
        best_threshold
    ) * 100

    anomaly_history.append(
        anomaly_score
    )

    smoothed_score = np.mean(
        anomaly_history
    )

    if smoothed_score < 180:

        status = "NORMAL"

        color = (0,255,0)

        fall_risk = "LOW"

    elif smoothed_score < 300:

        status = "SLIGHTLY IRREGULAR"

        color = (0,255,255)

        fall_risk = "MEDIUM"

    else:

        status = "HIGH FALL RISK"

        color = (0,0,255)

        fall_risk = "HIGH"

    stability_score = max(
        0,
        100 - smoothed_score
    )

    gait_confidence = max(
        0,
        100 - smoothed_score
    )

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

    cv2.imshow(
        "AI Gait Analytics",
        display_frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()

