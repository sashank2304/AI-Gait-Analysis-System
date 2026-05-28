from collections import deque
from pathlib import Path
from time import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils.pose import PoseEstimator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "best_pose_gait_model.keras"
THRESHOLD_PATH = PROJECT_ROOT / "best_pose_threshold.npy"

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 99
SMOOTHING_WINDOW = 15
HISTORY_LIMIT = 160


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


class GaitInferenceEngine:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.threshold_path = THRESHOLD_PATH
        self.sequence_length = SEQUENCE_LENGTH
        self.feature_count = FEATURE_COUNT

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing trained model: {MODEL_PATH}")
        if not THRESHOLD_PATH.exists():
            raise FileNotFoundError(f"Missing saved threshold: {THRESHOLD_PATH}")

        self.model = self._load_trained_model(MODEL_PATH)
        self.saved_threshold = float(np.asarray(np.load(THRESHOLD_PATH)).reshape(-1)[0])
        self.pose_estimator = PoseEstimator()

        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.anomaly_history = deque(maxlen=SMOOTHING_WINDOW)
        self.metric_history = deque(maxlen=HISTORY_LIMIT)
        self.latest_prediction = self._empty_prediction()

    def _load_trained_model(self, model_path):
        try:
            return load_model(str(model_path), compile=False)
        except ValueError as error:
            try:
                import h5py
                from keras.initializers import GlorotUniform, Orthogonal, Zeros
                from keras.layers import LSTM, Bidirectional, Dense, Dropout
                from keras.models import Sequential
                from keras.src.legacy.saving import legacy_h5_format

                custom_objects = {
                    "Sequential": Sequential,
                    "Bidirectional": Bidirectional,
                    "LSTM": LSTM,
                    "Dropout": Dropout,
                    "Dense": Dense,
                    "GlorotUniform": GlorotUniform,
                    "Orthogonal": Orthogonal,
                    "Zeros": Zeros,
                }

                with h5py.File(model_path, "r") as h5_file:
                    try:
                        return legacy_h5_format.load_model_from_hdf5(
                            h5_file,
                            custom_objects=custom_objects,
                            compile=False,
                        )
                    except Exception:
                        return self._load_bilstm_weights_from_hdf5(
                            h5_file,
                            legacy_h5_format,
                            Sequential,
                            Bidirectional,
                            LSTM,
                            Dropout,
                            Dense,
                        )
            except Exception as fallback_error:
                raise RuntimeError(
                    "Unable to load the saved gait model. The file exists, "
                    "but Keras could not read it as either a v3 .keras archive "
                    "or a legacy HDF5 model."
                ) from fallback_error

    def _load_bilstm_weights_from_hdf5(
        self,
        h5_file,
        legacy_h5_format,
        Sequential,
        Bidirectional,
        LSTM,
        Dropout,
        Dense,
    ):
        model = Sequential(
            [
                Bidirectional(
                    LSTM(128, return_sequences=True),
                    input_shape=(SEQUENCE_LENGTH, FEATURE_COUNT),
                ),
                Dropout(0.3),
                Bidirectional(LSTM(64)),
                Dropout(0.3),
                Dense(FEATURE_COUNT),
            ]
        )
        model(np.zeros((1, SEQUENCE_LENGTH, FEATURE_COUNT), dtype=np.float32))
        weight_group = h5_file["model_weights"] if "model_weights" in h5_file else h5_file
        legacy_h5_format.load_weights_from_hdf5_group(weight_group, model)
        return model

    def reset(self):
        self.sequence_buffer.clear()
        self.anomaly_history.clear()
        self.metric_history.clear()
        self.latest_prediction = self._empty_prediction()

    def _empty_prediction(self):
        return {
            "timestamp": time(),
            "cameraActive": False,
            "modelReady": True,
            "collecting": False,
            "warmingUp": False,
            "framesCollected": len(self.sequence_buffer),
            "sequenceLength": SEQUENCE_LENGTH,
            "featureCount": FEATURE_COUNT,
            "status": "WAITING FOR POSE",
            "fallRisk": "WAITING",
            "riskLevel": "waiting",
            "anomalyScore": 0.0,
            "rawAnomalyScore": 0.0,
            "gaitConfidence": 0.0,
            "stabilityScore": 0.0,
            "symmetryScore": 0.0,
            "savedThreshold": self.saved_threshold if hasattr(self, "saved_threshold") else 0.0,
            "calibrationRatio": 0.0,
            "landmarksDetected": False,
            "message": "Stand where your full body is visible, then start walking.",
        }

    def process_frame(self, frame, camera_active=True):
        annotated_frame = frame.copy()
        landmarks, pose_landmarks = self.pose_estimator.extract_landmarks(frame)
        annotated_frame = self.pose_estimator.draw_landmarks(annotated_frame, pose_landmarks)

        if landmarks is not None and landmarks.shape[0] == FEATURE_COUNT:
            self.sequence_buffer.append(landmarks)

        if not self.sequence_buffer:
            prediction = self._waiting_for_pose_prediction(landmarks is not None, camera_active)
            self.latest_prediction = prediction
            self._draw_overlay(annotated_frame, prediction)
            return annotated_frame, prediction

        sample = self._current_model_sample()
        reconstructed = self.model.predict(sample, verbose=0)

        raw_error = float(np.mean(np.square(sample[:, -1, :] - reconstructed)))
        self.anomaly_history.append(raw_error)
        smoothed_score = float(np.mean(self.anomaly_history))

        prediction = self._build_prediction(
            smoothed_score=smoothed_score,
            raw_error=raw_error,
            landmarks=landmarks,
            landmarks_detected=landmarks is not None,
            camera_active=camera_active,
            warming_up=len(self.sequence_buffer) < SEQUENCE_LENGTH,
        )

        self.latest_prediction = prediction
        self.metric_history.append(
            {
                "timestamp": prediction["timestamp"],
                "anomalyScore": prediction["anomalyScore"],
                "stabilityScore": prediction["stabilityScore"],
                "symmetryScore": prediction["symmetryScore"],
                "riskLevel": prediction["riskLevel"],
            }
        )
        self._draw_overlay(annotated_frame, prediction)
        return annotated_frame, prediction

    def _waiting_for_pose_prediction(self, landmarks_detected, camera_active):
        return {
            **self._empty_prediction(),
            "timestamp": time(),
            "cameraActive": camera_active,
            "collecting": False,
            "warmingUp": False,
            "framesCollected": len(self.sequence_buffer),
            "landmarksDetected": landmarks_detected,
            "status": "WAITING FOR POSE",
            "fallRisk": "WAITING",
            "riskLevel": "waiting",
            "message": "Move into frame so MediaPipe can detect all 33 body landmarks.",
        }

    def _current_model_sample(self):
        buffered_frames = list(self.sequence_buffer)

        if len(buffered_frames) < SEQUENCE_LENGTH:
            padding = [buffered_frames[0]] * (SEQUENCE_LENGTH - len(buffered_frames))
            buffered_frames = padding + buffered_frames

        return np.expand_dims(np.asarray(buffered_frames, dtype=np.float32), axis=0)

    def _build_prediction(
        self,
        smoothed_score,
        raw_error,
        landmarks,
        landmarks_detected,
        camera_active,
        warming_up=False,
    ):
        if smoothed_score < 0.001:
            status = "NORMAL"
            fall_risk = "LOW FALL RISK"
            risk_level = "low"
        elif smoothed_score < 0.002:
            status = "SLIGHTLY IRREGULAR"
            fall_risk = "MEDIUM FALL RISK"
            risk_level = "medium"
        else:
            status = "HIGH FALL RISK"
            fall_risk = "HIGH FALL RISK"
            risk_level = "high"

        variability = float(np.std(self.anomaly_history)) if len(self.anomaly_history) > 1 else 0.0
        symmetry_score = self._symmetry_score(landmarks)
        stability_score = clamp(100.0 - (smoothed_score / 0.06) * 72.0 - variability * 650.0)
        gait_confidence = clamp(100.0 - (smoothed_score / 0.09) * 100.0 - variability * 220.0)
        calibration_ratio = smoothed_score / self.saved_threshold if self.saved_threshold > 0 else 0.0

        return {
            "timestamp": time(),
            "cameraActive": camera_active,
            "modelReady": True,
            "collecting": False,
            "warmingUp": warming_up,
            "framesCollected": len(self.sequence_buffer),
            "sequenceLength": SEQUENCE_LENGTH,
            "featureCount": FEATURE_COUNT,
            "status": status,
            "fallRisk": fall_risk,
            "riskLevel": risk_level,
            "anomalyScore": round(smoothed_score, 5),
            "rawAnomalyScore": round(raw_error, 5),
            "gaitConfidence": round(gait_confidence, 1),
            "stabilityScore": round(stability_score, 1),
            "symmetryScore": round(symmetry_score, 1),
            "savedThreshold": round(self.saved_threshold, 5),
            "calibrationRatio": round(calibration_ratio, 3),
            "landmarksDetected": landmarks_detected,
            "message": self._insight_message(
                status,
                fall_risk,
                stability_score,
                symmetry_score,
                warming_up,
                len(self.sequence_buffer),
            ),
        }

    def _symmetry_score(self, landmarks):
        if landmarks is None or landmarks.shape[0] != FEATURE_COUNT:
            return 0.0

        points = landmarks.reshape(33, 3)
        body_pairs = [
            (11, 12),
            (13, 14),
            (15, 16),
            (23, 24),
            (25, 26),
            (27, 28),
            (29, 30),
            (31, 32),
        ]
        pair_offsets = []

        for left_idx, right_idx in body_pairs:
            left = points[left_idx]
            right = points[right_idx]
            pair_offsets.append(abs(left[1] - right[1]) + 0.45 * abs(left[2] - right[2]))

        mean_offset = float(np.mean(pair_offsets))
        return clamp(100.0 - mean_offset * 360.0)

    def _insight_message(
        self,
        status,
        fall_risk,
        stability_score,
        symmetry_score,
        warming_up=False,
        frames_collected=SEQUENCE_LENGTH,
    ):
        if warming_up:
            return (
                "Live prediction is running now. Confidence stabilizes as the "
                f"rolling window fills: {frames_collected}/{SEQUENCE_LENGTH} frames."
            )
        if status == "NORMAL":
            return "Movement pattern is stable with low fall-risk indicators."
        if status == "SLIGHTLY IRREGULAR":
            return "Gait variation is elevated. Keep full body visible and continue monitoring."
        if stability_score < 35 or symmetry_score < 45:
            return "High-risk gait pattern detected with reduced stability or body symmetry."
        return f"{fall_risk} detected. Review walking pattern and consider supervised assessment."

    def _draw_overlay(self, frame, prediction):
        risk_colors = {
            "low": (48, 255, 160),
            "medium": (0, 215, 255),
            "high": (64, 82, 255),
            "waiting": (255, 221, 92),
        }
        color = risk_colors.get(prediction["riskLevel"], (255, 255, 255))

        cv2.rectangle(frame, (18, 18), (470, 174), (8, 14, 28), -1)
        cv2.rectangle(frame, (18, 18), (470, 174), color, 2)
        cv2.putText(
            frame,
            f"STATUS: {prediction['status']}",
            (34, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"{prediction['fallRisk']} | Score {prediction['anomalyScore']:.5f}",
            (34, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (238, 244, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Stability {prediction['stabilityScore']:.1f}%  Symmetry {prediction['symmetryScore']:.1f}%",
            (34, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (205, 222, 255),
            2,
        )

        if prediction.get("warmingUp"):
            cv2.putText(
                frame,
                f"Live warm-up window: {prediction['framesCollected']}/{SEQUENCE_LENGTH}",
                (34, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 221, 92),
                1,
            )
