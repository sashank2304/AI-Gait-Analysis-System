import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


TASK_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]


class PoseEstimator:
    def __init__(self):
        self.uses_tasks_api = not hasattr(mp, "solutions")

        if self.uses_tasks_api:
            if not TASK_MODEL_PATH.exists():
                raise FileNotFoundError(
                    "MediaPipe Tasks requires backend/models/pose_landmarker_lite.task. "
                    "Download it before starting the backend."
                )

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(TASK_MODEL_PATH))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.mp_pose = None
            self.drawer = None
            self.drawing_styles = None
            self.pose = None
        else:
            self.mp_pose = mp.solutions.pose
            self.drawer = mp.solutions.drawing_utils
            self.drawing_styles = mp.solutions.drawing_styles
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def extract_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        if self.uses_tasks_api:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect(image)
            if not results.pose_landmarks:
                return None, None
            pose_landmarks = results.pose_landmarks[0]
        else:
            results = self.pose.process(rgb_frame)
            if not results.pose_landmarks:
                return None, None
            pose_landmarks = results.pose_landmarks

        source_landmarks = (
            pose_landmarks
            if self.uses_tasks_api
            else pose_landmarks.landmark
        )

        if len(source_landmarks) != 33:
            return None, None

        landmarks = []
        for landmark in source_landmarks:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        return np.asarray(landmarks, dtype=np.float32), pose_landmarks

    def draw_landmarks(self, frame, pose_landmarks):
        if not pose_landmarks:
            return frame

        if self.uses_tasks_api:
            self._draw_task_landmarks(frame, pose_landmarks)
        else:
            self.drawer.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style(),
            )
        return frame

    def _draw_task_landmarks(self, frame, pose_landmarks):
        height, width = frame.shape[:2]
        points = []

        for landmark in pose_landmarks:
            points.append((int(landmark.x * width), int(landmark.y * height)))

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(points) or end_idx >= len(points):
                continue

            cv2.line(frame, points[start_idx], points[end_idx], (45, 212, 191), 2, cv2.LINE_AA)

        for point in points:
            cv2.circle(frame, point, 3, (236, 253, 245), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 5, (14, 165, 233), 1, cv2.LINE_AA)
