import threading
import time

import cv2


class CameraService:
    def __init__(self, inference_engine, camera_index=0):
        self.inference_engine = inference_engine
        self.camera_index = camera_index
        self.capture = None
        self.thread = None
        self.lock = threading.Lock()
        self.is_running = False
        self.latest_jpeg = None
        self.last_error = None

    def start(self):
        if self.is_running:
            return {"ok": True, "message": "Camera already running."}

        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            self.last_error = "Unable to open webcam. Check camera permissions and device availability."
            self.capture = None
            return {"ok": False, "message": self.last_error}

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        self.is_running = True
        self.last_error = None
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return {"ok": True, "message": "Camera started."}

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)

        if self.capture is not None:
            self.capture.release()

        self.capture = None
        self.thread = None
        with self.lock:
            self.latest_jpeg = None

    def _capture_loop(self):
        while self.is_running and self.capture is not None:
            ok, frame = self.capture.read()
            if not ok:
                self.last_error = "Camera frame could not be read."
                time.sleep(0.08)
                continue

            frame = cv2.flip(frame, 1)
            try:
                annotated_frame, _ = self.inference_engine.process_frame(
                    frame,
                    camera_active=self.is_running,
                )
            except Exception as error:
                self.last_error = str(error)
                annotated_frame = frame.copy()
                cv2.rectangle(annotated_frame, (18, 18), (720, 92), (8, 14, 28), -1)
                cv2.putText(
                    annotated_frame,
                    "Live inference error - check backend terminal",
                    (34, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (64, 82, 255),
                    2,
                )
            encoded_ok, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])

            if encoded_ok:
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()

            time.sleep(0.01)

    def frame_stream(self):
        while True:
            with self.lock:
                frame = self.latest_jpeg

            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            else:
                time.sleep(0.08)

            if not self.is_running and frame is None:
                time.sleep(0.2)
