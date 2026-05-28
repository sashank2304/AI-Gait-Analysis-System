# ============================================================
# GAIT DATASET VIDEO RECORDER
# ============================================================
#
# PURPOSE:
# Record NORMAL and IRREGULAR walking videos automatically
# for training your gait analytics model.
#
# FEATURES:
# ✅ Automatic recording timer
# ✅ Saves videos automatically
# ✅ Fullscreen webcam preview
# ✅ Countdown before recording
# ✅ Saves organized dataset
# ✅ Works in VS Code
# ✅ Friends can use it easily
#
# ============================================================

# ============================================================
# INSTALL DEPENDENCIES
# ============================================================

# python -m pip install opencv-python

# ============================================================
# IMPORTS
# ============================================================

import cv2
import os
import time
from datetime import datetime

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_PATH = "gait_video_dataset"

VIDEO_DURATION = 30   # seconds

FPS = 40

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ============================================================
# CREATE DATASET FOLDERS
# ============================================================

normal_path = os.path.join(
    OUTPUT_PATH,
    "normal_videos"
)

irregular_path = os.path.join(
    OUTPUT_PATH,
    "irregular_videos"
)

os.makedirs(normal_path, exist_ok=True)
os.makedirs(irregular_path, exist_ok=True)

# ============================================================
# INTRO
# ============================================================

print("\n===================================")
print("AI GAIT DATASET RECORDER")
print("===================================")

print("\nInstructions:")
print("1 -> Record NORMAL walking")
print("2 -> Record IRREGULAR walking")
print("ESC -> Exit")

print("\nRecording Tips:")
print("- Full body visible")
print("- Keep feet visible")
print("- Walk naturally")
print("- Change clothes/lighting occasionally")
print("- Use different walking speeds")

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():

    print("ERROR: Could not open webcam")
    exit()

# ============================================================
# RECORD FUNCTION
# ============================================================

def record_video(label_name, save_folder):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{label_name}_{timestamp}.mp4"

    save_path = os.path.join(
        save_folder,
        filename
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        save_path,
        fourcc,
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    # ========================================================
    # COUNTDOWN
    # ========================================================

    countdown_seconds = 5

    start_countdown = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        elapsed = int(
            time.time() - start_countdown
        )

        remaining = countdown_seconds - elapsed

        cv2.putText(
            frame,
            f"Recording starts in: {remaining}",
            (250, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 255),
            4
        )

        cv2.putText(
            frame,
            f"Prepare for {label_name.upper()} WALKING",
            (120, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow(
            "Gait Dataset Recorder",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if remaining <= 0:
            break

        if key == 27:
            return

    # ========================================================
    # START RECORDING
    # ========================================================

    print(f"\nRecording {label_name} walking...")

    start_time = time.time()

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        elapsed_time = time.time() - start_time

        remaining_time = int(
            VIDEO_DURATION - elapsed_time
        )

        # ====================================================
        # RECORD FRAME
        # ====================================================

        out.write(frame)

        frame_count += 1

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.putText(
            frame,
            f"RECORDING: {label_name.upper()}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"Time Left: {remaining_time}s",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {frame_count}",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "Gait Dataset Recorder",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ====================================================
        # STOP CONDITIONS
        # ====================================================

        if elapsed_time >= VIDEO_DURATION:
            break

        if key == 27:
            break

    out.release()

    print("\n===================================")
    print("VIDEO SAVED SUCCESSFULLY")
    print("===================================")

    print(f"Label: {label_name}")
    print(f"Frames Recorded: {frame_count}")
    print(f"Saved At:\n{save_path}")

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ========================================================
    # UI
    # ========================================================

    cv2.putText(
        frame,
        "AI GAIT DATASET RECORDER",
        (250, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        "1 -> NORMAL WALKING",
        (80, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "2 -> IRREGULAR WALKING",
        (80, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 165, 255),
        3
    )

    cv2.putText(
        frame,
        "ESC -> EXIT",
        (80, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "Keep Full Body Visible",
        (80, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Feet Must Be Visible",
        (80, 560),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Gait Dataset Recorder",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ========================================================
    # NORMAL WALKING
    # ========================================================

    if key == ord('1'):

        record_video(
            "normal",
            normal_path
        )

    # ========================================================
    # IRREGULAR WALKING
    # ========================================================

    elif key == ord('2'):

        record_video(
            "irregular",
            irregular_path
        )

    # ========================================================
    # EXIT
    # ========================================================

    elif key == 27:
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nProgram Closed Successfully")