"""
collect_data.py
----------------
Records hand-landmark samples from your webcam and appends them to a CSV
file, labeled with whatever class name you pass in. Use this to build a
training set for train_model.py.

Usage:
    python collect_data.py improper
    python collect_data.py normal
    python collect_data.py thumbs_up

Controls:
    s -> start/stop recording (while recording, every detected hand's
         landmarks are saved as one row, several times per second)
    q -> quit

Tips:
- Record ~150-300 samples per class for a decent first model.
- Vary hand position, distance from camera, and slight rotation while
  recording so the model generalizes instead of memorizing one pose.
- Run this once per label (e.g. "improper", "normal", "thumbs_up",
  "peace", "fist") to build a balanced dataset.
"""

import csv
import os
import sys

import cv2
import mediapipe as mp

import config


def main():
    if len(sys.argv) < 2:
        print("Usage: python collect_data.py <label_name>")
        print("Example: python collect_data.py improper")
        sys.exit(1)

    label = sys.argv[1]

    os.makedirs(config.DATA_DIR, exist_ok=True)
    file_exists = os.path.isfile(config.DATA_CSV)

    csv_file = open(config.DATA_CSV, "a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y")] + ["label"]
        writer.writerow(header)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    recording = False
    saved_count = 0

    print(f"Collecting samples for label: '{label}'")
    print("Press 's' to start/stop recording, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if recording:
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y])
                    row.append(label)
                    writer.writerow(row)
                    saved_count += 1

        status_text = f"RECORDING ({saved_count} saved)" if recording else "Paused"
        status_color = (0, 0, 255) if recording else (0, 200, 0)
        cv2.putText(frame, f"Label: {label} | {status_text}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, "s = start/stop | q = quit", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            recording = not recording
        elif key == ord("q"):
            break

    cap.release()
    hands.close()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved_count} samples for label '{label}' to {config.DATA_CSV}")


if __name__ == "__main__":
    main()