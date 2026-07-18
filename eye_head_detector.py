"""
eye_head_detector.py
---------------------
Approximates eye-contact / head-position from 2D FaceMesh landmark
geometry (shared via face_engine.FaceEngine), calibrated to you at the
start of the session -- same pattern as posture_detector.py.

NOTE ON ACCURACY: this is a lightweight heuristic based on head
orientation, not true 3D eyeball gaze-tracking. It's a solid proxy for
"are you generally facing the camera" (which is what matters for
interviews/presentations) but it won't catch pure eyeball movement
that happens without any head movement.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import config

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


@dataclass
class HeadReading:
    nose_px: Tuple[int, int]
    yaw_ratio: float     # ~0.5 = centered; <0.5 turned one way, >0.5 the other
    pitch_ratio: float   # nose position between eyes(low) and chin(high); higher = looking down
    roll_deg: float      # tilt of the eye line from horizontal


class EyeHeadDetector:
    def __init__(self):
        self.baseline_yaw: Optional[float] = None
        self.baseline_pitch: Optional[float] = None
        self._calib_samples: List[Tuple[float, float]] = []
        self._history = deque(maxlen=config.HEAD_STABILITY_WINDOW)

    def read(self, face_landmarks, frame_w: int, frame_h: int) -> Optional[HeadReading]:
        if face_landmarks is None:
            return None

        lm = face_landmarks

        def px(i):
            return (lm[i].x * frame_w, lm[i].y * frame_h)

        nose, chin = px(NOSE_TIP), px(CHIN)
        l_eye, r_eye = px(LEFT_EYE_OUTER), px(RIGHT_EYE_OUTER)

        eye_mid = ((l_eye[0] + r_eye[0]) / 2, (l_eye[1] + r_eye[1]) / 2)
        yaw_ratio = (nose[0] - l_eye[0]) / (r_eye[0] - l_eye[0] + 1e-6)
        face_height = math.hypot(chin[0] - eye_mid[0], chin[1] - eye_mid[1]) or 1.0
        pitch_ratio = (nose[1] - eye_mid[1]) / face_height
        roll_deg = math.degrees(math.atan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

        reading = HeadReading(
            nose_px=(int(nose[0]), int(nose[1])),
            yaw_ratio=yaw_ratio,
            pitch_ratio=pitch_ratio,
            roll_deg=roll_deg,
        )
        self._history.append((yaw_ratio, pitch_ratio))
        return reading

    # ---------------------- Calibration ----------------------
    def add_calibration_sample(self, reading: HeadReading):
        self._calib_samples.append((reading.yaw_ratio, reading.pitch_ratio))

    def finalize_calibration(self) -> bool:
        if not self._calib_samples:
            return False
        n = len(self._calib_samples)
        self.baseline_yaw = sum(y for y, _ in self._calib_samples) / n
        self.baseline_pitch = sum(p for _, p in self._calib_samples) / n
        self._calib_samples = []
        return True

    def reset_calibration(self):
        self.baseline_yaw = None
        self.baseline_pitch = None
        self._calib_samples = []

    def is_calibrated(self) -> bool:
        return self.baseline_yaw is not None and self.baseline_pitch is not None

    def calibration_progress(self) -> int:
        return len(self._calib_samples)

    # ---------------------- Classification ----------------------
    def classify(self, reading: HeadReading) -> Tuple[List[str], bool]:
        """Returns (active_issue_names, eye_contact_ok)."""
        if not self.is_calibrated():
            return [], True

        issues = []
        delta_yaw = reading.yaw_ratio - self.baseline_yaw
        delta_pitch = reading.pitch_ratio - self.baseline_pitch

        if abs(delta_yaw) > config.YAW_THRESHOLD:
            issues.append("looking_away")
        if delta_pitch > config.PITCH_DOWN_THRESHOLD:
            issues.append("looking_down")
        elif delta_pitch < -config.PITCH_UP_THRESHOLD:
            issues.append("looking_up")
        if abs(reading.roll_deg) > config.HEAD_ROLL_THRESHOLD:
            issues.append("head_tilted")

        eye_contact_ok = "looking_away" not in issues and "looking_down" not in issues
        return issues, eye_contact_ok

    def stability_score(self) -> int:
        """0-100, higher = head has been steadier recently."""
        if len(self._history) < 5:
            return 100
        yaws = [h[0] for h in self._history]
        pitches = [h[1] for h in self._history]
        spread = (max(yaws) - min(yaws)) + (max(pitches) - min(pitches))
        return max(0, 100 - int(spread * 400))

    def close(self):
        pass  # FaceEngine owns the model; nothing to release here