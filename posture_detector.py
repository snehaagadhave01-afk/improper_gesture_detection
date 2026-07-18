"""
posture_detector.py
--------------------
Front-camera-friendly posture check, calibrated to you.

Consumes Pose landmarks produced once per frame by pose_engine.PoseEngine
(shared with body_language_detector.py) -- no model of its own.

  torso_len       = shoulder-mid to hip-mid vertical distance
                     (shrinks when you lean/bend forward -- "slouching")
  neck_len        = nose to shoulder-mid vertical distance
                     (shrinks when your head drops -- "bent neck")
  shoulder_tilt   = angle of the shoulder line from horizontal
                     ("uneven shoulders")
  lateral_offset  = sideways offset of your torso from your hips
                     ("leaning too much" to one side)
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import config


@dataclass
class PostureReading:
    nose_px: Tuple[int, int]
    shoulder_mid_px: Tuple[int, int]
    hip_mid_px: Tuple[int, int]
    left_shoulder_px: Tuple[int, int]
    right_shoulder_px: Tuple[int, int]
    torso_len: float
    neck_len: float
    shoulder_tilt_deg: float
    lateral_offset: float


class PostureDetector:
    def __init__(self):
        self.baseline_torso: Optional[float] = None
        self.baseline_neck: Optional[float] = None
        self.baseline_lateral: Optional[float] = None
        self._calib_samples: List[Tuple[float, float, float]] = []

    def read(self, landmarks, PoseLandmark, frame_w: int, frame_h: int) -> Optional[PostureReading]:
        if landmarks is None:
            return None

        P = PoseLandmark
        lm = landmarks
        required = [P.NOSE, P.LEFT_SHOULDER, P.RIGHT_SHOULDER, P.LEFT_HIP, P.RIGHT_HIP]
        if min(lm[p].visibility for p in required) < 0.5:
            return None

        def px(p):
            return (lm[p].x * frame_w, lm[p].y * frame_h)

        nose = px(P.NOSE)
        l_sh, r_sh = px(P.LEFT_SHOULDER), px(P.RIGHT_SHOULDER)
        l_hip, r_hip = px(P.LEFT_HIP), px(P.RIGHT_HIP)

        shoulder_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
        hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

        dx, dy = r_sh[0] - l_sh[0], r_sh[1] - l_sh[1]
        shoulder_width = math.hypot(dx, dy) or 1.0

        # Normalized by shoulder width so moving closer/farther from the
        # camera doesn't mask real bending -- only the SHAPE of your
        # posture matters, not how big you appear in the frame.
        torso_len = (hip_mid[1] - shoulder_mid[1]) / shoulder_width
        neck_len = (shoulder_mid[1] - nose[1]) / shoulder_width

        shoulder_tilt = math.degrees(math.atan2(dy, dx))
        lateral_offset = (shoulder_mid[0] - hip_mid[0]) / shoulder_width

        return PostureReading(
            nose_px=(int(nose[0]), int(nose[1])),
            shoulder_mid_px=(int(shoulder_mid[0]), int(shoulder_mid[1])),
            hip_mid_px=(int(hip_mid[0]), int(hip_mid[1])),
            left_shoulder_px=(int(l_sh[0]), int(l_sh[1])),
            right_shoulder_px=(int(r_sh[0]), int(r_sh[1])),
            torso_len=torso_len,
            neck_len=neck_len,
            shoulder_tilt_deg=shoulder_tilt,
            lateral_offset=lateral_offset,
        )

    # ---------------------- Calibration ----------------------
    def add_calibration_sample(self, reading: PostureReading):
        self._calib_samples.append((reading.torso_len, reading.neck_len, reading.lateral_offset))

    def finalize_calibration(self) -> bool:
        if not self._calib_samples:
            return False
        n = len(self._calib_samples)
        self.baseline_torso = sum(t for t, _, _ in self._calib_samples) / n
        self.baseline_neck = sum(nk for _, nk, _ in self._calib_samples) / n
        self.baseline_lateral = sum(l for _, _, l in self._calib_samples) / n
        self._calib_samples = []
        return True

    def reset_calibration(self):
        self.baseline_torso = None
        self.baseline_neck = None
        self.baseline_lateral = None
        self._calib_samples = []

    def is_calibrated(self) -> bool:
        return self.baseline_torso is not None and self.baseline_neck is not None

    def calibration_progress(self) -> int:
        return len(self._calib_samples)

    # ---------------------- Classification ----------------------
    def classify(self, reading: PostureReading) -> List[Tuple[str, str]]:
        """Returns a list of (issue_name, guidance_message) currently active."""
        if not self.is_calibrated():
            return []

        issues = []
        torso_ratio = reading.torso_len / self.baseline_torso
        neck_ratio = reading.neck_len / self.baseline_neck
        tilt = abs(reading.shoulder_tilt_deg)
        lateral_dev = abs(reading.lateral_offset - self.baseline_lateral)

        if torso_ratio < config.TORSO_RATIO_THRESHOLD:
            issues.append(("slouching", "You're bending forward - sit up straight"))
        if neck_ratio < config.NECK_RATIO_THRESHOLD:
            issues.append(("bent_neck", "Your head is drooping - lift your head up"))
        if tilt > config.SHOULDER_TILT_THRESHOLD:
            issues.append(("uneven_shoulders", "Keep both shoulders level"))
        if lateral_dev > config.LATERAL_LEAN_THRESHOLD:
            issues.append(("leaning", "You're leaning to one side - straighten up"))

        return issues