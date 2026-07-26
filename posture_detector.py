"""
posture_detector.py
--------------------
Calibrated posture check, using MediaPipe Pose's real depth (Z) data --
not just 2D pixel positions -- so forward leaning is detected directly
instead of being inferred (and sometimes masked) from 2D proxies.

MediaPipe Pose gives each landmark an x, y, AND z. z is depth relative
to your hips: more negative = closer to the camera than your hips.
That's exactly the signal for "are you leaning toward the screen."

  shoulder_depth   = how far forward your shoulders are vs your hips
                      (more negative than your calibrated baseline =
                      leaning/bending forward -- "slouching")
  neck_torso_ratio = neck length (nose-to-shoulder) divided by torso
                      length (shoulder-to-hip), in pixels. This ratio
                      is automatically scale-invariant (both shrink/grow
                      together as you move closer/farther from camera),
                      so it isolates real head-drop -- "bent neck"
  shoulder_tilt    = angle of the shoulder line from horizontal
                      ("uneven shoulders")
  lateral_offset   = sideways offset of your torso from your hips,
                      normalized by shoulder width ("leaning too much"
                      to one side)
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
    shoulder_depth: float
    neck_torso_ratio: float
    shoulder_tilt_deg: float
    lateral_offset: float


class PostureDetector:
    def __init__(self):
        self.baseline_depth: Optional[float] = None
        self.baseline_ratio: Optional[float] = None
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

        # Real depth: MediaPipe's z is relative to the hip midpoint --
        # more negative shoulder z means your shoulders are closer to
        # the camera than your hips, i.e. you're leaning forward.
        shoulder_depth = (lm[P.LEFT_SHOULDER].z + lm[P.RIGHT_SHOULDER].z) / 2

        torso_len_px = hip_mid[1] - shoulder_mid[1]
        neck_len_px = shoulder_mid[1] - nose[1]
        neck_torso_ratio = neck_len_px / (torso_len_px if abs(torso_len_px) > 1e-3 else 1e-3)

        dx, dy = r_sh[0] - l_sh[0], r_sh[1] - l_sh[1]
        shoulder_tilt = math.degrees(math.atan2(dy, dx))
        shoulder_width = math.hypot(dx, dy) or 1.0
        lateral_offset = (shoulder_mid[0] - hip_mid[0]) / shoulder_width

        return PostureReading(
            nose_px=(int(nose[0]), int(nose[1])),
            shoulder_mid_px=(int(shoulder_mid[0]), int(shoulder_mid[1])),
            hip_mid_px=(int(hip_mid[0]), int(hip_mid[1])),
            left_shoulder_px=(int(l_sh[0]), int(l_sh[1])),
            right_shoulder_px=(int(r_sh[0]), int(r_sh[1])),
            shoulder_depth=shoulder_depth,
            neck_torso_ratio=neck_torso_ratio,
            shoulder_tilt_deg=shoulder_tilt,
            lateral_offset=lateral_offset,
        )

    # ---------------------- Calibration ----------------------
    def add_calibration_sample(self, reading: PostureReading):
        self._calib_samples.append((reading.shoulder_depth, reading.neck_torso_ratio, reading.lateral_offset))

    def finalize_calibration(self) -> bool:
        if not self._calib_samples:
            return False
        n = len(self._calib_samples)
        self.baseline_depth = sum(d for d, _, _ in self._calib_samples) / n
        self.baseline_ratio = sum(r for _, r, _ in self._calib_samples) / n
        self.baseline_lateral = sum(l for _, _, l in self._calib_samples) / n
        self._calib_samples = []
        return True

    def reset_calibration(self):
        self.baseline_depth = None
        self.baseline_ratio = None
        self.baseline_lateral = None
        self._calib_samples = []

    def is_calibrated(self) -> bool:
        return self.baseline_depth is not None and self.baseline_ratio is not None

    def calibration_progress(self) -> int:
        return len(self._calib_samples)

    # ---------------------- Classification ----------------------
    def classify(self, reading: PostureReading) -> List[Tuple[str, str]]:
        """Returns a list of (issue_name, guidance_message) currently active."""
        if not self.is_calibrated():
            return []

        issues = []
        depth_delta = reading.shoulder_depth - self.baseline_depth   # more negative = leaning forward
        ratio_change = reading.neck_torso_ratio / self.baseline_ratio if self.baseline_ratio else 1.0
        tilt = abs(reading.shoulder_tilt_deg)
        lateral_dev = abs(reading.lateral_offset - self.baseline_lateral)

        if depth_delta < -config.FORWARD_LEAN_THRESHOLD:
            issues.append(("slouching", "You're bending forward - sit up straight"))
        if ratio_change < config.NECK_TORSO_RATIO_THRESHOLD:
            issues.append(("bent_neck", "Your head is drooping - lift your head up"))
        if tilt > config.SHOULDER_TILT_THRESHOLD:
            issues.append(("uneven_shoulders", "Keep both shoulders level"))
        if lateral_dev > config.LATERAL_LEAN_THRESHOLD:
            issues.append(("leaning", "You're leaning to one side - straighten up"))

        return issues