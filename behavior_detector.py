"""
behavior_detector.py
---------------------
Face-derived checks that don't need their own MediaPipe model -- they
consume landmarks produced once per frame by face_engine.FaceEngine
(shared with eye_head_detector.py) to avoid running FaceMesh twice.

  extract_face_points()  -> pull out mouth/eye pixel coordinates
  hand_near_face()        -> is a fingertip touching mouth/eyes?
  MovementMonitor         -> flags repeated large jumps of a tracked
                              point (used for the general "stay still"
                              / unwanted-movement check)
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import config

MOUTH_CENTER = 13
LEFT_EYE = 159
RIGHT_EYE = 386

FINGERTIP_INDICES = [4, 8, 12, 16, 20]


@dataclass
class FaceResult:
    visible: bool
    mouth_px: Optional[Tuple[int, int]] = None
    left_eye_px: Optional[Tuple[int, int]] = None
    right_eye_px: Optional[Tuple[int, int]] = None


def extract_face_points(face_landmarks, frame_w: int, frame_h: int) -> FaceResult:
    if face_landmarks is None:
        return FaceResult(visible=False)

    lm = face_landmarks
    mouth = (int(lm[MOUTH_CENTER].x * frame_w), int(lm[MOUTH_CENTER].y * frame_h))
    left_eye = (int(lm[LEFT_EYE].x * frame_w), int(lm[LEFT_EYE].y * frame_h))
    right_eye = (int(lm[RIGHT_EYE].x * frame_w), int(lm[RIGHT_EYE].y * frame_h))
    return FaceResult(visible=True, mouth_px=mouth, left_eye_px=left_eye, right_eye_px=right_eye)


def hand_near_face(hand_points: List[Tuple[int, int]], face: FaceResult, threshold: float) -> bool:
    """True if any fingertip is within `threshold` pixels of mouth/eyes."""
    if not face.visible:
        return False

    face_points = [p for p in (face.mouth_px, face.left_eye_px, face.right_eye_px) if p]
    if not face_points:
        return False

    for idx in FINGERTIP_INDICES:
        if idx >= len(hand_points):
            continue
        fx, fy = hand_points[idx]
        for tx, ty in face_points:
            if math.hypot(fx - tx, fy - ty) < threshold:
                return True
    return False


class MovementMonitor:
    """
    Tracks a reference point (e.g. mouth center) across frames. If it
    jumps by more than MOVEMENT_THRESHOLD pixels on many frames within a
    rolling window, that's flagged as excessive/unwanted movement.
    Also tracks a "still streak" for inactivity detection (study mode).
    """

    def __init__(self):
        self._prev_point: Optional[Tuple[float, float]] = None
        self._history = deque(maxlen=config.MOVEMENT_WINDOW_FRAMES)
        self.still_streak = 0

    def update(self, point: Optional[Tuple[float, float]]) -> bool:
        if point is None:
            self._prev_point = None
            self._history.append(0)
            return False

        if self._prev_point is not None:
            dist = math.hypot(point[0] - self._prev_point[0], point[1] - self._prev_point[1])
            jumped = dist > config.MOVEMENT_THRESHOLD
            self._history.append(1 if jumped else 0)
            self.still_streak = 0 if jumped else self.still_streak + 1
        else:
            self._history.append(0)

        self._prev_point = point
        return sum(self._history) >= config.MOVEMENT_CONFIRM_COUNT