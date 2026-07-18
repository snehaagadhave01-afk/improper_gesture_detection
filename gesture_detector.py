"""
gesture_detector.py
--------------------
Wraps MediaPipe Hands and exposes a simple `process(frame)` method that
returns per-hand results: landmarks, handedness, finger states, and the
matched gesture rule (if any).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mediapipe as mp

import config
import utils


@dataclass
class HandResult:
    handedness: str                                   # "Left" or "Right"
    landmarks_px: List[Tuple[int, int]]                # pixel coords for drawing
    landmarks_norm: List[Tuple[float, float]]          # normalized 0-1 coords (used as ML features)
    finger_state: Tuple[bool, bool, bool, bool, bool]  # (thumb,index,middle,ring,pinky)
    gesture: Optional[dict] = field(default=None)      # matched rule from config.GESTURE_RULES


class GestureDetector:
    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._mp_drawing = mp.solutions.drawing_utils

    def process(self, frame_rgb, frame_w: int, frame_h: int) -> List[HandResult]:
        """
        Run MediaPipe on an RGB frame and return a list of HandResult,
        one per detected hand.
        """
        results = self._hands.process(frame_rgb)
        hands_out: List[HandResult] = []

        if not results.multi_hand_landmarks:
            return hands_out

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label  # "Left" / "Right"

            norm_points = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            px_points = [(int(x * frame_w), int(y * frame_h)) for x, y in norm_points]

            finger_state = utils.get_finger_states(norm_points, label)
            gesture = utils.classify_gesture(finger_state, config.GESTURE_RULES)

            hands_out.append(
                HandResult(
                    handedness=label,
                    landmarks_px=px_points,
                    landmarks_norm=norm_points,
                    finger_state=finger_state,
                    gesture=gesture,
                )
            )

        return hands_out

    def close(self):
        self._hands.close()