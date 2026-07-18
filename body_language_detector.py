"""
body_language_detector.py
---------------------------
Body-language checks from shared Pose landmarks (pose_engine.PoseEngine):
arms crossed, hands in pockets (heuristic), restless/fidgeting hands,
and an "openness" score (used for standing posture / presentation mode).

Notes on the heuristics:
- "Hands in pockets" can't be seen directly -- it's inferred from wrists
  being near hip height AND poorly tracked (occluded), which is what
  happens when a hand disappears into a pocket. It's an approximation.
- "Arms crossed" and "fidgeting" both require the pattern to hold for
  several consecutive frames before triggering, so a single quick
  gesture doesn't set it off.
"""

import math
from collections import deque
from typing import Dict, List, Tuple

import config


class BodyLanguageDetector:
    def __init__(self):
        self._left_wrist_history = deque(maxlen=config.MOVEMENT_WINDOW_FRAMES)
        self._right_wrist_history = deque(maxlen=config.MOVEMENT_WINDOW_FRAMES)
        self._prev_left_wrist = None
        self._prev_right_wrist = None
        self._arms_crossed_frames = 0
        self._hands_pocket_frames = 0

    def analyze(self, landmarks, PoseLandmark, frame_w: int, frame_h: int) -> Tuple[List[Tuple[str, str]], Dict]:
        if landmarks is None:
            self._arms_crossed_frames = 0
            self._hands_pocket_frames = 0
            return [], {}

        P = PoseLandmark
        lm = landmarks

        def px(p):
            return (lm[p].x * frame_w, lm[p].y * frame_h)

        def vis(p):
            return lm[p].visibility

        required = [P.LEFT_WRIST, P.RIGHT_WRIST, P.LEFT_SHOULDER, P.RIGHT_SHOULDER, P.LEFT_HIP, P.RIGHT_HIP]
        if min(vis(p) for p in required) < 0.4:
            self._arms_crossed_frames = 0
            self._hands_pocket_frames = 0
            return [], {}

        l_wrist, r_wrist = px(P.LEFT_WRIST), px(P.RIGHT_WRIST)
        l_shoulder, r_shoulder = px(P.LEFT_SHOULDER), px(P.RIGHT_SHOULDER)
        l_hip, r_hip = px(P.LEFT_HIP), px(P.RIGHT_HIP)

        shoulder_width = math.hypot(r_shoulder[0] - l_shoulder[0], r_shoulder[1] - l_shoulder[1]) or 1.0
        chest_top = min(l_shoulder[1], r_shoulder[1])
        chest_bottom = max(l_hip[1], r_hip[1])

        issues: List[Tuple[str, str]] = []

        # --- Arms crossed: wrists swapped to the opposite side, at torso
        # height, held for several frames
        wrists_crossed = l_wrist[0] > r_shoulder[0] and r_wrist[0] < l_shoulder[0]
        wrists_at_torso = (
            chest_top - shoulder_width * 0.3 < l_wrist[1] < chest_bottom
            and chest_top - shoulder_width * 0.3 < r_wrist[1] < chest_bottom
        )
        self._arms_crossed_frames = self._arms_crossed_frames + 1 if (wrists_crossed and wrists_at_torso) else 0
        if self._arms_crossed_frames >= config.ARMS_CROSSED_CONFIRM_FRAMES:
            issues.append(("arms_crossed", "Try an open posture - uncross your arms"))

        # --- Hands in pockets (heuristic): wrists near/below hip level and
        # poorly visible (occluded), sustained for a bit
        near_hips = l_wrist[1] > l_hip[1] - shoulder_width * 0.2 and r_wrist[1] > r_hip[1] - shoulder_width * 0.2
        low_visibility = vis(P.LEFT_WRIST) < 0.35 or vis(P.RIGHT_WRIST) < 0.35
        self._hands_pocket_frames = self._hands_pocket_frames + 1 if (near_hips and low_visibility) else 0
        if self._hands_pocket_frames >= config.HANDS_POCKET_CONFIRM_FRAMES:
            issues.append(("hands_in_pockets", "Keep your hands visible and relaxed"))

        # --- Restless / fidgeting hands
        def big_jump(prev, curr):
            return prev is not None and math.hypot(curr[0] - prev[0], curr[1] - prev[1]) > config.FIDGET_MOVE_THRESHOLD

        self._left_wrist_history.append(1 if big_jump(self._prev_left_wrist, l_wrist) else 0)
        self._right_wrist_history.append(1 if big_jump(self._prev_right_wrist, r_wrist) else 0)
        self._prev_left_wrist, self._prev_right_wrist = l_wrist, r_wrist

        if sum(self._left_wrist_history) + sum(self._right_wrist_history) >= config.FIDGET_CONFIRM_COUNT:
            issues.append(("fidgeting", "Relax your hands - avoid restless movement"))

        # --- Openness score: wrist spread relative to shoulder width
        # (low = closed off / arms tucked in, high = open, relaxed stance)
        wrist_spread = math.hypot(r_wrist[0] - l_wrist[0], r_wrist[1] - l_wrist[1])
        openness = max(0, min(100, int(100 * wrist_spread / (shoulder_width * 2.2))))
        already_flagged = any(name == "arms_crossed" for name, _ in issues)
        if openness < config.LOW_OPENNESS_THRESHOLD and not already_flagged:
            issues.append(("poor_openness", "Open up your posture - relax your arms at your sides"))

        return issues, {"openness_score": openness}