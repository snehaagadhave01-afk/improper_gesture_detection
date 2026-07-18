"""
pose_engine.py
--------------
Runs MediaPipe Pose ONCE per frame and shares the raw landmarks with
both posture_detector.py and body_language_detector.py, so we don't
pay the cost of running pose estimation twice per frame.
"""

import mediapipe as mp

import config


class PoseEngine:
    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    @property
    def PoseLandmark(self):
        return self._mp_pose.PoseLandmark

    def process(self, frame_rgb):
        """Returns raw landmark list, or None if no person detected."""
        results = self._pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks.landmark

    def close(self):
        self._pose.close()